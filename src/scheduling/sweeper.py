"""Periodic housekeeping: expired holds, and conversation checkpoints.

    python -m src.scheduling.sweeper

Run it on a schedule -- Task Scheduler on Windows, cron elsewhere -- every few
minutes. Each run is independent and takes no lock beyond the rows it touches,
so two overlapping runs are harmless: the second finds nothing to do.

WHY A COMMAND RATHER THAN A JOB FRAMEWORK
-----------------------------------------
The stack lists Procrastinate for M2's background work, and that is still the
right call for the one thing that genuinely needs it: a transactional outbox,
where a dispatch must be retried until it succeeds and must not be lost if the
process dies mid-send. That arrives with the channel layer.

What M2 actually has is two sweeps that delete rows nobody is waiting on. A
missed run costs nothing -- the next one catches up, and neither sweep is load
bearing, because `availability` already treats an expired hold as free and the
retention window is measured in days. Adding a worker process, its tables and
its migration to delete rows on a timer would be more moving parts than the
problem has.

WHAT THIS IS NOT
----------------
It is not what makes an expired hold release its slot. That happens the moment
`expires_at` passes, in `availability.free_windows`, whether or not this has
run. The sweep keeps the table from growing without bound and keeps a schedule
a human reads in the database honest; correctness does not depend on it.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.audit.context import ActorKind, AuditContext, audit_context
from src.conversation.checkpointer import sweep_expired_threads
from src.core.config import Settings, get_settings
from src.core.db import configure_event_loop_policy, get_sessionmaker
from src.core.logging import get_logger
from src.core.tenancy import ClinicScope, apply_clinic_scope
from src.registry.models import Clinic
from src.scheduling.booking import sweep_expired_holds

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SweepReport:
    """What one run removed."""

    holds: int
    threads: int
    clinics: int


async def sweep_holds_for_every_clinic(
    session: AsyncSession, *, now: dt.datetime
) -> tuple[int, int]:
    """Clear expired holds across every clinic. Returns (holds, clinics).

    Row-level security scopes each delete to one clinic, so the clinics are
    walked rather than swept in a single statement: a job that could reach
    across clinics would be the one piece of code able to, and that is not a
    privilege worth creating for housekeeping.

    Flushes but does not commit. The clinic scope is transaction-local, so
    committing here would clear it mid-walk and leave the next clinic's delete
    seeing nothing; `run_once` owns the transaction and commits once at the
    end. A sweep that is interrupted simply rolls back and the next run
    repeats it, which costs nothing -- these rows are already past their time.
    """
    clinic_ids = list((await session.scalars(select(Clinic.id))).all())
    removed = 0
    for clinic_id in clinic_ids:
        await apply_clinic_scope(session, ClinicScope(clinic_id=clinic_id))
        removed += await sweep_expired_holds(session, clinic_id=clinic_id, now=now)
    return removed, len(clinic_ids)


async def run_once(
    settings: Settings, *, now: dt.datetime | None = None, holds_only: bool = False
) -> SweepReport:
    """One pass of every sweep. Safe to call concurrently with itself."""
    moment = now or dt.datetime.now(dt.UTC)
    context = AuditContext(
        actor_kind=ActorKind.SYSTEM, actor_id="sweeper", purpose="scheduled_housekeeping"
    )
    with audit_context(context):
        async with get_sessionmaker()() as session:
            holds, clinics = await sweep_holds_for_every_clinic(session, now=moment)
            await session.commit()

    # Checkpoints are keyed by thread, not by clinic, and the checkpointer owns
    # its own connection; it is swept outside the ORM session for that reason.
    threads = 0 if holds_only else await sweep_expired_threads(settings)

    report = SweepReport(holds=holds, threads=threads, clinics=clinics)
    log.info(
        "sweeper.run",
        expired_holds_removed=report.holds,
        checkpoint_threads_removed=report.threads,
        clinics=report.clinics,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.scheduling.sweeper",
        description="Remove expired slot holds and conversation checkpoints past retention.",
    )
    parser.add_argument(
        "--holds-only",
        action="store_true",
        help="Sweep expired holds and leave conversation checkpoints alone.",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_event_loop_policy()

    report = asyncio.run(run_once(settings, holds_only=args.holds_only))

    print(
        f"Removed {report.holds} expired hold(s) across {report.clinics} clinic(s) "
        f"and {report.threads} conversation thread(s) past retention."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through `main`
    raise SystemExit(main())
