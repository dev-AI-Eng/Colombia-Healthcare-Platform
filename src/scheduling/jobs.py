"""The background scheduler: Procrastinate, with M2's sweeps as periodic tasks.

    Rev2, M2 deliverable: "Background scheduler wired to APScheduler/Celery
    beat."

Procrastinate rather than either of those, for the reason the stack table gives:
it keeps its queue in the PostgreSQL database we already run, so there is no
Redis or RabbitMQ to deploy, and it speaks psycopg 3, which is the one driver
this project uses. APScheduler holds its schedule in memory unless given a
store, and Celery needs a broker. The deliverable asks for a background
scheduler; this is one, on the database that is already there.

HOW IT RUNS
-----------
    python -m src.scheduling.jobs worker     # the long-running process
    python -m src.scheduling.jobs schema     # create its tables, once

The worker claims due jobs and runs them. The schedule lives in the database,
so restarting the worker does not lose it and two workers do not run the same
sweep twice -- Procrastinate's periodic deferral is keyed on the task and the
timestamp, so a given minute's run is deferred exactly once however many
workers are watching.

`python -m src.scheduling.sweeper` still works and does the same thing in one
pass. It is what to reach for in development, in a cron entry, or to catch up
after the worker has been down; this module is what to deploy. Both call the
same functions, so neither can drift from the other.

ITS TABLES LIVE IN THEIR OWN SCHEMA
-----------------------------------
`jobs`, reached through a `search_path`, the same arrangement the LangGraph
checkpointer uses for `conversation`. Procrastinate ships its own SQL and
tracks its own versions, so putting its tables in `app` would mix them with the
hand-written Alembic migrations that `test_migrations.py` compares against the
models. A separate schema keeps "the reviewed SQL is the SQL that runs" true of
everything in `app`, and keeps the fidelity test honest without an exclusion
list.

WHAT IT DOES NOT DO YET
-----------------------
No outbox. The work that genuinely needs a job queue is message dispatch -- a
reminder that must be retried until it succeeds and must not be lost if the
process dies mid-send -- and that arrives with the channel layer in M4. This
module exists so that when it does, the queue, the worker and the deployment
story are already running and proven, rather than being introduced alongside
the first thing that depends on them.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt

from procrastinate import App, PsycopgConnector
from psycopg import AsyncConnection, sql
from psycopg.conninfo import make_conninfo

from src.core.config import Settings, get_settings
from src.core.db import configure_event_loop_policy
from src.core.logging import get_logger
from src.scheduling import sweeper

log = get_logger(__name__)

#: Procrastinate's own tables. Not `app`: see the module docstring.
JOBS_SCHEMA = "jobs"

#: Every few minutes. Holds expire on a 15-minute default, so a lapsed hold is
#: cleared well inside the window where a patient might retry, and an
#: unrecorded attendance is noticed the same evening.
SWEEP_CRON = "*/5 * * * *"

#: The queue name, so a later outbox worker can be given its own.
SWEEP_QUEUE = "housekeeping"


def jobs_conninfo(settings: Settings, *, admin: bool) -> str:
    """A connection string whose `search_path` points at the jobs schema."""
    return make_conninfo(settings.conninfo(admin=admin), options=f"-c search_path={JOBS_SCHEMA}")


def build_app(settings: Settings, *, admin: bool = False) -> App:
    """The Procrastinate app. Built per call rather than at import.

    At import it would read settings before `main.py` has loaded the
    environment, and a module-level connection is the thing that makes a test
    suite reach a database it was not given.
    """
    return App(connector=PsycopgConnector(conninfo=jobs_conninfo(settings, admin=admin)))


async def ensure_jobs_schema(settings: Settings) -> None:
    """Create the schema and Procrastinate's tables, and grant the runtime role.

    Idempotent, but not because Procrastinate's own call is: `apply_schema` is
    a plain CREATE run, and a second call raises `DuplicateObject` on the first
    type it tries to create. So the tables are applied only when they are not
    there, and the grants are reapplied every time -- those are `GRANT`, which
    is idempotent, and reapplying them is what picks up a table added by a
    Procrastinate upgrade.

    Called from `main.py` alongside the checkpoint schema, so a clean checkout
    still comes up with one command.
    """
    schema = sql.Identifier(JOBS_SCHEMA)
    role = sql.Identifier(settings.app_db_user)

    async with await AsyncConnection.connect(
        settings.conninfo(admin=True), autocommit=True
    ) as conn:
        await conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(schema))
        already = await (
            await conn.execute(
                "SELECT to_regclass(%s) IS NOT NULL",
                (f"{JOBS_SCHEMA}.procrastinate_jobs",),
            )
        ).fetchone()

    if not (already and already[0]):
        # The owner creates the tables; the runtime role only uses them.
        app = build_app(settings, admin=True)
        async with app.open_async():
            await app.schema_manager.apply_schema_async()

    async with await AsyncConnection.connect(
        settings.conninfo(admin=True), autocommit=True
    ) as conn:
        await conn.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(schema, role))
        await conn.execute(
            sql.SQL("GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA {} TO {}").format(
                schema, role
            )
        )
        # Procrastinate's job ids come from sequences, and its queries call its
        # own functions and procedures.
        await conn.execute(
            sql.SQL("GRANT USAGE ON ALL SEQUENCES IN SCHEMA {} TO {}").format(schema, role)
        )
        await conn.execute(
            sql.SQL("GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA {} TO {}").format(schema, role)
        )
        await conn.execute(
            sql.SQL("GRANT EXECUTE ON ALL PROCEDURES IN SCHEMA {} TO {}").format(schema, role)
        )


def register(app: App) -> App:
    """Attach the periodic tasks to an app.

    Separate from `build_app` so a test can register against an app it
    controls, and so the task bodies are reachable without a worker.

    The sweep takes Procrastinate's `timestamp` argument and ignores it in
    favour of the clock inside `run_once`: the timestamp is the minute the job
    was *scheduled* for, and a job that waited in the queue would otherwise
    sweep against a moment that has passed. Nothing here depends on the exact
    instant, but a sweep that reasons about "before today" should use the real
    now, not the intended one.
    """

    @app.periodic(cron=SWEEP_CRON, queue=SWEEP_QUEUE)
    @app.task(name="housekeeping.sweep", queue=SWEEP_QUEUE)
    async def sweep(timestamp: int) -> None:
        settings = get_settings()
        report = await sweeper.run_once(settings)
        log.info(
            "jobs.sweep",
            scheduled_for=dt.datetime.fromtimestamp(timestamp, dt.UTC).isoformat(),
            expired_holds_removed=report.holds,
            checkpoint_threads_removed=report.threads,
            reminders_due=report.reminders_due,
            attendance_flagged=report.attendance_flagged,
        )

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.scheduling.jobs",
        description=(
            "The background scheduler. `schema` creates its tables once; "
            "`worker` runs the process that executes the periodic sweeps."
        ),
    )
    parser.add_argument("command", choices=("worker", "schema"))
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="How many jobs to run at once. One is right while the only job is a sweep.",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_event_loop_policy()

    if args.command == "schema":
        asyncio.run(ensure_jobs_schema(settings))
        print(f"Procrastinate schema applied to '{JOBS_SCHEMA}'.")
        return 0

    app = register(build_app(settings))
    print(
        f"Worker starting. The sweep runs on '{SWEEP_CRON}' "
        f"(queue '{SWEEP_QUEUE}'). Ctrl+C to stop."
    )
    app.run_worker(queues=[SWEEP_QUEUE], concurrency=args.concurrency)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through `main`
    raise SystemExit(main())
