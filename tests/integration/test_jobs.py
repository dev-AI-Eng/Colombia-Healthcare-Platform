"""The background scheduler: its schema, its schedule, and what its task does.

What matters here is not that Procrastinate works -- it has its own tests --
but that our wiring is right: the tables land in their own schema rather than
in `app`, the runtime role can use them, the periodic task is registered on the
cron we think it is, and the task body runs the same sweeps the command does.

No worker process is started. Running one would mean waiting on a clock, and
the thing worth checking is the task body, which is callable directly.
"""

from __future__ import annotations

import datetime as dt

import psycopg
import pytest

from src.core.config import Settings
from src.scheduling import jobs, sweeper


async def test_the_schema_is_created_and_usable_by_the_runtime_role(
    settings: Settings,
) -> None:
    """Idempotent, and the API's role can reach it.

    The grants matter as much as the tables: the worker connects as the runtime
    role, and a missing GRANT on a sequence or function only shows up when a
    job is actually deferred.
    """
    await jobs.ensure_jobs_schema(settings)
    await jobs.ensure_jobs_schema(settings)  # twice: it must not fail

    with psycopg.connect(settings.conninfo(admin=True)) as conn:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT table_name FROM information_schema.tables"
                " WHERE table_schema = %s ORDER BY table_name",
                (jobs.JOBS_SCHEMA,),
            ).fetchall()
        ]
    assert "procrastinate_jobs" in tables, tables

    # The runtime role, which is what the worker connects as.
    with psycopg.connect(jobs.jobs_conninfo(settings, admin=False)) as conn:
        count = conn.execute("SELECT count(*) FROM procrastinate_jobs").fetchone()
    assert count is not None


async def test_no_procrastinate_table_lands_in_the_application_schemas(
    settings: Settings,
) -> None:
    """`app` holds hand-written, reviewed SQL only.

    `test_migrations.py` compares `app` and `audit` against the models, and
    Procrastinate's tables are not in the models. Letting them into `app` would
    mean either a failing fidelity test or an exclusion list hiding real drift.
    """
    await jobs.ensure_jobs_schema(settings)

    with psycopg.connect(settings.conninfo(admin=True)) as conn:
        strays = conn.execute(
            "SELECT table_schema, table_name FROM information_schema.tables"
            " WHERE table_name LIKE 'procrastinate%%'"
            " AND table_schema IN ('app', 'audit', 'onboarding', 'public')"
        ).fetchall()
    assert strays == [], strays


def test_the_sweep_is_registered_as_a_periodic_task(settings: Settings) -> None:
    """The schedule is the deliverable; a task nobody scheduled runs never."""
    app = jobs.register(jobs.build_app(settings))

    assert "housekeeping.sweep" in app.tasks

    periodic = app.periodic_registry.periodic_tasks
    assert len(periodic) == 1, periodic
    (registered,) = periodic.values()
    assert registered.task.name == "housekeeping.sweep"
    # The cron the module documents, not whatever was last typed.
    assert str(registered.cron) == jobs.SWEEP_CRON


def test_the_cron_fires_every_five_minutes(settings: Settings) -> None:
    """Read back from the parsed schedule rather than trusting the string."""
    app = jobs.register(jobs.build_app(settings))
    (registered,) = app.periodic_registry.periodic_tasks.values()

    at = dt.datetime(2026, 10, 10, 9, 0, tzinfo=dt.UTC).timestamp()
    nxt = registered.croniter.get_next(start_time=at)

    assert nxt - at == 5 * 60


async def test_the_task_body_runs_the_same_sweeps_as_the_command(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker and the command must not drift apart.

    Both call `sweeper.run_once`. If the task body grew its own copy of the
    sweep logic, this would be the test that noticed.
    """
    calls: list[Settings] = []

    async def record(s: Settings, **kwargs: object) -> sweeper.SweepReport:
        calls.append(s)
        return sweeper.SweepReport(holds=0, threads=0, clinics=1)

    monkeypatch.setattr(sweeper, "run_once", record)

    app = jobs.register(jobs.build_app(settings))
    task = app.tasks["housekeeping.sweep"]
    await task(timestamp=int(dt.datetime.now(dt.UTC).timestamp()))

    assert len(calls) == 1


def test_the_worker_watches_the_queue_the_sweep_is_on(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker on the wrong queue is a worker that does nothing, silently.

    Checked by running `main("worker")` with the worker stubbed and comparing
    the queues it asked for against the queue the task is actually registered
    on. Comparing both against `SWEEP_QUEUE` would compare a constant with
    itself and pass however the two drifted apart.
    """
    asked: dict[str, object] = {}

    def fake_run_worker(self, **kwargs: object) -> None:  # type: ignore[no-untyped-def]
        asked.update(kwargs)

    monkeypatch.setattr(jobs.App, "run_worker", fake_run_worker)
    assert jobs.main(["worker"]) == 0

    app = jobs.register(jobs.build_app(settings))
    (registered,) = app.periodic_registry.periodic_tasks.values()
    task_queue = registered.task.queue

    assert asked["queues"] == [task_queue], (
        f"the worker listens on {asked['queues']} but the sweep is queued on "
        f"{task_queue!r}, so nothing would ever run it"
    )
