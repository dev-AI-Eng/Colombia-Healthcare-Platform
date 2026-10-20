"""The manager report, and the attendance steps that feed it.

Two things are being tested here, and the second matters more than the first.

The numbers: a funnel built from events rather than status, so an appointment
rescheduled twice and then attended is visible as both.

**The honesty of the numbers**: every percentage carries its formula, a rate
with no denominator is None rather than 0%, and a group too small to publish
is suppressed rather than rounded. A report that quietly prints 0% for a clinic
with no appointments, or a no-show rate from three patients in one barrio, is
worse than no report — it invites a decision it cannot support.
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.audit.context import AuditContext
from src.core.timezones import BOGOTA
from src.scheduling import attendance, booking, reporting
from src.scheduling.attendance import Refusal, Step
from src.scheduling.events import ACTOR_PATIENT, ACTOR_STAFF
from src.scheduling.models import Appointment, AppointmentStatus
from src.scheduling.reporting import FORMULAS, MIN_GROUP_SIZE, ReportFilter
from tests.integration.factories import Graph, appointment, at, create_graph


@pytest.fixture(autouse=True)
def _audited(staff_context: AuditContext) -> AuditContext:
    return staff_context


def _filters(graph: Graph) -> ReportFilter:
    """A range wide enough to contain everything `at()` builds."""
    today = dt.datetime.now(tz=BOGOTA).date()
    return ReportFilter(
        clinic_id=graph.clinic_id,
        start_date=today - dt.timedelta(days=30),
        end_date=today + dt.timedelta(days=60),
    )


async def _booked(session: AsyncSession, graph: Graph, day: int) -> Appointment:
    row = appointment(graph, at(day, 9, 0))
    session.add(row)
    await session.flush()
    return row


# ------------------------------------------------- the attendance steps


async def test_check_in_start_end_record_their_times(session: AsyncSession) -> None:
    graph = await create_graph(session)
    row = await _booked(session, graph, 3)
    base = dt.datetime.now(tz=dt.UTC)

    for step, offset in ((Step.CHECK_IN, 0), (Step.START, 5), (Step.END, 25)):
        result = await attendance.record(
            session,
            clinic_id=graph.clinic_id,
            appointment_id=row.id,
            step=step,
            now=base + dt.timedelta(minutes=offset),
        )
        assert result.ok, (step, result.refusal)

    await session.refresh(row)
    assert row.checked_in_at is not None
    assert row.started_at is not None
    assert row.ended_at is not None
    assert row.status is AppointmentStatus.COMPLETED


async def test_starting_before_check_in_is_refused(session: AsyncSession) -> None:
    """A consultation that starts before the patient arrived would put a
    negative delay into the punctuality report."""
    graph = await create_graph(session)
    row = await _booked(session, graph, 3)

    result = await attendance.record(
        session,
        clinic_id=graph.clinic_id,
        appointment_id=row.id,
        step=Step.START,
        now=dt.datetime.now(tz=dt.UTC),
    )
    assert not result.ok
    assert result.refusal is Refusal.NOT_ACTIVE


async def test_pressing_the_same_button_twice_is_reported_not_raised(
    session: AsyncSession,
) -> None:
    """A receptionist double-clicking has done nothing wrong."""
    graph = await create_graph(session)
    row = await _booked(session, graph, 3)
    now = dt.datetime.now(tz=dt.UTC)

    first = await attendance.record(
        session, clinic_id=graph.clinic_id, appointment_id=row.id, step=Step.CHECK_IN, now=now
    )
    second = await attendance.record(
        session, clinic_id=graph.clinic_id, appointment_id=row.id, step=Step.CHECK_IN, now=now
    )

    assert first.ok
    assert not second.ok
    assert second.refusal is Refusal.ALREADY_RECORDED


async def test_a_cancelled_appointment_cannot_be_checked_in(
    session: AsyncSession,
) -> None:
    graph = await create_graph(session)
    row = await _booked(session, graph, 3)
    await booking.cancel(session, clinic_id=graph.clinic_id, appointment_id=row.id)

    result = await attendance.record(
        session,
        clinic_id=graph.clinic_id,
        appointment_id=row.id,
        step=Step.CHECK_IN,
        now=dt.datetime.now(tz=dt.UTC),
    )
    assert not result.ok
    assert result.refusal is Refusal.NOT_ACTIVE


async def test_marking_a_no_show_records_it(session: AsyncSession) -> None:
    graph = await create_graph(session)
    row = await _booked(session, graph, 3)

    result = await attendance.record(
        session,
        clinic_id=graph.clinic_id,
        appointment_id=row.id,
        step=Step.NO_SHOW,
        now=dt.datetime.now(tz=dt.UTC),
    )
    assert result.ok
    await session.refresh(row)
    assert row.status is AppointmentStatus.NO_SHOW


# ------------------------------------------------------------ the funnel


async def test_the_funnel_counts_what_happened_not_what_is(
    session: AsyncSession,
) -> None:
    """The reason ADR-14 exists.

    This appointment was rescheduled and then its replacement was cancelled.
    Current status says `rescheduled` and `cancelled`; the funnel has to show
    both the move and the cancellation, because a manager asking "how often do
    we move appointments" cannot read that off a status column.
    """
    # `hours=True`: `reschedule` now refuses a destination the doctor does not
    # work, so a test that moves an appointment needs the doctor to have a
    # schedule. The tests here that only insert rows do not.
    graph = await create_graph(session, hours=True)
    row = await _booked(session, graph, 3)

    moved = await booking.reschedule(
        session,
        clinic_id=graph.clinic_id,
        appointment_id=row.id,
        start=at(5, 11, 0),
        end=at(5, 11, 20),
    )
    assert moved.appointment is not None
    await booking.cancel(
        session,
        clinic_id=graph.clinic_id,
        appointment_id=moved.appointment.id,
        actor=ACTOR_PATIENT,
    )

    counts = await reporting.funnel(session, _filters(graph))
    assert counts.rescheduled == 1
    assert counts.cancelled == 1


async def test_every_percentage_carries_its_formula(session: AsyncSession) -> None:
    """Rev3 requires the formula printed on the report.

    "No-show %" has at least three defensible denominators, and a manager
    comparing months needs to know it means the same thing both times.
    """
    graph = await create_graph(session)
    await _booked(session, graph, 3)

    report = await reporting.summary(session, _filters(graph))

    assert set(report.percentages) == set(report.formulas)
    for name, formula in FORMULAS.items():
        assert report.formulas[name] == formula
        assert "÷" in formula, f"{name} does not state its denominator"


async def test_a_rate_with_no_denominator_is_none_not_zero(
    session: AsyncSession,
) -> None:
    """A clinic with no appointments has no no-show rate. Printing 0% would
    read as perfect attendance."""
    graph = await create_graph(session)

    report = await reporting.summary(session, _filters(graph))

    assert report.funnel.scheduled == 0
    assert report.percentages["no_show_pct"] is None
    assert report.percentages["compliance_pct"] is None


async def test_compliance_counts_appointments_nobody_touched(
    session: AsyncSession,
) -> None:
    """The denominator is appointments in the range, not appointments with
    events. A join would drop exactly the rows that make compliance meaningful:
    the ones where nothing happened."""
    graph = await create_graph(session)
    for day in (3, 5, 7):
        await _booked(session, graph, day)

    counts = await reporting.funnel(session, _filters(graph))
    assert counts.scheduled == 3
    assert counts.completed == 0
    assert counts.percentages()["compliance_pct"] == 0.0


async def test_cancellations_split_by_who_cancelled(session: AsyncSession) -> None:
    """ "Patients cancel 30%" and "we cancel 30%" are different problems."""
    graph = await create_graph(session)
    theirs = await _booked(session, graph, 3)
    ours = await _booked(session, graph, 5)

    await booking.cancel(
        session,
        clinic_id=graph.clinic_id,
        appointment_id=theirs.id,
        reason="patient_request",
        actor=ACTOR_PATIENT,
    )
    await booking.cancel(
        session,
        clinic_id=graph.clinic_id,
        appointment_id=ours.id,
        reason="doctor_unavailable",
        actor=ACTOR_STAFF,
    )

    reasons = await reporting.cancellation_reasons(session, _filters(graph))
    by_actor = {(r.reason, r.actor): r.count for r in reasons}
    assert by_actor[("patient_request", ACTOR_PATIENT)] == 1
    assert by_actor[("doctor_unavailable", ACTOR_STAFF)] == 1


async def test_the_funnel_is_isolated_per_clinic(session: AsyncSession) -> None:
    """Row-level security (§7). One clinic's funnel must not contain
    another's rows."""
    from src.core.tenancy import ClinicScope, apply_clinic_scope

    first = await create_graph(session, name="Clínica Uno")
    row = await _booked(session, first, 3)
    await booking.cancel(session, clinic_id=first.clinic_id, appointment_id=row.id)
    await session.flush()

    second = await create_graph(session, name="Clínica Dos")
    await apply_clinic_scope(session, ClinicScope(clinic_id=second.clinic_id))

    counts = await reporting.funnel(session, _filters(second))
    assert counts.cancelled == 0
    assert counts.scheduled == 0


def test_the_suppression_threshold_is_defensible() -> None:
    """Below five, a neighbourhood rate is noise and it also identifies
    people: "2 of 3 missed" in a small barrio names them."""
    assert MIN_GROUP_SIZE >= 5


def test_every_step_is_fully_declared() -> None:
    """Adding a `Step` without its table entries is a KeyError at runtime.

    That is exactly how `NO_SHOW` shipped broken for an hour: it was in the
    enum, in `_EVENT` and in `_ALLOWED_FROM`, and missing from `_REQUIRES`, so
    the one path that used it raised. Checked here because a mapping that must
    be exhaustive and is not is invisible until a user finds it.
    """
    from src.scheduling.attendance import _ALLOWED_FROM, _BECOMES, _EVENT, _REQUIRES, Step

    for table, name in (
        (_EVENT, "_EVENT"),
        (_REQUIRES, "_REQUIRES"),
        (_ALLOWED_FROM, "_ALLOWED_FROM"),
        (_BECOMES, "_BECOMES"),
    ):
        missing = [step.value for step in Step if step not in table]
        assert not missing, f"{name} is missing {missing}"
