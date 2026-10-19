"""Regression tests for the client's QA audit findings.

Each test reproduces a finding the reviewer raised, and each was confirmed to
fail against the code as submitted before the corresponding fix landed. They
are grouped by finding so a reviewer can map a test back to the report.

Two deliberate choices. The scenarios are the reviewer's own, not easier
versions of them -- the expired-hold test skips the sweeper exactly as the
report demands, and the cross-clinic test runs as the runtime role rather than
the migration owner. And where a finding was a *missing* test rather than a
defect, the test fails if the behaviour regresses, not merely when the happy
path breaks.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import uuid

import psycopg
import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.audit.context import AuditContext
from src.audit.models import AccessLogEntry
from src.core.config import Settings
from src.core.tenancy import ClinicScope, apply_clinic_scope
from src.core.timezones import BOGOTA
from src.scheduling import availability, booking, offers, reminders
from src.scheduling.booking import Outcome
from src.scheduling.models import (
    Appointment,
    AppointmentStatus,
    AvailabilityException,
    AvailabilityRule,
    Escalation,
    EscalationReason,
    ExceptionKind,
)
from tests.integration.factories import Graph, appointment, at, create_graph


@pytest.fixture(autouse=True)
def _audited(staff_context: AuditContext) -> AuditContext:
    return staff_context


async def _clear_rules(session: AsyncSession, graph: Graph) -> None:
    """Remove the factory's wide weekly hours.

    `create_graph` gives the doctor 06:00-22:00 every day so the booking tests
    are not all about availability. A test about availability itself has to
    start from a doctor with no schedule.
    """
    await session.execute(
        delete(AvailabilityRule).where(
            AvailabilityRule.clinic_id == graph.clinic_id,
            AvailabilityRule.doctor_id == graph.doctor_id,
        )
    )
    await session.flush()


def _maker(settings: Settings) -> async_sessionmaker[AsyncSession]:
    engine = create_async_engine(settings.app_database_url, poolclass=NullPool)
    return async_sessionmaker(engine, expire_on_commit=False)


# ========================================================= finding 2: expired holds


@pytest.mark.concurrency
async def test_an_expired_hold_does_not_block_a_booking_without_the_sweeper(
    session: AsyncSession, settings: Settings
) -> None:
    """QA finding 2, in the reviewer's own words: "skip housekeeping".

    `availability.free_windows` stops counting a hold once it has expired, but
    the exclusion constraint counts every `hold` row whatever its expiry. So a
    slot the availability API advertised as free was refused by the booking API
    until the five-minute sweep caught up. Confirmed against the submitted
    code: the booking came back TAKEN.

    The fix cannot live in the constraint. An exclusion predicate may only call
    IMMUTABLE functions and `now()` is STABLE; PostgreSQL rejects it with
    "functions in index predicate must be marked IMMUTABLE". So `book()`
    reclaims the expired hold inside its own transaction instead.

    Across connections, because that is the real situation: one conversation
    held the slot and committed, then lapsed.
    """
    graph = await create_graph(session, hours=True)
    start = at(6, 9, 0)
    end = start + dt.timedelta(minutes=20)

    held = await booking.hold(
        session,
        clinic_id=graph.clinic_id,
        patient_id=graph.patient_id,
        doctor_id=graph.doctor_id,
        location_id=graph.location_id,
        appointment_type_id=graph.appointment_type_id,
        start=start,
        end=end,
        now=at(6, 8, 0),
        minutes=15,
    )
    assert held.appointment is not None
    assert held.appointment.expires_at is not None
    expires_at = held.appointment.expires_at
    await session.commit()

    # 08:30: the hold lapsed at 08:15 and nothing has swept it.
    now = at(6, 8, 30)
    assert expires_at < now, "the hold under test has not expired"

    # Availability already calls this free, which is the inconsistency.
    free = availability.free_windows(
        [availability.Window(start.date(), dt.time(9, 0), dt.time(10, 0))],
        [availability.Busy(start, end, expires_at=expires_at)],
        now=now,
    )
    assert free, "availability should already ignore an expired hold"

    maker = _maker(settings)
    try:
        async with maker() as other:
            await apply_clinic_scope(other, ClinicScope(clinic_id=graph.clinic_id))
            result = await booking.book(
                other,
                clinic_id=graph.clinic_id,
                patient_id=graph.patient_id,
                doctor_id=graph.doctor_id,
                location_id=graph.location_id,
                appointment_type_id=graph.appointment_type_id,
                start=start,
                end=end,
                now=now,
            )
            assert result.appointment is not None, (
                f"an expired hold still blocked the booking ({result.outcome})"
            )
            assert result.appointment.status is AppointmentStatus.SCHEDULED
            await other.rollback()
    finally:
        await maker.kw["bind"].dispose()


async def test_a_live_hold_still_blocks_a_booking(session: AsyncSession) -> None:
    """The other half of the fix, and the one that matters more.

    Reclaiming expired holds must not reclaim live ones: a hold exists to stop
    exactly this booking while the patient reads their WhatsApp message.
    """
    graph = await create_graph(session, hours=True)
    start = at(6, 11, 0)
    end = start + dt.timedelta(minutes=20)
    now = at(6, 10, 50)

    held = await booking.hold(
        session,
        clinic_id=graph.clinic_id,
        patient_id=graph.patient_id,
        doctor_id=graph.doctor_id,
        location_id=graph.location_id,
        appointment_type_id=graph.appointment_type_id,
        start=start,
        end=end,
        now=now,
        minutes=15,
    )
    assert held.appointment is not None

    result = await booking.book(
        session,
        clinic_id=graph.clinic_id,
        patient_id=graph.patient_id,
        doctor_id=graph.doctor_id,
        location_id=graph.location_id,
        appointment_type_id=graph.appointment_type_id,
        start=start,
        end=end,
        now=now,
    )
    assert result.appointment is None, "a live hold was reclaimed"
    assert result.outcome is Outcome.TAKEN


# ========================================================= finding 6: buffers


def test_a_buffer_is_reserved_between_slots_not_only_inside_the_span() -> None:
    """QA finding 6, measured rather than argued.

    Against the submitted code, a 30-minute appointment with a 10-minute buffer
    produced slots at 09:00, 09:30 and 10:00 -- measured gaps of 0.0 minutes.
    The buffer was required to *fit* inside the span but never reserved, so the
    clinic advertised back-to-back consultations with no turnaround.
    """
    opens = dt.datetime(2026, 11, 2, 9, 0, tzinfo=BOGOTA)
    slots = availability.slots_in(
        [(opens, opens + dt.timedelta(hours=2))],
        duration_minutes=30,
        buffer_minutes=10,
    )
    assert len(slots) >= 2, "not enough slots to measure a gap"

    gaps = [(slots[i + 1].start - slots[i].end).total_seconds() / 60 for i in range(len(slots) - 1)]
    assert all(gap >= 10 for gap in gaps), f"the buffer is not reserved between slots: {gaps}"


def test_a_buffer_after_an_existing_appointment_pushes_the_next_start() -> None:
    """QA finding 6's second half, in the reviewer's exact numbers.

    "Verify a 09:00-09:30 appointment with a 10-minute buffer prevents a 09:30
    start but permits 09:40 when the remaining schedule allows it."

    The buffer is a gap the doctor needs *after* a consultation, so it comes
    out of the free time, not only out of the candidate. Before the fix the
    free span began at 09:30 and 09:30 was offered.
    """
    # A Tuesday. The 2nd is a holiday in Colombia -- All Souls', moved to the
    # Monday by the Ley Emiliani -- and open_windows closes the day, which
    # would make this test pass for the wrong reason.
    day = dt.date(2026, 11, 3)
    nine = dt.datetime(2026, 11, 3, 9, 0, tzinfo=BOGOTA)
    slots = availability.bookable_slots(
        start_date=day,
        end_date=day,
        rules=[
            availability.Rule(
                weekday=day.weekday(),
                start=dt.time(9, 0),
                end=dt.time(12, 0),
                valid_from=dt.date(2026, 1, 1),
            )
        ],
        busy=[availability.Busy(nine, nine + dt.timedelta(minutes=30))],
        duration_minutes=30,
        buffer_minutes=10,
        now=dt.datetime(2026, 11, 1, 9, 0, tzinfo=BOGOTA),
    )
    offered = [s.start.astimezone(BOGOTA).strftime("%H:%M") for s in slots]
    assert "09:30" not in offered, f"09:30 was offered with no turnaround: {offered}"
    assert "09:40" in offered, f"09:40 should still be offered: {offered}"


def test_a_zero_buffer_still_allows_back_to_back_appointments() -> None:
    """The buffer fix must not invent a gap nobody asked for. A clinic with no
    buffer configured runs appointments end to end, which is why the exclusion
    constraint uses half-open ranges."""
    opens = dt.datetime(2026, 11, 2, 9, 0, tzinfo=BOGOTA)
    slots = availability.slots_in(
        [(opens, opens + dt.timedelta(hours=1))],
        duration_minutes=30,
        buffer_minutes=0,
    )
    assert len(slots) == 2
    assert slots[1].start == slots[0].end


def test_an_explicit_step_still_overrides_the_buffer() -> None:
    """A clinic that asks for overlapping candidates every 15 minutes still gets
    them: `step_minutes` is the caller's explicit choice and the buffer must not
    quietly widen it. Only one candidate survives booking, which is the
    documented intent."""
    opens = dt.datetime(2026, 11, 2, 9, 0, tzinfo=BOGOTA)
    slots = availability.slots_in(
        [(opens, opens + dt.timedelta(hours=1))],
        duration_minutes=30,
        buffer_minutes=10,
        step_minutes=15,
    )
    assert [s.start.astimezone(BOGOTA).strftime("%H:%M") for s in slots] == ["09:00", "09:15"]


# ========================================================= finding 4: clinic ownership


async def test_an_appointment_cannot_reference_another_clinics_doctor(
    session: AsyncSession, settings: Settings
) -> None:
    """QA finding 4, which the reviewer could only call a static risk.

    It was demonstrable, and it was demonstrated: as the runtime role scoped to
    clinic A, an appointment in clinic A referencing clinic B's doctor was
    accepted. Row-level security compares the row's own `clinic_id` against the
    scope and never looks at what the row points at, so an id-only foreign key
    proved existence and nothing more.

    Run through psycopg as the runtime role, as the report asks -- the owner
    account bypasses row-level security and would prove less.
    """
    mine = await create_graph(session, name="Clínica Propia", hours=True)
    theirs = await create_graph(session, name="Clínica Vecina", hours=True)
    await session.commit()

    start = at(7, 9, 0)
    with psycopg.connect(settings.conninfo(admin=False)) as conn:
        conn.execute(f"SET app.clinic_id = '{mine.clinic_id}'")
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            conn.execute(
                "INSERT INTO app.appointments "
                "(id, clinic_id, patient_id, doctor_id, location_id, "
                " appointment_type_id, during, status) "
                "VALUES (%s,%s,%s,%s,%s,%s, tstzrange(%s,%s,'[)'), 'scheduled')",
                (
                    uuid.uuid4(),
                    mine.clinic_id,
                    mine.patient_id,
                    theirs.doctor_id,  # another clinic's doctor
                    mine.location_id,
                    mine.appointment_type_id,
                    start,
                    start + dt.timedelta(minutes=20),
                ),
            )
        conn.rollback()


async def test_a_booking_within_one_clinic_still_works(session: AsyncSession) -> None:
    """The composite keys must not break the ordinary case."""
    graph = await create_graph(session, hours=True)
    result = await booking.book(
        session,
        clinic_id=graph.clinic_id,
        patient_id=graph.patient_id,
        doctor_id=graph.doctor_id,
        location_id=graph.location_id,
        appointment_type_id=graph.appointment_type_id,
        start=at(8, 9, 0),
        end=at(8, 9, 20),
    )
    assert result.appointment is not None, result.outcome


# ========================================================= finding 1: concurrent reschedule


@pytest.mark.concurrency
async def test_two_sessions_rescheduling_one_appointment_admit_exactly_one(
    session: AsyncSession, settings: Settings
) -> None:
    """QA finding 1: M2's exit criterion, raced on the path that was missing one.

    The reviewer was right that the existing concurrency test races `book()`
    rather than `reschedule()`, so the criterion was never demonstrated for a
    move. This races two reschedules of the *same* original to different
    destinations: exactly one may win, and exactly one active appointment may
    remain afterwards.
    """
    graph = await create_graph(session, hours=True)
    original = appointment(graph, at(9, 9, 0))
    session.add(original)
    await session.flush()
    original_id = original.id
    await session.commit()

    maker = _maker(settings)

    async def move(hour: int) -> Outcome:
        async with maker() as other:
            await apply_clinic_scope(other, ClinicScope(clinic_id=graph.clinic_id))
            try:
                result = await booking.reschedule(
                    other,
                    clinic_id=graph.clinic_id,
                    appointment_id=original_id,
                    start=at(9, hour, 0),
                    end=at(9, hour, 20),
                )
                await other.commit()
                return result.outcome
            except Exception:
                await other.rollback()
                return Outcome.TAKEN

    try:
        outcomes = await asyncio.gather(move(14), move(16))
    finally:
        await maker.kw["bind"].dispose()

    assert outcomes.count(Outcome.BOOKED) == 1, f"{outcomes}: exactly one move may win"

    await apply_clinic_scope(session, ClinicScope(clinic_id=graph.clinic_id))
    active = (
        await session.execute(
            select(func.count())
            .select_from(Appointment)
            .where(
                Appointment.clinic_id == graph.clinic_id,
                Appointment.status.in_(booking.ACTIVE_STATUSES),
            )
        )
    ).scalar_one()
    assert active == 1, f"{active} active appointments remain; exactly one may"


@pytest.mark.concurrency
async def test_two_appointments_cannot_reschedule_into_the_same_slot(
    session: AsyncSession, settings: Settings
) -> None:
    """The reviewer's second scenario: different originals, one destination."""
    graph = await create_graph(session, hours=True)
    first = appointment(graph, at(10, 9, 0))
    second = appointment(graph, at(10, 11, 0))
    session.add_all([first, second])
    await session.flush()
    ids = (first.id, second.id)
    await session.commit()

    maker = _maker(settings)
    destination = at(10, 15, 0)

    async def move(appointment_id: uuid.UUID) -> Outcome:
        async with maker() as other:
            await apply_clinic_scope(other, ClinicScope(clinic_id=graph.clinic_id))
            try:
                result = await booking.reschedule(
                    other,
                    clinic_id=graph.clinic_id,
                    appointment_id=appointment_id,
                    start=destination,
                    end=destination + dt.timedelta(minutes=20),
                )
                await other.commit()
                return result.outcome
            except Exception:
                await other.rollback()
                return Outcome.TAKEN

    try:
        outcomes = await asyncio.gather(*(move(i) for i in ids))
    finally:
        await maker.kw["bind"].dispose()

    assert outcomes.count(Outcome.BOOKED) == 1, (
        f"{outcomes}: both reschedules took the same destination slot"
    )


# ========================================================= finding 10: escalation uniqueness


@pytest.mark.concurrency
async def test_two_concurrent_attendance_sweeps_raise_one_escalation(
    session: AsyncSession, settings: Settings
) -> None:
    """QA finding 10, raced rather than run twice in sequence.

    The sweep reads the existing escalations and then inserts for the rest, so
    two overlapping sweeps could both see "none" and both insert -- the
    receptionist is then asked the same question twice. Running the sweep twice
    in one session never showed this, because the second call saw the first
    call's flush.

    The partial unique index makes the second transaction fail instead, which
    is the correct outcome: the other sweep won and the question has been asked.
    """
    graph = await create_graph(session, hours=True)
    past = appointment(graph, at(1, 9, 0))
    session.add(past)
    await session.flush()
    appointment_id = past.id
    await session.commit()

    maker = _maker(settings)
    now = at(5, 9, 0)  # the appointment ended before today began

    async def sweep() -> int:
        async with maker() as other:
            await apply_clinic_scope(other, ClinicScope(clinic_id=graph.clinic_id))
            try:
                raised = await reminders.flag_unrecorded_attendance(
                    other, clinic_id=graph.clinic_id, now=now
                )
                await other.commit()
                return raised
            except Exception:
                await other.rollback()
                return 0

    try:
        await asyncio.gather(sweep(), sweep())
    finally:
        await maker.kw["bind"].dispose()

    await apply_clinic_scope(session, ClinicScope(clinic_id=graph.clinic_id))
    count = (
        await session.execute(
            select(func.count())
            .select_from(Escalation)
            .where(
                Escalation.clinic_id == graph.clinic_id,
                Escalation.reason == EscalationReason.ATTENDANCE_UNRECORDED,
                Escalation.appointment_id == appointment_id,
            )
        )
    ).scalar_one()
    assert count == 1, f"{count} attendance escalations exist for one appointment; exactly one may"


# ========================================================= finding 8: exception-only availability


async def test_a_doctor_with_only_an_extra_hours_exception_is_offerable(
    session: AsyncSession,
) -> None:
    """QA finding 8: `find_slots` returned [] before loading the exceptions.

    A doctor with no recurring weekly schedule but an explicit extra-hours
    window had that window silently ignored, because the early return fired on
    the empty rule list. A locum covering one Saturday is exactly this shape.
    """
    graph = await create_graph(session)
    when = dt.date(2026, 11, 7)  # a Saturday; the only hours are the exception
    session.add(
        AvailabilityException(
            clinic_id=graph.clinic_id,
            doctor_id=graph.doctor_id,
            on_date=when,
            kind=ExceptionKind.EXTRA_HOURS,
            start_time=dt.time(9, 0),
            end_time=dt.time(12, 0),
        )
    )
    await session.flush()

    found = await offers.find_slots(
        session,
        offers.SlotSearch(
            clinic_id=graph.clinic_id,
            doctor_id=graph.doctor_id,
            appointment_type_id=graph.appointment_type_id,
            start_date=when,
            end_date=when,
        ),
        now=dt.datetime(2026, 11, 1, 9, 0, tzinfo=BOGOTA),
    )
    assert found, "an extra-hours exception with no weekly rules offered nothing"


async def test_a_doctor_with_no_schedule_at_all_still_offers_nothing(
    session: AsyncSession,
) -> None:
    """The early return it replaced was not wrong, only too early. A doctor with
    neither rules nor exceptions must still offer nothing rather than falling
    back to some default day."""
    graph = await create_graph(session)
    when = dt.date(2026, 11, 9)
    found = await offers.find_slots(
        session,
        offers.SlotSearch(
            clinic_id=graph.clinic_id,
            doctor_id=graph.doctor_id,
            appointment_type_id=graph.appointment_type_id,
            start_date=when,
            end_date=when,
        ),
        now=dt.datetime(2026, 11, 1, 9, 0, tzinfo=BOGOTA),
    )
    assert found == []


# ========================================================= finding 7: duplicate offers


async def test_overlapping_rules_do_not_produce_duplicate_slots(
    session: AsyncSession,
) -> None:
    """QA finding 7: open windows were appended without merging, so two
    overlapping rules generated the same start time twice. The patient was then
    offered what looked like several choices and was really one, and an offer
    that tried to hold the duplicate lost to the constraint."""
    graph = await create_graph(session)
    target = dt.date(2026, 11, 10)  # a Tuesday

    for start_hour, end_hour in ((8, 12), (10, 14)):
        session.add(
            AvailabilityRule(
                clinic_id=graph.clinic_id,
                doctor_id=graph.doctor_id,
                location_id=graph.location_id,
                weekday=target.weekday(),
                start_time=dt.time(start_hour, 0),
                end_time=dt.time(end_hour, 0),
                valid_from=dt.date(2026, 1, 1),
            )
        )
    await session.flush()

    found = await offers.find_slots(
        session,
        offers.SlotSearch(
            clinic_id=graph.clinic_id,
            doctor_id=graph.doctor_id,
            appointment_type_id=graph.appointment_type_id,
            start_date=target,
            end_date=target,
        ),
        now=dt.datetime(2026, 11, 1, 9, 0, tzinfo=BOGOTA),
        limit=100,
    )
    starts = [slot.start for slot in found]
    assert len(starts) == len(set(starts)), "overlapping rules produced duplicate slots"


# ========================================================= findings 11 and 12: audit


async def test_rescheduling_audits_both_appointments(session: AsyncSession) -> None:
    """QA finding 11: the replacement's creation was recorded, the original's
    release was not -- so an auditor reading the log could not see that the
    first appointment had been given up, only that a second appeared."""
    graph = await create_graph(session, hours=True)
    original = appointment(graph, at(11, 9, 0))
    session.add(original)
    await session.flush()
    original_id = original.id

    result = await booking.reschedule(
        session,
        clinic_id=graph.clinic_id,
        appointment_id=original_id,
        start=at(11, 15, 0),
        end=at(11, 15, 20),
    )
    assert result.appointment is not None, result.outcome

    recorded = set(
        (
            await session.scalars(
                select(AccessLogEntry.resource_id).where(AccessLogEntry.resource == "appointments")
            )
        ).all()
    )
    assert str(original_id) in recorded, "the original's release was not audited"
    assert str(result.appointment.id) in recorded, "the replacement was not audited"


async def test_an_attendance_audit_entry_names_the_escalation_it_created(
    session: AsyncSession,
) -> None:
    """QA finding 12: the entry said `resource="escalations"` while carrying the
    *appointment* id, so an investigator asked to produce escalation X could not
    find the access that created it."""
    graph = await create_graph(session, hours=True)
    past = appointment(graph, at(1, 9, 0))
    session.add(past)
    await session.flush()

    raised = await reminders.flag_unrecorded_attendance(
        session, clinic_id=graph.clinic_id, now=at(5, 9, 0)
    )
    assert raised == 1

    escalation_ids = {
        str(row)
        for row in (
            await session.scalars(
                select(Escalation.id).where(Escalation.clinic_id == graph.clinic_id)
            )
        ).all()
    }
    logged = set(
        (
            await session.scalars(
                select(AccessLogEntry.resource_id).where(AccessLogEntry.resource == "escalations")
            )
        ).all()
    )
    assert logged, "no escalation access was recorded at all"
    assert logged <= escalation_ids, (
        f"an escalation audit entry names something that is not an escalation: "
        f"{logged - escalation_ids}"
    )


# ========================================================= finding 3: eligibility


async def _schedule(session: AsyncSession, graph: Graph, day: dt.date) -> None:
    """Narrow the graph's doctor to 08:00-12:00 on `day`'s weekday.

    `create_graph` gives every doctor 06:00-22:00 all week, so that the booking
    tests are not all about availability. These tests are about exactly that,
    so they delete the factory's rules first -- otherwise 15:00 falls inside
    the wide window and the test passes for no reason.
    """
    await session.execute(
        delete(AvailabilityRule).where(
            AvailabilityRule.clinic_id == graph.clinic_id,
            AvailabilityRule.doctor_id == graph.doctor_id,
        )
    )
    session.add(
        AvailabilityRule(
            clinic_id=graph.clinic_id,
            doctor_id=graph.doctor_id,
            location_id=graph.location_id,
            weekday=day.weekday(),
            start_time=dt.time(8, 0),
            end_time=dt.time(12, 0),
            valid_from=dt.date(2026, 1, 1),
        )
    )
    await session.flush()


async def test_a_booking_outside_the_doctors_hours_is_refused(
    session: AsyncSession,
) -> None:
    """QA finding 3, in the reviewer's own example.

    "A doctor is available from 08:00 to 12:00. A direct API request books an
    otherwise free 15:00-15:30 interval. The exclusion constraint permits the
    insert because no conflicting appointment exists."

    Confirmed against the submitted code: 201. The constraint answers "is this
    time still free", never "does the clinic work then", and nothing else asked.
    """
    graph = await create_graph(session)
    day = dt.date(2026, 11, 3)  # a Tuesday, not a holiday
    await _schedule(session, graph, day)
    afternoon = dt.datetime.combine(day, dt.time(15, 0), tzinfo=BOGOTA)

    with pytest.raises(booking.Ineligible) as refused:
        await booking.book(
            session,
            clinic_id=graph.clinic_id,
            patient_id=graph.patient_id,
            doctor_id=graph.doctor_id,
            location_id=graph.location_id,
            appointment_type_id=graph.appointment_type_id,
            start=afternoon,
            end=afternoon + dt.timedelta(minutes=20),
        )
    # The receptionist reads this, so it names the hours rather than a rule id.
    assert "08:00-12:00" in refused.value.detail_es


async def test_a_booking_inside_the_doctors_hours_is_allowed(
    session: AsyncSession,
) -> None:
    """The check must not refuse the ordinary case."""
    graph = await create_graph(session)
    day = dt.date(2026, 11, 3)
    await _schedule(session, graph, day)
    morning = dt.datetime.combine(day, dt.time(9, 0), tzinfo=BOGOTA)

    result = await booking.book(
        session,
        clinic_id=graph.clinic_id,
        patient_id=graph.patient_id,
        doctor_id=graph.doctor_id,
        location_id=graph.location_id,
        appointment_type_id=graph.appointment_type_id,
        start=morning,
        end=morning + dt.timedelta(minutes=20),
    )
    assert result.appointment is not None, result.outcome


async def test_a_booking_on_a_public_holiday_is_refused(session: AsyncSession) -> None:
    """A holiday closes the recurring schedule, and the booking path has to
    honour that as much as the slot search does. 2 November 2026 is All Souls,
    moved to the Monday by the Ley Emiliani."""
    graph = await create_graph(session)
    holiday = dt.date(2026, 11, 2)
    await _schedule(session, graph, holiday)
    morning = dt.datetime.combine(holiday, dt.time(9, 0), tzinfo=BOGOTA)

    with pytest.raises(booking.Ineligible):
        await booking.book(
            session,
            clinic_id=graph.clinic_id,
            patient_id=graph.patient_id,
            doctor_id=graph.doctor_id,
            location_id=graph.location_id,
            appointment_type_id=graph.appointment_type_id,
            start=morning,
            end=morning + dt.timedelta(minutes=20),
        )


async def test_a_booking_during_a_dated_closure_is_refused(session: AsyncSession) -> None:
    """A dated `unavailable` exception closes a day the clinic would otherwise
    work -- a doctor on leave. The reviewer asked for "during a closure"."""
    graph = await create_graph(session)
    day = dt.date(2026, 11, 3)
    await _schedule(session, graph, day)
    session.add(
        AvailabilityException(
            clinic_id=graph.clinic_id,
            doctor_id=graph.doctor_id,
            on_date=day,
            kind=ExceptionKind.UNAVAILABLE,
        )
    )
    await session.flush()
    morning = dt.datetime.combine(day, dt.time(9, 0), tzinfo=BOGOTA)

    with pytest.raises(booking.Ineligible):
        await booking.book(
            session,
            clinic_id=graph.clinic_id,
            patient_id=graph.patient_id,
            doctor_id=graph.doctor_id,
            location_id=graph.location_id,
            appointment_type_id=graph.appointment_type_id,
            start=morning,
            end=morning + dt.timedelta(minutes=20),
        )


async def test_a_duration_that_is_not_the_appointment_types_is_refused(
    session: AsyncSession,
) -> None:
    """The reviewer's "invalid duration".

    The factory's appointment type is 20 minutes. A caller sending a 10-minute
    end was booking a consultation shorter than the clinic allots, which is how
    a diary ends up with four patients in an hour that holds three.
    """
    graph = await create_graph(session)
    day = dt.date(2026, 11, 3)
    await _schedule(session, graph, day)
    morning = dt.datetime.combine(day, dt.time(9, 0), tzinfo=BOGOTA)

    with pytest.raises(booking.Ineligible) as refused:
        await booking.book(
            session,
            clinic_id=graph.clinic_id,
            patient_id=graph.patient_id,
            doctor_id=graph.doctor_id,
            location_id=graph.location_id,
            appointment_type_id=graph.appointment_type_id,
            start=morning,
            end=morning + dt.timedelta(minutes=10),
        )
    assert "20" in refused.value.detail_es


async def test_a_refused_booking_writes_no_appointment_and_no_audit_entry(
    session: AsyncSession,
) -> None:
    """The reviewer asked for this explicitly: "Verify that failed requests
    leave no appointment or misleading audit entry." An audit entry for a
    booking that never happened is worse than none, because it is evidence of a
    disclosure that did not occur."""
    graph = await create_graph(session)
    day = dt.date(2026, 11, 3)
    await _schedule(session, graph, day)
    afternoon = dt.datetime.combine(day, dt.time(15, 0), tzinfo=BOGOTA)

    before = (await session.execute(select(func.count()).select_from(AccessLogEntry))).scalar_one()
    with pytest.raises(booking.Ineligible):
        await booking.book(
            session,
            clinic_id=graph.clinic_id,
            patient_id=graph.patient_id,
            doctor_id=graph.doctor_id,
            location_id=graph.location_id,
            appointment_type_id=graph.appointment_type_id,
            start=afternoon,
            end=afternoon + dt.timedelta(minutes=20),
        )

    appointments = (
        await session.execute(
            select(func.count())
            .select_from(Appointment)
            .where(Appointment.clinic_id == graph.clinic_id)
        )
    ).scalar_one()
    after = (await session.execute(select(func.count()).select_from(AccessLogEntry))).scalar_one()
    assert appointments == 0, "a refused booking wrote an appointment"
    assert after == before, "a refused booking wrote an audit entry"


async def test_an_authorised_override_books_outside_hours_and_is_audited(
    session: AsyncSession,
) -> None:
    """The report's own requirement: "If an intentional manual override is
    required, model it as a separate, authorized operation with an audit
    trail."

    A receptionist fitting a patient in after hours is a real thing a clinic
    does, so refusing it outright would be the wrong fix. It is a named
    argument carrying a reason, and it leaves its own audit entry so an auditor
    can ask which bookings bypassed the hours.
    """
    graph = await create_graph(session)
    day = dt.date(2026, 11, 3)
    await _schedule(session, graph, day)
    afternoon = dt.datetime.combine(day, dt.time(15, 0), tzinfo=BOGOTA)

    result = await booking.book(
        session,
        clinic_id=graph.clinic_id,
        patient_id=graph.patient_id,
        doctor_id=graph.doctor_id,
        location_id=graph.location_id,
        appointment_type_id=graph.appointment_type_id,
        start=afternoon,
        end=afternoon + dt.timedelta(minutes=20),
        override_reason="La paciente viaja mañana; la doctora autorizó la cita.",
    )
    assert result.appointment is not None, result.outcome

    overrides = set(
        (
            await session.scalars(
                select(AccessLogEntry.resource_id).where(
                    AccessLogEntry.resource == "appointment_overrides"
                )
            )
        ).all()
    )
    assert overrides == {str(result.appointment.id)}, (
        "an override must leave its own audit entry naming the appointment"
    )


async def test_a_reschedule_into_closed_hours_is_refused(session: AsyncSession) -> None:
    """A move is a booking with an extra step, and it was the looser path."""
    graph = await create_graph(session)
    day = dt.date(2026, 11, 3)
    await _schedule(session, graph, day)
    morning = dt.datetime.combine(day, dt.time(9, 0), tzinfo=BOGOTA)
    booked = await booking.book(
        session,
        clinic_id=graph.clinic_id,
        patient_id=graph.patient_id,
        doctor_id=graph.doctor_id,
        location_id=graph.location_id,
        appointment_type_id=graph.appointment_type_id,
        start=morning,
        end=morning + dt.timedelta(minutes=20),
    )
    assert booked.appointment is not None

    with pytest.raises(booking.Ineligible):
        await booking.reschedule(
            session,
            clinic_id=graph.clinic_id,
            appointment_id=booked.appointment.id,
            start=dt.datetime.combine(day, dt.time(22, 0), tzinfo=BOGOTA),
            end=dt.datetime.combine(day, dt.time(22, 20), tzinfo=BOGOTA),
        )
