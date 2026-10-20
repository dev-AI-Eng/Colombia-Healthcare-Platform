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
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.audit.context import AuditContext
from src.audit.models import AccessLogEntry
from src.core.config import Settings
from src.core.crypto import blind_index
from src.core.tenancy import ClinicScope, apply_clinic_scope
from src.core.timezones import BOGOTA
from src.registry.models import DocumentType, Location, Patient
from src.scheduling import availability, booking, offers, reminders
from src.scheduling.booking import BookingResult, Outcome
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
        # The fixture dates are fixed, and the real clock has passed them.
        # Since the past-booking refusal, a test booking a fixture date has to
        # say which moment it is booking from.
        now=at(8, 8, 0),
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
                    now=at(9, 8, 0),
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
                    now=at(10, 8, 0),
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
        # No try/except. An earlier version of this test caught every exception
        # and returned 0, which supplied the error handling production lacked:
        # the sweep was crashing on the unique index with an IntegrityError and
        # this harness hid it, so the test passed while one losing sweep rolled
        # back every clinic's housekeeping. Letting it propagate is what makes
        # the test sensitive to the defect it claims to guard.
        async with maker() as other:
            await apply_clinic_scope(other, ClinicScope(clinic_id=graph.clinic_id))
            raised = await reminders.flag_unrecorded_attendance(
                other, clinic_id=graph.clinic_id, now=now
            )
            await other.commit()
            return raised

    try:
        raised = await asyncio.gather(sweep(), sweep())
    finally:
        await maker.kw["bind"].dispose()

    # Both sweeps complete, and between them they raise the question once. The
    # loser returning 0 is the ON CONFLICT no-op; a loser that raised would
    # abort the whole sweep, which is the failure this asserts against.
    assert sorted(raised) == [0, 1], (
        f"the two sweeps returned {raised}; exactly one should have raised the "
        f"escalation and the other should have found it already taken"
    )

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
        now=at(11, 8, 0),
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


# ========================================================= finding 5: idempotency scope

#: The reviewer's six scenarios, as one table. Only the first two are replays.
# The 13th, a Tuesday. The 12th is Día de la Raza, moved to the Monday by the
# Ley Emiliani, and open_windows closes a holiday -- so these tests would fail
# on eligibility rather than on the idempotency behaviour they are about.
_FIXED = at(13, 9, 0)


async def _book_with(
    session: AsyncSession, graph: Graph, key: str, **over: object
) -> BookingResult:
    args: dict[str, object] = {
        "clinic_id": graph.clinic_id,
        "patient_id": graph.patient_id,
        "doctor_id": graph.doctor_id,
        "location_id": graph.location_id,
        "appointment_type_id": graph.appointment_type_id,
        "start": _FIXED,
        "end": _FIXED + dt.timedelta(minutes=20),
        "idempotency_key": key,
        "now": at(13, 8, 0),
    }
    args.update(over)
    return await booking.book(session, **args)  # type: ignore[arg-type]


async def test_an_exact_replay_returns_the_first_appointment(session: AsyncSession) -> None:
    """The reviewer's first scenario, and the one that must keep working.

    "Replay the exact same booking request." A redelivered webhook must not
    create a second appointment, which is the whole purpose of the key.
    """
    graph = await create_graph(session, hours=True)
    first = await _book_with(session, graph, "replay-1")
    assert first.appointment is not None, first.outcome

    again = await _book_with(session, graph, "replay-1")
    assert again.outcome is Outcome.DUPLICATE
    assert again.appointment is not None
    assert again.appointment.id == first.appointment.id


async def test_an_exact_hold_replay_returns_the_first_hold(session: AsyncSession) -> None:
    """ "Replay the exact same hold request." A hold carries a status of its own,
    so the identity check has to allow a hold to replay as a hold."""
    graph = await create_graph(session, hours=True)
    common = {
        "clinic_id": graph.clinic_id,
        "patient_id": graph.patient_id,
        "doctor_id": graph.doctor_id,
        "location_id": graph.location_id,
        "appointment_type_id": graph.appointment_type_id,
        "start": _FIXED,
        "end": _FIXED + dt.timedelta(minutes=20),
        "now": at(13, 8, 0),
        "idempotency_key": "hold-replay-1",
    }
    first = await booking.hold(session, **common)  # type: ignore[arg-type]
    assert first.appointment is not None, first.outcome

    again = await booking.hold(session, **common)  # type: ignore[arg-type]
    assert again.outcome is Outcome.DUPLICATE
    assert again.appointment is not None
    assert again.appointment.id == first.appointment.id


async def test_a_booking_key_reused_for_a_hold_is_a_conflict(session: AsyncSession) -> None:
    """The reviewer's third scenario: "Reuse a booking key for a hold."

    Before the fix this returned the booking and the API reported 201, so a
    caller that asked for a fifteen-minute hold was told it held the slot and
    had in fact been handed a confirmed appointment it did not create.
    """
    graph = await create_graph(session, hours=True)
    booked = await _book_with(session, graph, "crossed-1")
    assert booked.appointment is not None, booked.outcome

    with pytest.raises(booking.KeyReused):
        await booking.hold(
            session,
            clinic_id=graph.clinic_id,
            patient_id=graph.patient_id,
            doctor_id=graph.doctor_id,
            location_id=graph.location_id,
            appointment_type_id=graph.appointment_type_id,
            start=_FIXED,
            end=_FIXED + dt.timedelta(minutes=20),
            now=at(13, 8, 0),
            idempotency_key="crossed-1",
        )


async def test_a_key_reused_for_another_patient_is_a_conflict(session: AsyncSession) -> None:
    """ "Reuse a key for a different patient's booking."

    This is the one with a patient-safety edge: the first caller's appointment
    was returned as though it were the second patient's, so a confirmation
    would have gone to the wrong person about the wrong appointment.
    """
    graph = await create_graph(session, hours=True)
    other = Patient(
        clinic_id=graph.clinic_id,
        document_type=DocumentType.CC,
        document_number="9988776655",
        document_number_bidx=blind_index("9988776655"),
        given_names="Beatriz",
        family_names="Núñez Peña",
    )
    session.add(other)
    await session.flush()

    booked = await _book_with(session, graph, "shared-1")
    assert booked.appointment is not None, booked.outcome

    with pytest.raises(booking.KeyReused):
        await _book_with(
            session,
            graph,
            "shared-1",
            patient_id=other.id,
            start=_FIXED + dt.timedelta(hours=1),
            end=_FIXED + dt.timedelta(hours=1, minutes=20),
        )


async def test_a_key_reused_across_clinics_is_a_conflict(
    session: AsyncSession, settings: Settings
) -> None:
    """ "Reuse a key across clinics."

    The column is unique table-wide, so the second clinic's insert collides.
    Row-level security hides the other clinic's row, so `_existing_by_key`
    finds nothing and the violation is re-raised -- which is correct, but as a
    database error rather than an answer. Asserting on the behaviour either way:
    the second clinic must not be handed the first clinic's appointment.
    """
    mine = await create_graph(session, name="Clínica Una", hours=True)
    booked = await _book_with(session, mine, "cross-clinic-1")
    assert booked.appointment is not None, booked.outcome
    await session.commit()

    theirs = await create_graph(session, name="Clínica Dos", hours=True)
    await session.commit()

    maker = _maker(settings)
    try:
        async with maker() as other:
            await apply_clinic_scope(other, ClinicScope(clinic_id=theirs.clinic_id))
            with pytest.raises((DBAPIError, booking.KeyReused)):
                await _book_with(other, theirs, "cross-clinic-1")
            await other.rollback()
    finally:
        await maker.kw["bind"].dispose()


async def test_a_key_reused_for_a_different_reschedule_destination_is_a_conflict(
    session: AsyncSession,
) -> None:
    """ "Reuse a key for a different reschedule destination."

    The dangerous version of this one: the first reschedule's appointment was
    returned for a request naming a different time, so the caller believed the
    patient had been moved to a time nobody had booked.
    """
    graph = await create_graph(session, hours=True)
    first = await _book_with(session, graph, "resched-origin")
    assert first.appointment is not None, first.outcome

    moved = await booking.reschedule(
        session,
        clinic_id=graph.clinic_id,
        appointment_id=first.appointment.id,
        start=at(13, 14, 0),
        end=at(13, 14, 20),
        idempotency_key="resched-1",
        now=at(13, 8, 0),
    )
    assert moved.appointment is not None, moved.outcome

    second = await _book_with(session, graph, "resched-second")
    assert second.appointment is not None, second.outcome
    with pytest.raises(booking.KeyReused):
        await booking.reschedule(
            session,
            clinic_id=graph.clinic_id,
            appointment_id=second.appointment.id,
            start=at(13, 16, 0),  # a different destination under the same key
            end=at(13, 16, 20),
            idempotency_key="resched-1",
            now=at(13, 8, 0),
        )


async def test_an_exact_reschedule_replay_does_not_move_the_patient_twice(
    session: AsyncSession,
) -> None:
    """An exact replay of a reschedule still succeeds, and leaves exactly one
    active appointment. The original was transitioned by the first call, and
    `reschedule` now checks that rather than assuming it."""
    graph = await create_graph(session, hours=True)
    booked = await _book_with(session, graph, "replay-resched-origin")
    assert booked.appointment is not None, booked.outcome

    args = {
        "clinic_id": graph.clinic_id,
        "appointment_id": booked.appointment.id,
        "start": at(13, 15, 0),
        "end": at(13, 15, 20),
        "idempotency_key": "replay-resched-1",
        "now": at(13, 8, 0),
    }
    first = await booking.reschedule(session, **args)  # type: ignore[arg-type]
    assert first.appointment is not None, first.outcome

    again = await booking.reschedule(session, **args)  # type: ignore[arg-type]
    assert again.outcome is Outcome.DUPLICATE

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
    assert active == 1, f"{active} active appointments after a replayed move; expected 1"


# ========================================================= finding 9: location


async def test_a_search_for_one_location_ignores_the_others_hours(
    session: AsyncSession,
) -> None:
    """QA finding 9, in the reviewer's own scenario.

    "Configure the same doctor at two locations with different hours. A search
    for Location A must not return availability that exists only at Location
    B."

    Before the fix `_load_rules` ignored location entirely, so a slot derived
    from the north site's afternoon hours could be held against the south
    site's id -- the patient is told 15:00 at a building the doctor is not in.
    """
    graph = await create_graph(session)
    await _clear_rules(session, graph)
    north = Location(clinic_id=graph.clinic_id, name="Sede Norte", address="Calle 100")
    session.add(north)
    await session.flush()

    target = dt.date(2026, 11, 10)  # a Tuesday
    session.add_all(
        [
            # The original site: mornings only.
            AvailabilityRule(
                clinic_id=graph.clinic_id,
                doctor_id=graph.doctor_id,
                location_id=graph.location_id,
                weekday=target.weekday(),
                start_time=dt.time(8, 0),
                end_time=dt.time(11, 0),
                valid_from=dt.date(2026, 1, 1),
            ),
            # The north site: afternoons only.
            AvailabilityRule(
                clinic_id=graph.clinic_id,
                doctor_id=graph.doctor_id,
                location_id=north.id,
                weekday=target.weekday(),
                start_time=dt.time(14, 0),
                end_time=dt.time(17, 0),
                valid_from=dt.date(2026, 1, 1),
            ),
        ]
    )
    await session.flush()

    def search(location_id: uuid.UUID | None) -> offers.SlotSearch:
        return offers.SlotSearch(
            clinic_id=graph.clinic_id,
            doctor_id=graph.doctor_id,
            appointment_type_id=graph.appointment_type_id,
            start_date=target,
            end_date=target,
            location_id=location_id,
        )

    now = dt.datetime(2026, 11, 1, 9, 0, tzinfo=BOGOTA)
    original = await offers.find_slots(session, search(graph.location_id), now=now, limit=100)
    assert original, "the original site should still offer its mornings"
    hours = {s.start.astimezone(BOGOTA).hour for s in original}
    assert hours <= {8, 9, 10}, f"a search for one site returned another's hours: {sorted(hours)}"

    at_north = await offers.find_slots(session, search(north.id), now=now, limit=100)
    assert at_north, "the north site should offer its afternoons"
    north_hours = {s.start.astimezone(BOGOTA).hour for s in at_north}
    assert north_hours <= {14, 15, 16}, sorted(north_hours)

    # And a search with no location still sees everything, which is what a
    # single-site clinic needs and the only behaviour that existed before.
    both = await offers.find_slots(session, search(None), now=now, limit=100)
    assert {s.start.astimezone(BOGOTA).hour for s in both} > hours


async def test_offering_refuses_to_hold_a_slot_at_a_different_location(
    session: AsyncSession,
) -> None:
    """The mismatch the reviewer described, made impossible rather than unlikely.

    `offer` took `location_id` for the hold while the search ignored it, so the
    two could disagree silently. They cannot now: a caller passing both must
    pass the same one.
    """
    graph = await create_graph(session, hours=True)
    north = Location(clinic_id=graph.clinic_id, name="Sede Norte", address="Calle 100")
    session.add(north)
    await session.flush()

    with pytest.raises(ValueError, match="cannot be held at another"):
        await offers.offer(
            session,
            offers.SlotSearch(
                clinic_id=graph.clinic_id,
                doctor_id=graph.doctor_id,
                appointment_type_id=graph.appointment_type_id,
                start_date=dt.date(2026, 11, 10),
                end_date=dt.date(2026, 11, 10),
                location_id=graph.location_id,
            ),
            patient_id=graph.patient_id,
            location_id=north.id,  # not the location the search was for
            now=dt.datetime(2026, 11, 1, 9, 0, tzinfo=BOGOTA),
        )


# ========================================================= finding 14: reminder claim


@pytest.mark.concurrency
async def test_two_workers_do_not_report_the_same_reminder(
    session: AsyncSession, settings: Settings
) -> None:
    """QA finding 14's first risk: "Simulate two workers claiming the same
    reminder."

    Two sweeps running at once used to both report every due reminder, because
    the read took no lock and `mark_sent` happened later. The patient then gets
    the same message twice. `FOR UPDATE SKIP LOCKED` divides the rows instead.

    The two sweeps are deliberately overlapping in time: the first holds its
    lock until its transaction ends, so the second must see nothing.
    """
    graph = await create_graph(session, hours=True)
    start = dt.datetime.now(tz=dt.UTC) + dt.timedelta(hours=23)
    session.add(
        appointment(
            graph,
            start.astimezone(BOGOTA).replace(minute=0, second=0, microsecond=0),
        )
    )
    await session.flush()
    await session.commit()

    maker = _maker(settings)
    now = dt.datetime.now(tz=dt.UTC)
    started = asyncio.Event()
    release = asyncio.Event()

    async def first() -> list[reminders.DueReminder]:
        async with maker() as other:
            await apply_clinic_scope(other, ClinicScope(clinic_id=graph.clinic_id))
            due = await reminders.reminders_due(other, clinic_id=graph.clinic_id, now=now)
            started.set()
            await release.wait()  # hold the lock while the second sweep runs
            await other.rollback()
            return due

    async def second() -> list[reminders.DueReminder]:
        await started.wait()
        async with maker() as other:
            await apply_clinic_scope(other, ClinicScope(clinic_id=graph.clinic_id))
            try:
                # The timeout is the assertion. Without SKIP LOCKED this query
                # blocks on the first sweep's row lock, and the first sweep is
                # waiting on this one to finish -- the two tasks deadlock and
                # the suite hangs rather than failing. Ten seconds turns that
                # into a readable failure; the query takes milliseconds when
                # the claim is in place.
                return await asyncio.wait_for(
                    reminders.reminders_due(other, clinic_id=graph.clinic_id, now=now),
                    timeout=10,
                )
            except TimeoutError:
                pytest.fail(
                    "the second sweep blocked on the first sweep's lock: "
                    "reminders_due is not claiming rows with SKIP LOCKED"
                )
            finally:
                release.set()
                await other.rollback()

    try:
        mine, theirs = await asyncio.gather(first(), second())
    finally:
        await maker.kw["bind"].dispose()

    assert len(mine) == 1, f"the first sweep should claim the reminder: {mine}"
    assert theirs == [], f"the second sweep claimed a reminder the first held: {theirs}"


async def test_a_reminder_released_without_being_sent_is_found_again(
    session: AsyncSession, settings: Settings
) -> None:
    """The other half of the claim: a worker that crashes must not lose the
    reminder. The lock ends with the transaction, so the next sweep sees it --
    at-least-once, which is the behaviour M4's outbox builds on."""
    graph = await create_graph(session, hours=True)
    start = dt.datetime.now(tz=dt.UTC) + dt.timedelta(hours=23)
    session.add(
        appointment(
            graph,
            start.astimezone(BOGOTA).replace(minute=0, second=0, microsecond=0),
        )
    )
    await session.flush()
    await session.commit()

    maker = _maker(settings)
    now = dt.datetime.now(tz=dt.UTC)
    try:
        # A worker that found it and died.
        async with maker() as crashed:
            await apply_clinic_scope(crashed, ClinicScope(clinic_id=graph.clinic_id))
            found = await reminders.reminders_due(crashed, clinic_id=graph.clinic_id, now=now)
            assert len(found) == 1
            await crashed.rollback()

        async with maker() as retry:
            await apply_clinic_scope(retry, ClinicScope(clinic_id=graph.clinic_id))
            again = await reminders.reminders_due(retry, clinic_id=graph.clinic_id, now=now)
            await retry.rollback()
    finally:
        await maker.kw["bind"].dispose()

    assert len(again) == 1, "a reminder nobody sent was lost when the worker died"


# ===================================================== round 2: F1, location in the write path


async def test_a_booking_is_refused_at_a_location_where_the_doctor_has_no_hours(
    session: AsyncSession,
) -> None:
    """Round-2 finding 1: QA finding 9's fix landed in `offers` and not in the
    write path.

    `offers._load_rules` gained a location filter; `booking._rules_for` did not,
    so `check_eligible` answered "does the doctor work at this hour" from every
    site's hours at once. Measured against the code as fixed-for-finding-9: a
    search at Sede Sur returned 0 slots while a booking at Sede Sur was
    accepted, for a doctor whose only rule was at Sede Norte.

    The patient travels to the wrong building, the doctor is across town, and
    the appointment still occupies the doctor's time in the exclusion
    constraint -- so a real Sede Norte patient is refused that slot.
    """
    graph = await create_graph(session)
    await _clear_rules(session, graph)
    south = Location(clinic_id=graph.clinic_id, name="Sede Sur", address="Calle 80")
    session.add(south)
    await session.flush()

    day = dt.date(2026, 11, 10)  # a Tuesday
    session.add(
        AvailabilityRule(
            clinic_id=graph.clinic_id,
            doctor_id=graph.doctor_id,
            location_id=graph.location_id,  # the original site only
            weekday=day.weekday(),
            start_time=dt.time(8, 0),
            end_time=dt.time(12, 0),
            valid_from=dt.date(2026, 1, 1),
        )
    )
    await session.flush()

    morning = dt.datetime.combine(day, dt.time(9, 0), tzinfo=BOGOTA)
    before = dt.datetime.combine(day, dt.time(7, 0), tzinfo=BOGOTA)

    # The site with hours still works.
    ok = await booking.book(
        session,
        clinic_id=graph.clinic_id,
        patient_id=graph.patient_id,
        doctor_id=graph.doctor_id,
        location_id=graph.location_id,
        appointment_type_id=graph.appointment_type_id,
        start=morning,
        end=morning + dt.timedelta(minutes=20),
        now=before,
    )
    assert ok.appointment is not None, ok.outcome

    # The site without hours does not.
    with pytest.raises(booking.Ineligible):
        await booking.book(
            session,
            clinic_id=graph.clinic_id,
            patient_id=graph.patient_id,
            doctor_id=graph.doctor_id,
            location_id=south.id,
            appointment_type_id=graph.appointment_type_id,
            start=morning + dt.timedelta(hours=1),
            end=morning + dt.timedelta(hours=1, minutes=20),
            now=before,
        )


async def test_a_hold_is_refused_at_a_location_where_the_doctor_has_no_hours(
    session: AsyncSession,
) -> None:
    """The same for a hold, which is the path a conversation actually takes --
    and a hold at the wrong site blocks the slot for everyone while it lives."""
    graph = await create_graph(session)
    await _clear_rules(session, graph)
    south = Location(clinic_id=graph.clinic_id, name="Sede Sur", address="Calle 80")
    session.add(south)
    await session.flush()

    day = dt.date(2026, 11, 11)  # a Wednesday
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

    morning = dt.datetime.combine(day, dt.time(9, 0), tzinfo=BOGOTA)
    with pytest.raises(booking.Ineligible):
        await booking.hold(
            session,
            clinic_id=graph.clinic_id,
            patient_id=graph.patient_id,
            doctor_id=graph.doctor_id,
            location_id=south.id,
            appointment_type_id=graph.appointment_type_id,
            start=morning,
            end=morning + dt.timedelta(minutes=20),
            now=dt.datetime.combine(day, dt.time(7, 0), tzinfo=BOGOTA),
        )


# ===================================================== round 2: F2, bookings in the past


async def test_a_booking_in_the_past_is_refused(session: AsyncSession) -> None:
    """Round-2 finding 2: `check_eligible` took `now` and never read it.

    Nothing refused a booking whose time had already gone. `bookable_slots`
    drops a past slot -- "offering a patient a time that has already gone is
    worse than offering nothing" -- and the write path had no equivalent, so a
    slow conversation turn or a redelivered webhook could book into the past.
    """
    graph = await create_graph(session, hours=True)
    day = dt.date(2026, 11, 12)  # a Thursday
    past = dt.datetime.combine(day, dt.time(9, 0), tzinfo=BOGOTA)

    with pytest.raises(booking.Ineligible) as refused:
        await booking.book(
            session,
            clinic_id=graph.clinic_id,
            patient_id=graph.patient_id,
            doctor_id=graph.doctor_id,
            location_id=graph.location_id,
            appointment_type_id=graph.appointment_type_id,
            start=past,
            end=past + dt.timedelta(minutes=20),
            now=past + dt.timedelta(days=1),  # a day later
        )
    # A receptionist reads this, so it says the hour has gone rather than
    # naming a rule.
    assert "pasó" in refused.value.detail_es


async def test_a_hold_in_the_past_is_refused(session: AsyncSession) -> None:
    """A hold minted in the past expires before it is created: a dead row that
    still occupies the exclusion constraint until a sweep runs, which is QA
    finding 2's failure mode re-entered through another door."""
    graph = await create_graph(session, hours=True)
    day = dt.date(2026, 11, 12)
    past = dt.datetime.combine(day, dt.time(11, 0), tzinfo=BOGOTA)

    with pytest.raises(booking.Ineligible):
        await booking.hold(
            session,
            clinic_id=graph.clinic_id,
            patient_id=graph.patient_id,
            doctor_id=graph.doctor_id,
            location_id=graph.location_id,
            appointment_type_id=graph.appointment_type_id,
            start=past,
            end=past + dt.timedelta(minutes=20),
            now=past + dt.timedelta(hours=2),
        )


async def test_the_importer_may_still_transcribe_a_past_appointment(
    session: AsyncSession,
) -> None:
    """The past-booking refusal must not break the import.

    An import records appointments the clinic already held, which are all in
    the past by definition. `override_reason` is how that intent is stated, and
    the refusal does not apply to it -- the same reasoning that lets an import
    book outside today's configured hours.
    """
    graph = await create_graph(session, hours=True)
    day = dt.date(2026, 11, 12)
    past = dt.datetime.combine(day, dt.time(15, 0), tzinfo=BOGOTA)

    result = await booking.book(
        session,
        clinic_id=graph.clinic_id,
        patient_id=graph.patient_id,
        doctor_id=graph.doctor_id,
        location_id=graph.location_id,
        appointment_type_id=graph.appointment_type_id,
        start=past,
        end=past + dt.timedelta(minutes=20),
        now=past + dt.timedelta(days=30),
        override_reason="imported from the clinic's own records",
    )
    assert result.appointment is not None, result.outcome
