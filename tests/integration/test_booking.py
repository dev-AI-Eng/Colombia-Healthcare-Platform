"""The booking service: what happens when two patients want one time.

`test_constraints.py` proves the database rejects an overlap. These prove the
service layer *works with* that rejection rather than crashing on it: a lost
race becomes a value a conversation flow can act on, the caller's transaction
survives, and nothing is written that the audit log does not record.

The guarantee each test exists for is named in its docstring. None of them
should be weakened to make a change pass.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.audit.context import AuditContext
from src.audit.models import AccessLogEntry
from src.core.config import Settings
from src.core.tenancy import ClinicScope, apply_clinic_scope
from src.scheduling import booking
from src.scheduling.booking import Outcome
from src.scheduling.models import Appointment, AppointmentStatus
from tests.integration.factories import Graph, at, create_graph

SLOT = at(5, 9)
LATER = at(5, 11)


@pytest.fixture(autouse=True)
def _audited(staff_context: AuditContext) -> AuditContext:
    """Every booking writes an audit entry, which needs an actor bound.

    Requested for the whole module rather than named in each signature: a test
    here that wrote without an audit context would be testing a path that
    cannot exist in the application, where the middleware always binds one.
    """
    return staff_context


async def _book(session: AsyncSession, graph: Graph, start=SLOT, **kwargs):  # type: ignore[no-untyped-def]
    return await booking.book(
        session,
        clinic_id=graph.clinic_id,
        patient_id=graph.patient_id,
        doctor_id=graph.doctor_id,
        location_id=graph.location_id,
        appointment_type_id=graph.appointment_type_id,
        start=start,
        end=start + timedelta(minutes=20),
        **kwargs,
    )


async def _hold(session: AsyncSession, graph: Graph, start=SLOT, **kwargs):  # type: ignore[no-untyped-def]
    return await booking.hold(
        session,
        clinic_id=graph.clinic_id,
        patient_id=graph.patient_id,
        doctor_id=graph.doctor_id,
        location_id=graph.location_id,
        appointment_type_id=graph.appointment_type_id,
        start=start,
        end=start + timedelta(minutes=20),
        now=at(5, 8),
        **kwargs,
    )


async def _count(session: AsyncSession) -> int:
    return (await session.execute(select(func.count()).select_from(Appointment))).scalar_one()


# ------------------------------------------------------------------ booking
async def test_a_free_slot_is_booked(session: AsyncSession) -> None:
    graph = await create_graph(session)
    result = await _book(session, graph)
    assert result.outcome is Outcome.BOOKED
    assert result.ok
    assert result.appointment is not None
    assert result.appointment.status is AppointmentStatus.SCHEDULED


async def test_a_taken_slot_reports_taken_rather_than_raising(session: AsyncSession) -> None:
    """A conversation flow has to be able to act on this.

    Letting the DBAPIError escape would surface a PostgreSQL code to whatever
    is talking to the patient, and would poison the transaction so nothing
    already written could commit.
    """
    graph = await create_graph(session)
    assert (await _book(session, graph)).outcome is Outcome.BOOKED

    second = await _book(session, graph, start=SLOT + timedelta(minutes=10))
    assert second.outcome is Outcome.TAKEN
    assert second.appointment is None


async def test_the_transaction_survives_a_lost_race(session: AsyncSession) -> None:
    """The savepoint is what makes a lost race recoverable.

    Without it the failed insert poisons the whole transaction: the caller
    cannot go on to offer a different time, and the audit entries written
    before the attempt are lost. This books, loses a race, then books again on
    the same session.
    """
    graph = await create_graph(session)
    assert (await _book(session, graph)).outcome is Outcome.BOOKED
    assert (await _book(session, graph)).outcome is Outcome.TAKEN

    recovered = await _book(session, graph, start=LATER)
    assert recovered.outcome is Outcome.BOOKED, "the session must still be usable"
    assert await _count(session) == 2


async def test_back_to_back_appointments_both_fit(session: AsyncSession) -> None:
    """`during` is half-open, so 09:00-09:20 and 09:20-09:40 do not overlap."""
    graph = await create_graph(session)
    assert (await _book(session, graph)).outcome is Outcome.BOOKED
    assert (await _book(session, graph, start=SLOT + timedelta(minutes=20))).outcome is (
        Outcome.BOOKED
    )


async def test_booking_writes_exactly_one_audit_entry(session: AsyncSession) -> None:
    """Every patient-data write is audited, inside the same transaction."""
    graph = await create_graph(session)
    before = (await session.execute(select(func.count()).select_from(AccessLogEntry))).scalar_one()
    result = await _book(session, graph)
    after = (await session.execute(select(func.count()).select_from(AccessLogEntry))).scalar_one()
    assert after == before + 1
    assert result.appointment is not None


async def test_a_lost_race_writes_no_audit_entry(session: AsyncSession) -> None:
    """The log must not claim a booking that never happened."""
    graph = await create_graph(session)
    await _book(session, graph)
    before = (await session.execute(select(func.count()).select_from(AccessLogEntry))).scalar_one()
    assert (await _book(session, graph)).outcome is Outcome.TAKEN
    after = (await session.execute(select(func.count()).select_from(AccessLogEntry))).scalar_one()
    assert after == before


# ------------------------------------------------------------- idempotency
async def test_a_replayed_request_returns_the_first_appointment(session: AsyncSession) -> None:
    """A redelivered webhook must not create a second booking (ADR-16)."""
    graph = await create_graph(session)
    first = await _book(session, graph, idempotency_key="wh-1")
    assert first.outcome is Outcome.BOOKED

    replay = await _book(session, graph, start=LATER, idempotency_key="wh-1")
    assert replay.outcome is Outcome.DUPLICATE
    assert replay.appointment is not None
    assert replay.appointment.id == first.appointment.id  # type: ignore[union-attr]
    assert await _count(session) == 1, "the replay created nothing"


# -------------------------------------------------------------------- holds
async def test_a_hold_blocks_a_booking(session: AsyncSession) -> None:
    """ADR-17: a hold and a booking share one exclusion constraint."""
    graph = await create_graph(session)
    assert (await _hold(session, graph)).outcome is Outcome.BOOKED
    assert (await _book(session, graph)).outcome is Outcome.TAKEN


async def test_a_booking_blocks_a_hold(session: AsyncSession) -> None:
    """The reverse, which is the half that is easy to forget."""
    graph = await create_graph(session)
    assert (await _book(session, graph)).outcome is Outcome.BOOKED
    assert (await _hold(session, graph)).outcome is Outcome.TAKEN


async def test_a_live_hold_confirms_into_a_booking(session: AsyncSession) -> None:
    graph = await create_graph(session)
    held = await _hold(session, graph, minutes=15)
    assert held.appointment is not None

    confirmed = await booking.confirm_hold(
        session,
        clinic_id=graph.clinic_id,
        appointment_id=held.appointment.id,
        now=at(5, 8, 10),
    )
    assert confirmed.outcome is Outcome.BOOKED
    assert confirmed.appointment is not None
    assert confirmed.appointment.status is AppointmentStatus.SCHEDULED
    assert confirmed.appointment.expires_at is None, "a scheduled row carries no expiry"


async def test_an_expired_hold_cannot_be_confirmed(session: AsyncSession) -> None:
    """The time may have gone to somebody else; the flow re-offers (ADR-17)."""
    graph = await create_graph(session)
    held = await _hold(session, graph, minutes=15)
    assert held.appointment is not None

    late = await booking.confirm_hold(
        session,
        clinic_id=graph.clinic_id,
        appointment_id=held.appointment.id,
        now=at(5, 8, 30),  # the hold lapsed at 08:15
    )
    assert late.outcome is Outcome.TAKEN


async def test_releasing_a_hold_frees_the_slot(session: AsyncSession) -> None:
    graph = await create_graph(session)
    held = await _hold(session, graph)
    assert held.appointment is not None

    assert await booking.release_hold(
        session, clinic_id=graph.clinic_id, appointment_id=held.appointment.id
    )
    assert (await _book(session, graph)).outcome is Outcome.BOOKED


async def test_releasing_leaves_no_row_behind(session: AsyncSession) -> None:
    """An offer nobody accepted is not part of the patient's history."""
    graph = await create_graph(session)
    held = await _hold(session, graph)
    assert held.appointment is not None
    await booking.release_hold(
        session, clinic_id=graph.clinic_id, appointment_id=held.appointment.id
    )
    assert await _count(session) == 0


async def test_the_sweep_removes_only_lapsed_holds(session: AsyncSession) -> None:
    """Housekeeping: a live hold and a real booking must both survive it."""
    graph = await create_graph(session)
    await _hold(session, graph, minutes=15)  # lapses 08:15
    await _hold(session, graph, start=LATER, minutes=120)  # lapses 10:00
    await _book(session, graph, start=at(5, 14))

    removed = await booking.sweep_expired_holds(session, clinic_id=graph.clinic_id, now=at(5, 9))
    assert removed == 1
    assert await _count(session) == 2


# ------------------------------------------------------- cancel and reschedule
async def test_cancelling_frees_the_time_and_keeps_the_row(session: AsyncSession) -> None:
    """History keeps the fact that an appointment existed and was cancelled."""
    graph = await create_graph(session)
    first = await _book(session, graph)
    assert first.appointment is not None

    assert await booking.cancel(
        session,
        clinic_id=graph.clinic_id,
        appointment_id=first.appointment.id,
        reason="El paciente no puede asistir",
    )
    assert first.appointment.status is AppointmentStatus.CANCELLED
    assert (await _book(session, graph)).outcome is Outcome.BOOKED, "the time is free again"
    assert await _count(session) == 2, "the cancelled row is still there"


async def test_cancelling_twice_reports_the_second_as_nothing_to_do(
    session: AsyncSession,
) -> None:
    graph = await create_graph(session)
    first = await _book(session, graph)
    assert first.appointment is not None
    assert await booking.cancel(
        session, clinic_id=graph.clinic_id, appointment_id=first.appointment.id
    )
    assert not await booking.cancel(
        session, clinic_id=graph.clinic_id, appointment_id=first.appointment.id
    )


async def test_rescheduling_moves_the_appointment_and_links_it_back(
    session: AsyncSession,
) -> None:
    graph = await create_graph(session)
    first = await _book(session, graph)
    assert first.appointment is not None

    moved = await booking.reschedule(
        session,
        clinic_id=graph.clinic_id,
        appointment_id=first.appointment.id,
        start=LATER,
        end=LATER + timedelta(minutes=20),
    )
    assert moved.outcome is Outcome.BOOKED
    assert moved.appointment is not None
    assert moved.appointment.rescheduled_from_appointment_id == first.appointment.id
    assert first.appointment.status is AppointmentStatus.RESCHEDULED


async def test_a_failed_reschedule_leaves_the_original_untouched(
    session: AsyncSession,
) -> None:
    """Releasing the old time before securing the new one would be the bug.

    If the new time is taken and the original has already been given up, the
    patient ends with no appointment at all.
    """
    graph = await create_graph(session)
    original = await _book(session, graph)
    blocker = await _book(session, graph, start=LATER)
    assert original.appointment is not None and blocker.appointment is not None

    attempt = await booking.reschedule(
        session,
        clinic_id=graph.clinic_id,
        appointment_id=original.appointment.id,
        start=LATER,
        end=LATER + timedelta(minutes=20),
    )
    assert attempt.outcome is Outcome.TAKEN
    assert original.appointment.status is AppointmentStatus.SCHEDULED, "still booked"
    assert await _count(session) == 2, "nothing new was written"


# --------------------------------------------------------------- refusals
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"status": AppointmentStatus.HOLD}, "use `hold` to create a hold"),
        ({"status": AppointmentStatus.CANCELLED}, "does not occupy the doctor's time"),
    ],
)
async def test_booking_refuses_a_status_that_is_not_a_booking(
    session: AsyncSession, kwargs: dict, message: str
) -> None:
    graph = await create_graph(session)
    with pytest.raises(ValueError, match=message):
        await _book(session, graph, **kwargs)


async def test_an_appointment_must_end_after_it_starts(session: AsyncSession) -> None:
    graph = await create_graph(session)
    with pytest.raises(ValueError, match="must end after it starts"):
        await booking.book(
            session,
            clinic_id=graph.clinic_id,
            patient_id=graph.patient_id,
            doctor_id=graph.doctor_id,
            location_id=graph.location_id,
            appointment_type_id=graph.appointment_type_id,
            start=SLOT,
            end=SLOT,
        )


# ------------------------------------------------- the headline guarantee
async def test_concurrent_bookings_through_the_service_admit_exactly_one(
    session: AsyncSession, settings: Settings
) -> None:
    """PDF criterion 1, at the layer a caller actually uses.

    `test_constraints.py` fires raw inserts at the constraint. This fires the
    booking service from separate connections, which is what a production
    reschedule storm looks like: every loser must come back as TAKEN rather
    than as an exception, and exactly one appointment must exist afterwards.
    """
    graph = await create_graph(session)
    await session.commit()

    engine = create_async_engine(settings.app_database_url, poolclass=NullPool)
    maker = async_sessionmaker(engine, expire_on_commit=False)

    async def attempt(offset: int) -> Outcome:
        async with maker() as racer:
            await apply_clinic_scope(racer, ClinicScope(clinic_id=graph.clinic_id))
            result = await _book(racer, graph, start=SLOT + timedelta(minutes=offset))
            await racer.commit()
            return result.outcome

    try:
        outcomes = await asyncio.gather(*(attempt(i) for i in range(12)))
    finally:
        await engine.dispose()

    assert outcomes.count(Outcome.BOOKED) == 1, outcomes
    assert set(outcomes) == {Outcome.BOOKED, Outcome.TAKEN}

    await apply_clinic_scope(session, ClinicScope(clinic_id=graph.clinic_id))
    assert await _count(session) == 1
