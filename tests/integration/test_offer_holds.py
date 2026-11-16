"""A reschedule completed onto an offered hold refuses rather than guesses.

`booking.move_into_hold` is the write that finishes a two-turn reschedule:
the patient picked a time the conversation offered and held. These tests go
straight at the booking function, under the graph, because each refusal is a
case the conversation reaches only through timing -- an offer that expired a
second before the reply, an appointment cancelled by staff in between -- and
in each the right answer is to change nothing and let the conversation offer
again (ADR-17).
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.audit.context import AuditContext
from src.scheduling import booking
from src.scheduling.models import Appointment, AppointmentStatus
from tests.integration.factories import Graph, appointment, at, create_graph

NOW = dt.datetime.now(tz=dt.UTC)


@pytest.fixture(autouse=True)
def _audited(staff_context: AuditContext) -> AuditContext:
    """Every write here is a patient-data write, and rule 8 refuses one with
    nobody to attribute it to."""
    return staff_context


async def _original_and_offer(
    session: AsyncSession, *, expires_in: dt.timedelta = dt.timedelta(minutes=15)
) -> tuple[Graph, Appointment, Appointment]:
    graph = await create_graph(session, hours=True)
    original = appointment(graph, at(3, 9, 0))
    session.add(original)
    await session.flush()
    offer = appointment(
        graph,
        at(4, 10, 0),
        status=AppointmentStatus.HOLD,
        expires_at=NOW + expires_in,
    )
    offer.rescheduled_from_appointment_id = original.id
    session.add(offer)
    await session.flush()
    return graph, original, offer


async def test_moving_into_a_live_offer_completes_the_reschedule(session: AsyncSession) -> None:
    graph, original, offer = await _original_and_offer(session)

    result = await booking.move_into_hold(
        session, clinic_id=graph.clinic_id, hold_id=offer.id, now=NOW
    )

    assert result.outcome is booking.Outcome.BOOKED
    await session.refresh(original)
    await session.refresh(offer)
    assert original.status is AppointmentStatus.RESCHEDULED
    assert offer.status is AppointmentStatus.CONFIRMED
    assert offer.expires_at is None


async def test_an_expired_offer_changes_nothing(session: AsyncSession) -> None:
    """The time may already belong to somebody else once its hold lapsed."""
    graph, original, offer = await _original_and_offer(session, expires_in=dt.timedelta(minutes=-1))

    result = await booking.move_into_hold(
        session, clinic_id=graph.clinic_id, hold_id=offer.id, now=NOW
    )

    assert result.outcome is booking.Outcome.TAKEN
    await session.refresh(original)
    assert original.status is AppointmentStatus.SCHEDULED


async def test_a_hold_that_is_not_an_offer_is_not_a_reschedule(session: AsyncSession) -> None:
    """A held booking with no appointment to replace is confirmed by
    `confirm_hold`, never by this path."""
    graph, _, offer = await _original_and_offer(session)
    offer.rescheduled_from_appointment_id = None
    await session.flush()

    result = await booking.move_into_hold(
        session, clinic_id=graph.clinic_id, hold_id=offer.id, now=NOW
    )

    assert result.outcome is booking.Outcome.TAKEN
    await session.refresh(offer)
    assert offer.status is AppointmentStatus.HOLD


async def test_an_offer_for_an_appointment_no_longer_active_changes_nothing(
    session: AsyncSession,
) -> None:
    """Staff cancelled the appointment between the offer and the reply. The
    patient's pick must not resurrect it as a new booking."""
    graph, original, offer = await _original_and_offer(session)
    original.status = AppointmentStatus.CANCELLED
    await session.flush()

    result = await booking.move_into_hold(
        session, clinic_id=graph.clinic_id, hold_id=offer.id, now=NOW
    )

    assert result.outcome is booking.Outcome.TAKEN
    await session.refresh(offer)
    assert offer.status is AppointmentStatus.HOLD


async def test_releasing_offers_leaves_the_appointment_and_other_holds(
    session: AsyncSession,
) -> None:
    """`release_offers` removes exactly the offers for one appointment."""
    graph, original, offer = await _original_and_offer(session)
    unrelated = appointment(
        graph,
        at(5, 11, 0),
        status=AppointmentStatus.HOLD,
        expires_at=NOW + dt.timedelta(minutes=15),
    )
    session.add(unrelated)
    await session.flush()

    released = await booking.release_offers(
        session, clinic_id=graph.clinic_id, appointment_id=original.id
    )

    assert released == 1
    remaining = set(
        (
            await session.execute(
                select(Appointment.id).where(Appointment.clinic_id == graph.clinic_id)
            )
        )
        .scalars()
        .all()
    )
    assert original.id in remaining
    assert unrelated.id in remaining
    assert offer.id not in remaining
