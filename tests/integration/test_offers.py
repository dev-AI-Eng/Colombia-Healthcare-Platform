"""Offering slots, and what happens when there are none.

PDF criterion 2 for M2: "fallback triggers when no slot satisfies the
patient". The test that matters is `test_no_acceptable_slot_escalates_to_a_human`
-- a fallback that only sends a message is a dead end, because nothing records
that the patient is still waiting and nobody is told to call them.
"""

from __future__ import annotations

import datetime as dt
import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.audit.context import AuditContext
from src.audit.models import AccessLogEntry
from src.core.timezones import BOGOTA
from src.scheduling import booking, offers
from src.scheduling.models import (
    Appointment,
    AppointmentStatus,
    AvailabilityException,
    AvailabilityRule,
    Escalation,
    EscalationReason,
    EscalationStatus,
    ExceptionKind,
)
from src.scheduling.offers import SlotSearch
from tests.integration.factories import Graph, create_graph

#: Monday 5 October 2026 through Friday the 9th. No Colombian holiday falls in
#: that week, so a holiday cannot silently empty the search.
MONDAY = dt.date(2026, 10, 5)
FRIDAY = dt.date(2026, 10, 9)
#: Before any slot in the week, so "already in the past" never interferes.
NOW = dt.datetime(2026, 10, 1, 12, tzinfo=BOGOTA)


@pytest.fixture(autouse=True)
def _audited(staff_context: AuditContext) -> AuditContext:
    """Escalations name a patient, so writing one needs an actor bound."""
    return staff_context


async def _weekday_mornings(session: AsyncSession, graph: Graph) -> None:
    """The doctor works 08:00-12:00, Monday to Friday, all of 2026."""
    for weekday in range(5):
        session.add(
            AvailabilityRule(
                clinic_id=graph.clinic_id,
                doctor_id=graph.doctor_id,
                location_id=graph.location_id,
                weekday=weekday,
                start_time=dt.time(8),
                end_time=dt.time(12),
                valid_from=dt.date(2026, 1, 1),
            )
        )
    await session.flush()


def _search(graph: Graph, **kwargs) -> SlotSearch:  # type: ignore[no-untyped-def]
    """A search over the test week, with any field overridable."""
    fields: dict[str, object] = {
        "clinic_id": graph.clinic_id,
        "doctor_id": graph.doctor_id,
        "appointment_type_id": graph.appointment_type_id,
        "start_date": MONDAY,
        "end_date": FRIDAY,
    }
    fields.update(kwargs)
    return SlotSearch(**fields)  # type: ignore[arg-type]


# ------------------------------------------------------------- finding slots
async def test_a_doctor_with_a_schedule_has_slots_to_offer(
    session: AsyncSession,
) -> None:
    graph = await create_graph(session)
    await _weekday_mornings(session, graph)
    slots = await offers.find_slots(session, _search(graph), now=NOW)
    assert len(slots) == offers.DEFAULT_OFFER_COUNT
    assert all(s.local_start.hour >= 8 for s in slots)


async def test_a_doctor_with_no_schedule_has_nothing_to_offer(
    session: AsyncSession,
) -> None:
    """An empty list, not an exception: "no appointment" is an answer."""
    graph = await create_graph(session)
    assert await offers.find_slots(session, _search(graph), now=NOW) == []


async def test_an_existing_appointment_is_not_offered_again(
    session: AsyncSession,
) -> None:
    """The search must agree with the constraint, or the patient picks a slot
    that the database then refuses."""
    graph = await create_graph(session)
    await _weekday_mornings(session, graph)
    first = (await offers.find_slots(session, _search(graph), now=NOW))[0]

    booked = await booking.book(
        session,
        clinic_id=graph.clinic_id,
        patient_id=graph.patient_id,
        doctor_id=graph.doctor_id,
        location_id=graph.location_id,
        appointment_type_id=graph.appointment_type_id,
        start=first.start,
        end=first.end,
    )
    assert booked.ok

    again = await offers.find_slots(session, _search(graph), now=NOW)
    assert first.start not in [s.start for s in again]


async def test_a_live_hold_is_not_offered_to_the_next_patient(
    session: AsyncSession,
) -> None:
    """ADR-17: a hold occupies the time exactly as a booking does."""
    graph = await create_graph(session)
    await _weekday_mornings(session, graph)
    first = (await offers.find_slots(session, _search(graph), now=NOW))[0]

    held = await booking.hold(
        session,
        clinic_id=graph.clinic_id,
        patient_id=graph.patient_id,
        doctor_id=graph.doctor_id,
        location_id=graph.location_id,
        appointment_type_id=graph.appointment_type_id,
        start=first.start,
        end=first.end,
        now=NOW,
    )
    assert held.ok
    again = await offers.find_slots(session, _search(graph), now=NOW)
    assert first.start not in [s.start for s in again]


async def test_a_day_the_doctor_is_away_offers_nothing(session: AsyncSession) -> None:
    graph = await create_graph(session)
    await _weekday_mornings(session, graph)
    session.add(
        AvailabilityException(
            clinic_id=graph.clinic_id,
            doctor_id=graph.doctor_id,
            on_date=MONDAY,
            kind=ExceptionKind.UNAVAILABLE,
        )
    )
    await session.flush()

    monday_only = _search(graph, end_date=MONDAY)  # type: ignore[arg-type]
    assert await offers.find_slots(session, monday_only, now=NOW) == []


# ------------------------------------------------- what the patient can manage
async def test_a_patient_who_can_only_do_afternoons_is_offered_nothing(
    session: AsyncSession,
) -> None:
    """The doctor works mornings; the filter is applied in clinic-local time."""
    graph = await create_graph(session)
    await _weekday_mornings(session, graph)
    afternoons = _search(graph, earliest=dt.time(14))  # type: ignore[arg-type]
    assert await offers.find_slots(session, afternoons, now=NOW) == []


async def test_a_weekday_filter_narrows_the_offer(session: AsyncSession) -> None:
    graph = await create_graph(session)
    await _weekday_mornings(session, graph)
    wednesdays = _search(graph, weekdays=frozenset({2}))  # type: ignore[arg-type]
    slots = await offers.find_slots(session, wednesdays, now=NOW)
    assert slots, "the doctor does work Wednesdays"
    assert {s.local_start.weekday() for s in slots} == {2}


# ------------------------------------------------- the criterion that matters
async def test_offering_holds_every_slot_it_shows(session: AsyncSession) -> None:
    """An offer the patient cannot accept is worse than no offer (ADR-17)."""
    graph = await create_graph(session)
    await _weekday_mornings(session, graph)

    result = await offers.offer(
        session,
        _search(graph),
        patient_id=graph.patient_id,
        location_id=graph.location_id,
        now=NOW,
    )
    assert result.offered_any
    assert result.escalation is None
    assert len(result.held) == offers.DEFAULT_OFFER_COUNT

    holds = (
        await session.execute(
            select(func.count())
            .select_from(Appointment)
            .where(Appointment.status == AppointmentStatus.HOLD)
        )
    ).scalar_one()
    assert holds == offers.DEFAULT_OFFER_COUNT


async def test_no_acceptable_slot_escalates_to_a_human(session: AsyncSession) -> None:
    """PDF criterion 2. The fallback has to leave a record staff work from.

    Replying to the patient and stopping there is a dead end: nobody is told
    they are still waiting, and the request is lost the moment the
    conversation ends.
    """
    graph = await create_graph(session)
    await _weekday_mornings(session, graph)

    impossible = _search(graph, earliest=dt.time(14))  # type: ignore[arg-type]
    result = await offers.offer(
        session,
        impossible,
        patient_id=graph.patient_id,
        location_id=graph.location_id,
        now=NOW,
    )

    assert not result.offered_any
    assert result.held == []
    assert result.escalation is not None
    assert result.escalation.reason is EscalationReason.NO_ACCEPTABLE_SLOT
    assert result.escalation.status is EscalationStatus.OPEN


async def test_the_escalation_says_what_was_looked_for(session: AsyncSession) -> None:
    """A queue entry a receptionist cannot act on is noise.

    The Spanish names the dates; the context carries the search so a screen can
    show it without re-running anything.
    """
    graph = await create_graph(session)
    await _weekday_mornings(session, graph)
    result = await offers.offer(
        session,
        _search(graph, earliest=dt.time(14)),  # type: ignore[arg-type]
        patient_id=graph.patient_id,
        location_id=graph.location_id,
        now=NOW,
    )
    assert result.escalation is not None
    assert "05/10/2026" in result.escalation.detail_es
    assert "09/10/2026" in result.escalation.detail_es
    assert result.escalation.context["earliest"] == "14:00:00"
    assert result.escalation.context["doctor_id"] == str(graph.doctor_id)


async def test_an_escalation_is_audited_against_the_patient(
    session: AsyncSession,
) -> None:
    """It names the patient and what they wanted, so it is patient data."""
    graph = await create_graph(session)
    await _weekday_mornings(session, graph)
    before = (await session.execute(select(func.count()).select_from(AccessLogEntry))).scalar_one()

    await offers.offer(
        session,
        _search(graph, earliest=dt.time(14)),  # type: ignore[arg-type]
        patient_id=graph.patient_id,
        location_id=graph.location_id,
        now=NOW,
    )
    after = (await session.execute(select(func.count()).select_from(AccessLogEntry))).scalar_one()
    assert after == before + 1


async def test_a_fully_booked_week_escalates_rather_than_offering_nothing(
    session: AsyncSession,
) -> None:
    """The realistic version: the doctor has a schedule, all of it is taken."""
    graph = await create_graph(session)
    session.add(
        AvailabilityRule(
            clinic_id=graph.clinic_id,
            doctor_id=graph.doctor_id,
            location_id=graph.location_id,
            weekday=0,  # Monday only
            start_time=dt.time(8),
            end_time=dt.time(9),
            valid_from=dt.date(2026, 1, 1),
        )
    )
    await session.flush()

    monday_only = _search(graph, end_date=MONDAY)  # type: ignore[arg-type]
    slots = await offers.find_slots(session, monday_only, now=NOW, limit=10)
    assert slots, "one hour of availability"
    for slot in slots:
        taken = await booking.book(
            session,
            clinic_id=graph.clinic_id,
            patient_id=graph.patient_id,
            doctor_id=graph.doctor_id,
            location_id=graph.location_id,
            appointment_type_id=graph.appointment_type_id,
            start=slot.start,
            end=slot.end,
        )
        assert taken.ok

    result = await offers.offer(
        session,
        monday_only,
        patient_id=graph.patient_id,
        location_id=graph.location_id,
        now=NOW,
    )
    assert result.escalation is not None
    assert result.escalation.reason is EscalationReason.NO_ACCEPTABLE_SLOT


async def test_one_escalation_row_per_failed_search(session: AsyncSession) -> None:
    """Two failed attempts are two things for staff to see, not one."""
    graph = await create_graph(session)
    await _weekday_mornings(session, graph)
    impossible = _search(graph, earliest=dt.time(14))  # type: ignore[arg-type]
    for _ in range(2):
        await offers.offer(
            session,
            impossible,
            patient_id=graph.patient_id,
            location_id=graph.location_id,
            now=NOW,
        )
    count = (await session.execute(select(func.count()).select_from(Escalation))).scalar_one()
    assert count == 2


async def test_a_slot_taken_between_search_and_hold_is_dropped_not_fatal(
    session: AsyncSession,
) -> None:
    """Another conversation can take a slot while this one is deciding.

    The lost slot leaves the offer; the rest stand. Escalating because one of
    three disappeared would be wrong.
    """
    graph = await create_graph(session)
    await _weekday_mornings(session, graph)
    search = _search(graph)
    first = (await offers.find_slots(session, search, now=NOW))[0]
    stolen = await booking.book(
        session,
        clinic_id=graph.clinic_id,
        patient_id=graph.patient_id,
        doctor_id=graph.doctor_id,
        location_id=graph.location_id,
        appointment_type_id=graph.appointment_type_id,
        start=first.start,
        end=first.end,
    )
    assert stolen.ok

    result = await offers.offer(
        session,
        search,
        patient_id=graph.patient_id,
        location_id=graph.location_id,
        now=NOW,
    )
    assert result.offered_any
    assert result.escalation is None
    assert first.start not in [r.appointment.during.lower for r in result.held]  # type: ignore[union-attr]


async def test_an_unknown_appointment_type_is_refused(session: AsyncSession) -> None:
    """A caller passing a type that does not exist is a bug, not a no-slot case.

    Returning an empty list would make a programming error look like a full
    diary, and the patient would be escalated for the wrong reason.
    """
    graph = await create_graph(session)
    await _weekday_mornings(session, graph)
    nonexistent = SlotSearch(
        clinic_id=graph.clinic_id,
        doctor_id=graph.doctor_id,
        appointment_type_id=uuid.uuid4(),
        start_date=MONDAY,
        end_date=FRIDAY,
    )
    with pytest.raises(ValueError, match="no appointment type"):
        await offers.find_slots(session, nonexistent, now=NOW)
