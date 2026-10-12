"""The slot-search route: what a doctor can genuinely offer.

M2's availability decision is that slots are computed, never stored, because a
stored table is correct right up until a doctor changes their hours or a
holiday is declared. These check that the route behaves as the decision
promises: a rule produces slots, an exception removes them, a public holiday
removes them, and an existing appointment removes the time it occupies.

The engine itself has its own unit tests. What is checked here is that the HTTP
route is wired to the engine rather than to something that looks like it.
"""

from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.audit.context import AuditContext
from src.core.tenancy import ClinicScope, apply_clinic_scope
from src.scheduling.models import (
    AvailabilityException,
    AvailabilityRule,
    ExceptionKind,
)
from tests.integration.factories import Graph, appointment, create_graph, scope_client

#: A Monday in March 2027, well clear of the seeded data and of any holiday.
MONDAY = dt.date(2027, 3, 1)

#: Día de San José, moved to Monday 22 March 2027 by Ley 51 de 1983. Chosen
#: because it is a holiday the "Emiliani" law shifts, so a hard-coded list
#: would get it wrong.
HOLIDAY_MONDAY = dt.date(2027, 3, 22)


@pytest.fixture(autouse=True)
def _audited(staff_context: AuditContext) -> AuditContext:
    return staff_context


@pytest.fixture
async def scoped(session: AsyncSession, client: TestClient):  # type: ignore[no-untyped-def]
    graph = await create_graph(session)
    await session.commit()
    await apply_clinic_scope(session, ClinicScope(clinic_id=graph.clinic_id))
    yield scope_client(client, graph), graph


async def _rule(
    session: AsyncSession,
    graph: Graph,
    *,
    weekday: int,
    start: dt.time = dt.time(8, 0),
    end: dt.time = dt.time(12, 0),
) -> AvailabilityRule:
    rule = AvailabilityRule(
        clinic_id=graph.clinic_id,
        doctor_id=graph.doctor_id,
        location_id=graph.location_id,
        weekday=weekday,
        start_time=start,
        end_time=end,
        valid_from=dt.date(2026, 1, 1),
    )
    session.add(rule)
    await session.commit()
    await apply_clinic_scope(session, ClinicScope(clinic_id=graph.clinic_id))
    return rule


def _search(graph: Graph, start: dt.date, end: dt.date, **extra: object) -> dict[str, object]:
    return {
        "doctor_id": str(graph.doctor_id),
        "appointment_type_id": str(graph.appointment_type_id),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        **extra,
    }


async def test_a_weekly_rule_produces_slots(scoped, session: AsyncSession) -> None:  # type: ignore[no-untyped-def]
    client, graph = scoped
    await _rule(session, graph, weekday=MONDAY.weekday())

    response = client.get("/review/availability/slots", params=_search(graph, MONDAY, MONDAY))

    assert response.status_code == 200, response.text
    slots = response.json()
    assert slots, "a rule covering this day should produce slots"
    # Inside the rule's window, in Colombia's zone, and the type's length.
    first = slots[0]
    assert first["start"].startswith(MONDAY.isoformat())
    assert first["start"].endswith("-05:00")
    assert first["duration_minutes"] == 20


async def test_no_rule_means_no_slots(scoped) -> None:  # type: ignore[no-untyped-def]
    """An empty list is an answer, not a failure.

    It is what the conversation graph escalates on, and what a receptionist
    sees when a week is full.
    """
    client, graph = scoped

    response = client.get("/review/availability/slots", params=_search(graph, MONDAY, MONDAY))

    assert response.status_code == 200
    assert response.json() == []


async def test_a_dated_exception_removes_the_day(scoped, session: AsyncSession) -> None:  # type: ignore[no-untyped-def]
    """A doctor taking a morning off is the case a stored slot table misses."""
    client, graph = scoped
    await _rule(session, graph, weekday=MONDAY.weekday())
    before = client.get("/review/availability/slots", params=_search(graph, MONDAY, MONDAY)).json()
    assert before

    session.add(
        AvailabilityException(
            clinic_id=graph.clinic_id,
            doctor_id=graph.doctor_id,
            on_date=MONDAY,
            kind=ExceptionKind.UNAVAILABLE,
        )
    )
    await session.commit()
    await apply_clinic_scope(session, ClinicScope(clinic_id=graph.clinic_id))

    after = client.get("/review/availability/slots", params=_search(graph, MONDAY, MONDAY)).json()

    assert after == [], "the exception should have cleared the day"


async def test_a_colombian_holiday_removes_the_day(scoped, session: AsyncSession) -> None:  # type: ignore[no-untyped-def]
    """The clinic is closed, and no rule says so: the holiday library does.

    Día de San José falls on 19 March 2027, a Friday, and Ley 51 de 1983 moves
    it to Monday the 22nd. Which holidays move is not derivable from a rule,
    which is why a maintained library decides rather than a list in the code.
    """
    client, graph = scoped
    await _rule(session, graph, weekday=HOLIDAY_MONDAY.weekday())

    # The same weekday a week earlier, to prove the rule itself works.
    ordinary = HOLIDAY_MONDAY - dt.timedelta(days=7)
    assert client.get(
        "/review/availability/slots", params=_search(graph, ordinary, ordinary)
    ).json(), "the rule should cover an ordinary Monday"

    holiday = client.get(
        "/review/availability/slots", params=_search(graph, HOLIDAY_MONDAY, HOLIDAY_MONDAY)
    ).json()

    assert holiday == [], "a public holiday should leave no slots"


async def test_an_existing_appointment_removes_its_time(  # type: ignore[no-untyped-def]
    scoped, session: AsyncSession
) -> None:
    """What is already booked is not offerable, which is the point."""
    client, graph = scoped
    await _rule(session, graph, weekday=MONDAY.weekday())
    free = client.get("/review/availability/slots", params=_search(graph, MONDAY, MONDAY)).json()
    assert free

    taken_start = dt.datetime.fromisoformat(free[0]["start"])
    session.add(appointment(graph, taken_start))
    await session.commit()
    await apply_clinic_scope(session, ClinicScope(clinic_id=graph.clinic_id))

    after = client.get("/review/availability/slots", params=_search(graph, MONDAY, MONDAY)).json()

    assert free[0]["start"] not in [slot["start"] for slot in after]


async def test_the_patient_can_narrow_the_hours(scoped, session: AsyncSession) -> None:  # type: ignore[no-untyped-def]
    """ "Afternoons only" is the constraint that produces M2's fallback."""
    client, graph = scoped
    await _rule(session, graph, weekday=MONDAY.weekday(), start=dt.time(8, 0), end=dt.time(12, 0))

    afternoon = client.get(
        "/review/availability/slots",
        params=_search(graph, MONDAY, MONDAY, earliest="14:00", latest="17:00"),
    ).json()

    # The doctor works mornings and the patient can only do afternoons, so
    # there is nothing to offer -- which is what the escalation exists for.
    assert afternoon == []


async def test_a_backwards_date_range_is_refused(scoped) -> None:  # type: ignore[no-untyped-def]
    client, graph = scoped

    response = client.get(
        "/review/availability/slots",
        params=_search(graph, MONDAY, MONDAY - dt.timedelta(days=1)),
    )

    assert response.status_code == 422
    assert "anterior" in response.json()["detail"]


async def test_one_clinic_cannot_see_another_doctors_slots(
    session: AsyncSession, client: TestClient
) -> None:
    """Row-level security, through the route rather than the engine."""
    theirs = await create_graph(session)
    await session.commit()
    await apply_clinic_scope(session, ClinicScope(clinic_id=theirs.clinic_id))
    await _rule(session, theirs, weekday=MONDAY.weekday())

    mine = await create_graph(session, name="Clínica Mía")
    await session.commit()
    scoped_client = scope_client(client, mine)

    response = scoped_client.get(
        "/review/availability/slots", params=_search(theirs, MONDAY, MONDAY)
    )

    # 404 rather than an empty list, and that is a stronger result: row-level
    # security hides the other clinic's appointment type, so the request fails
    # before any slot is computed. An empty list would also have been safe, but
    # this proves the scope stopped it rather than the engine finding nothing.
    assert response.status_code == 404, response.text
    assert response.json() != []
