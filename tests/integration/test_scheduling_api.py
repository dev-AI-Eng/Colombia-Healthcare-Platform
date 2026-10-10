"""The scheduling write routes: book, hold, confirm, cancel, reschedule.

These are thin shells over `src.scheduling.booking`, which has its own tests.
What is checked here is what the HTTP layer adds and could get wrong: the
status code a lost race produces, that a refused write leaves nothing behind,
that an idempotent retry does not book twice, that one clinic cannot touch
another's rows through a route, and that every write is audited.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.audit.models import AccessLogEntry
from src.core.tenancy import ClinicScope, apply_clinic_scope
from src.scheduling.models import (
    Appointment,
    AppointmentStatus,
    Escalation,
    EscalationReason,
    EscalationStatus,
)
from tests.integration.factories import Graph, appointment, at, create_graph, scope_client

#: Well clear of the seeded data, on a Tuesday.
START = at(13, 9)
ISO = "%Y-%m-%dT%H:%M:%S%z"


def _iso(moment: datetime) -> str:
    text = moment.strftime(ISO)
    return f"{text[:-2]}:{text[-2:]}"  # -0500 -> -05:00


def _body(graph: Graph, start: datetime, **extra: object) -> dict[str, object]:
    return {
        "patient_id": str(graph.patient_id),
        "doctor_id": str(graph.doctor_id),
        "location_id": str(graph.location_id),
        "appointment_type_id": str(graph.appointment_type_id),
        "start": _iso(start),
        "duration_minutes": 20,
        **extra,
    }


async def _count(session: AsyncSession, clinic_id: uuid.UUID) -> int:
    await apply_clinic_scope(session, ClinicScope(clinic_id=clinic_id))
    return (
        await session.scalar(
            select(func.count()).select_from(Appointment).where(Appointment.clinic_id == clinic_id)
        )
        or 0
    )


async def _audit_count(session: AsyncSession) -> int:
    return await session.scalar(select(func.count()).select_from(AccessLogEntry)) or 0


@pytest.fixture
async def scoped(session: AsyncSession, client: TestClient):  # type: ignore[no-untyped-def]
    graph = await create_graph(session)
    await session.commit()
    await apply_clinic_scope(session, ClinicScope(clinic_id=graph.clinic_id))
    yield scope_client(client, graph), graph


# ---------------------------------------------------------------------- book


async def test_booking_returns_the_appointment_it_wrote(scoped) -> None:  # type: ignore[no-untyped-def]
    client, graph = scoped

    response = client.post("/review/appointments", json=_body(graph, START))

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == AppointmentStatus.SCHEDULED
    assert body["start"] == _iso(START)
    assert body["duration_minutes"] == 20
    # The names the read routes carry, so one shape serves reads and writes.
    assert body["doctor_name"]
    assert body["location_name"]
    assert body["patient_name"]


async def test_booking_the_same_slot_twice_answers_409(scoped) -> None:  # type: ignore[no-untyped-def]
    """The database refuses it; the route has to turn that into an answer.

    409 rather than 500: the request was well formed and the answer is no.
    """
    client, graph = scoped
    assert client.post("/review/appointments", json=_body(graph, START)).status_code == 201

    clash = client.post("/review/appointments", json=_body(graph, START + timedelta(minutes=10)))

    assert clash.status_code == 409, clash.text
    assert "ocupado" in clash.json()["detail"]


async def test_a_refused_booking_writes_nothing(
    scoped,  # type: ignore[no-untyped-def]
    session: AsyncSession,
) -> None:
    client, graph = scoped
    client.post("/review/appointments", json=_body(graph, START))
    before = await _count(session, graph.clinic_id)

    client.post("/review/appointments", json=_body(graph, START + timedelta(minutes=5)))

    assert await _count(session, graph.clinic_id) == before


async def test_an_idempotent_retry_does_not_book_twice(
    scoped,  # type: ignore[no-untyped-def]
    session: AsyncSession,
) -> None:
    """A dropped connection must not cost the patient a second appointment."""
    client, graph = scoped
    body = _body(graph, START, idempotency_key="whatsapp-msg-7781")

    first = client.post("/review/appointments", json=body)
    second = client.post("/review/appointments", json=body)

    assert first.status_code == 201
    # The retry found its own earlier booking: it succeeded, it just wrote nothing.
    assert second.status_code == 201, second.text
    assert second.json()["id"] == first.json()["id"]
    assert await _count(session, graph.clinic_id) == 1


async def test_back_to_back_bookings_are_allowed(scoped) -> None:  # type: ignore[no-untyped-def]
    """Half-open ranges: 09:00-09:20 and 09:20-09:40 do not overlap."""
    client, graph = scoped

    first = client.post("/review/appointments", json=_body(graph, START))
    second = client.post("/review/appointments", json=_body(graph, START + timedelta(minutes=20)))

    assert (first.status_code, second.status_code) == (201, 201), second.text


async def test_a_time_without_an_offset_is_refused(scoped) -> None:  # type: ignore[no-untyped-def]
    """A naive datetime means the zone is a guess, and a guessed time is wrong."""
    client, graph = scoped
    body = _body(graph, START) | {"start": "2026-10-13T09:00:00"}

    response = client.post("/review/appointments", json=body)

    assert response.status_code == 422
    assert "offset" in response.text


async def test_an_appointment_must_end_after_it_starts(scoped) -> None:  # type: ignore[no-untyped-def]
    client, graph = scoped
    body = _body(graph, START) | {"end": _iso(START), "duration_minutes": None}

    response = client.post("/review/appointments", json=body)

    assert response.status_code == 422
    assert "end after it starts" in response.text


# ---------------------------------------------------------------------- hold


async def test_a_hold_carries_its_expiry(scoped) -> None:  # type: ignore[no-untyped-def]
    client, graph = scoped

    response = client.post("/review/appointments/holds", json=_body(graph, START, minutes=15))

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == AppointmentStatus.HOLD
    assert body["expires_at"] is not None


async def test_a_hold_blocks_a_booking_of_the_same_time(scoped) -> None:  # type: ignore[no-untyped-def]
    """The whole point of a hold: while it lives the slot is really the patient's."""
    client, graph = scoped
    assert client.post("/review/appointments/holds", json=_body(graph, START)).status_code == 201

    clash = client.post("/review/appointments", json=_body(graph, START))

    assert clash.status_code == 409


async def test_confirming_a_hold_makes_it_a_booking(scoped) -> None:  # type: ignore[no-untyped-def]
    client, graph = scoped
    held = client.post("/review/appointments/holds", json=_body(graph, START)).json()

    confirmed = client.post(f"/review/appointments/{held['id']}/confirm")

    assert confirmed.status_code == 200, confirmed.text
    body = confirmed.json()
    assert body["id"] == held["id"]
    assert body["status"] == AppointmentStatus.SCHEDULED
    assert body["expires_at"] is None


async def test_confirming_an_expired_hold_answers_409(
    scoped,  # type: ignore[no-untyped-def]
    session: AsyncSession,
) -> None:
    """Its slot was free in the meantime and may belong to somebody else now."""
    client, graph = scoped
    stale = appointment(graph, START, status=AppointmentStatus.HOLD, expires_at=at(1, 8))
    session.add(stale)
    await session.commit()

    response = client.post(f"/review/appointments/{stale.id}/confirm")

    assert response.status_code == 409, response.text
    assert "venció" in response.json()["detail"]


# -------------------------------------------------------------------- cancel


async def test_cancelling_frees_the_slot(scoped) -> None:  # type: ignore[no-untyped-def]
    client, graph = scoped
    booked = client.post("/review/appointments", json=_body(graph, START)).json()

    cancelled = client.post(
        f"/review/appointments/{booked['id']}/cancel",
        json={"reason": "El paciente pidió otro día"},
    )

    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["status"] == AppointmentStatus.CANCELLED
    # The time is bookable again.
    assert client.post("/review/appointments", json=_body(graph, START)).status_code == 201


async def test_cancelling_keeps_the_row(
    scoped,  # type: ignore[no-untyped-def]
    session: AsyncSession,
) -> None:
    """A cancellation is part of the patient's history, and M4 reports on it."""
    client, graph = scoped
    booked = client.post("/review/appointments", json=_body(graph, START)).json()

    client.post(f"/review/appointments/{booked['id']}/cancel", json={})

    await apply_clinic_scope(session, ClinicScope(clinic_id=graph.clinic_id))
    row = await session.get(Appointment, uuid.UUID(booked["id"]))
    assert row is not None
    assert row.status == AppointmentStatus.CANCELLED


async def test_cancelling_an_unknown_appointment_answers_404(scoped) -> None:  # type: ignore[no-untyped-def]
    client, _ = scoped

    response = client.post(f"/review/appointments/{uuid.uuid4()}/cancel", json={})

    assert response.status_code == 404


# ---------------------------------------------------------------- reschedule


async def test_rescheduling_writes_a_new_row_linked_to_the_old(
    scoped,  # type: ignore[no-untyped-def]
    session: AsyncSession,
) -> None:
    client, graph = scoped
    booked = client.post("/review/appointments", json=_body(graph, START)).json()
    later = START + timedelta(days=1)

    moved = client.post(
        f"/review/appointments/{booked['id']}/reschedule",
        json={"start": _iso(later), "duration_minutes": 20},
    )

    assert moved.status_code == 201, moved.text
    body = moved.json()
    assert body["id"] != booked["id"]
    assert body["start"] == _iso(later)

    await apply_clinic_scope(session, ClinicScope(clinic_id=graph.clinic_id))
    new_row = await session.get(Appointment, uuid.UUID(body["id"]))
    assert new_row is not None
    assert new_row.rescheduled_from_appointment_id == uuid.UUID(booked["id"])


async def test_a_failed_reschedule_leaves_the_original_alone(
    scoped,  # type: ignore[no-untyped-def]
    session: AsyncSession,
) -> None:
    """The patient must not lose the appointment they had to a move that failed."""
    client, graph = scoped
    mine = client.post("/review/appointments", json=_body(graph, START)).json()
    # Somebody else already has the time we are about to ask for.
    taken = START + timedelta(days=1)
    blocker = appointment(graph, taken)
    session.add(blocker)
    await session.commit()

    response = client.post(
        f"/review/appointments/{mine['id']}/reschedule",
        json={"start": _iso(taken), "duration_minutes": 20},
    )

    assert response.status_code == 409, response.text
    await apply_clinic_scope(session, ClinicScope(clinic_id=graph.clinic_id))
    row = await session.get(Appointment, uuid.UUID(mine["id"]))
    assert row is not None
    assert row.status == AppointmentStatus.SCHEDULED, "the original was cancelled anyway"


# ------------------------------------------------------- isolation and audit


async def test_a_write_route_cannot_reach_another_clinic(
    session: AsyncSession, client: TestClient
) -> None:
    """Row-level security, through HTTP rather than through the engine."""
    theirs = await create_graph(session)
    await session.commit()
    # `create_graph` leaves the scope bound to the clinic it just made, and the
    # insert needs that scope: without it row-level security refuses the write.
    await apply_clinic_scope(session, ClinicScope(clinic_id=theirs.clinic_id))
    their_appointment = appointment(theirs, START)
    session.add(their_appointment)
    await session.commit()

    mine = await create_graph(session, name="Clínica Mía")
    await session.commit()
    scoped = scope_client(client, mine)

    response = scoped.post(f"/review/appointments/{their_appointment.id}/cancel", json={})

    assert response.status_code == 404, response.text
    await apply_clinic_scope(session, ClinicScope(clinic_id=theirs.clinic_id))
    row = await session.get(Appointment, their_appointment.id)
    assert row is not None
    assert row.status == AppointmentStatus.SCHEDULED


async def test_every_write_route_records_an_audit_entry(
    scoped,  # type: ignore[no-untyped-def]
    session: AsyncSession,
) -> None:
    """Rule 8: every patient-data write is audit-logged.

    The audit-coverage test in `test_review_api.py` walks GET routes from the
    OpenAPI document; these are POSTs with bodies, so they are covered here.
    """
    client, graph = scoped

    async def count() -> int:
        return await _audit_count(session)

    before = await count()
    booked = client.post("/review/appointments", json=_body(graph, START)).json()
    assert await count() > before, "book wrote no audit entry"

    before = await count()
    held = client.post(
        "/review/appointments/holds", json=_body(graph, START + timedelta(hours=2))
    ).json()
    assert await count() > before, "hold wrote no audit entry"

    before = await count()
    client.post(f"/review/appointments/{held['id']}/confirm")
    assert await count() > before, "confirm wrote no audit entry"

    before = await count()
    moved = client.post(
        f"/review/appointments/{booked['id']}/reschedule",
        json={"start": _iso(START + timedelta(days=2)), "duration_minutes": 20},
    ).json()
    assert await count() > before, "reschedule wrote no audit entry"

    before = await count()
    client.post(f"/review/appointments/{moved['id']}/cancel", json={})
    assert await count() > before, "cancel wrote no audit entry"


async def test_a_write_route_renders_before_the_scope_is_cleared(
    scoped,  # type: ignore[no-untyped-def]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A commit mid-request clears the clinic scope and hides the written row.

    `apply_clinic_scope` sets a transaction-local setting, so a commit before
    the response is built leaves row-level security with no scope: the read
    that follows finds nothing and a successful write answers 404. This cost
    four of these tests a 404 before the ordering was fixed, and nothing
    objected when the fix was reverted, because `get_session` commits on return
    anyway and the response object survives in memory.

    So the trap is reproduced directly: commit inside the render, as an
    explicit call would, and prove the route stops working.
    """
    client, graph = scoped

    from src.api.review import scheduling as routes

    original = routes.repository.get_appointment

    async def commit_then_read(session, **kwargs):  # type: ignore[no-untyped-def]
        await session.commit()  # what an explicit commit before the read does
        return await original(session, **kwargs)

    monkeypatch.setattr(routes.repository, "get_appointment", commit_then_read)

    response = client.post("/review/appointments", json=_body(graph, START))

    assert response.status_code == 404, (
        "Committing before the read should clear the clinic scope and hide the "
        "row. It did not, so this test no longer guards the ordering."
    )


# ---------------------------------------------------------------- escalations


async def _escalate(
    session: AsyncSession, graph: Graph, *, detail: str = "No hay citas disponibles."
) -> Escalation:
    row = Escalation(
        clinic_id=graph.clinic_id,
        patient_id=graph.patient_id,
        reason=EscalationReason.NO_ACCEPTABLE_SLOT,
        status=EscalationStatus.OPEN,
        detail_es=detail,
        context={},
    )
    session.add(row)
    await session.commit()
    await apply_clinic_scope(session, ClinicScope(clinic_id=graph.clinic_id))
    return row


async def test_the_queue_lists_an_open_escalation_with_its_spanish_message(
    scoped,  # type: ignore[no-untyped-def]
    session: AsyncSession,
) -> None:
    """What a receptionist reads. The Spanish sentence is the actionable part."""
    client, graph = scoped
    await _escalate(session, graph, detail="No hay citas disponibles esta semana.")

    body = client.get("/review/escalations").json()

    assert body["total"] == 1
    (row,) = body["items"]
    assert row["status"] == "open"
    assert row["reason"] == EscalationReason.NO_ACCEPTABLE_SLOT
    assert row["detail_es"] == "No hay citas disponibles esta semana."
    assert row["patient_name"], "a receptionist needs to know who it is about"


async def test_resolving_an_escalation_closes_it_and_records_when(
    scoped,  # type: ignore[no-untyped-def]
    session: AsyncSession,
) -> None:
    client, graph = scoped
    row = await _escalate(session, graph)

    resolved = client.post(f"/review/escalations/{row.id}/resolve")

    assert resolved.status_code == 200, resolved.text
    body = resolved.json()
    assert body["status"] == "resolved"
    assert body["resolved_at"] is not None


async def test_an_escalation_is_never_deleted_by_resolving_it(
    scoped,  # type: ignore[no-untyped-def]
    session: AsyncSession,
) -> None:
    """Nothing leaves the queue without a record of who closed it and when."""
    client, graph = scoped
    row = await _escalate(session, graph)

    client.post(f"/review/escalations/{row.id}/resolve")

    still_there = client.get("/review/escalations").json()
    assert still_there["total"] == 1
    assert still_there["items"][0]["status"] == "resolved"


async def test_resolving_the_same_escalation_twice_answers_404(
    scoped,  # type: ignore[no-untyped-def]
    session: AsyncSession,
) -> None:
    """The first person dealt with it; that is the time that matters."""
    client, graph = scoped
    row = await _escalate(session, graph)

    first = client.post(f"/review/escalations/{row.id}/resolve")
    second = client.post(f"/review/escalations/{row.id}/resolve")

    assert first.status_code == 200
    assert second.status_code == 404, second.text


async def test_the_queue_can_be_filtered_to_what_is_still_open(
    scoped,  # type: ignore[no-untyped-def]
    session: AsyncSession,
) -> None:
    client, graph = scoped
    done = await _escalate(session, graph, detail="Ya resuelta.")
    await _escalate(session, graph, detail="Todavía pendiente.")
    client.post(f"/review/escalations/{done.id}/resolve")

    open_only = client.get("/review/escalations", params={"status": "open"}).json()

    assert open_only["total"] == 1
    assert open_only["items"][0]["detail_es"] == "Todavía pendiente."


async def test_one_clinic_cannot_resolve_another_clinics_escalation(
    session: AsyncSession, client: TestClient
) -> None:
    theirs = await create_graph(session)
    await session.commit()
    await apply_clinic_scope(session, ClinicScope(clinic_id=theirs.clinic_id))
    row = await _escalate(session, theirs)

    mine = await create_graph(session, name="Clínica Mía")
    await session.commit()
    scoped = scope_client(client, mine)

    response = scoped.post(f"/review/escalations/{row.id}/resolve")

    assert response.status_code == 404, response.text
