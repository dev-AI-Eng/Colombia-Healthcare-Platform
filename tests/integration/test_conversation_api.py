"""The conversation route the dashboard's simulator calls, over HTTP.

The graph is tested node by node in `test_conversation_graph.py`; this is the
one place a typed phone number enters the pipeline, so it is where §7's rule
about canonicalising before a blind-index lookup has to hold.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.review import live
from src.core.crypto import blind_index
from src.core.tenancy import ClinicScope, apply_clinic_scope
from src.identity.models import BindingRelationship, PhoneBinding
from tests.integration.factories import PHONE, appointment, at, create_graph, scope_client

TURN = "/review/conversation/turn"


@pytest.fixture
async def scoped(session: AsyncSession, client: TestClient):  # type: ignore[no-untyped-def]
    """A consenting patient reachable on `PHONE`, and a client pinned to their clinic."""
    graph = await create_graph(session, consent=True)
    session.add(
        PhoneBinding(
            clinic_id=graph.clinic_id,
            phone_e164_bidx=blind_index(PHONE),
            patient_id=graph.patient_id,
            relationship_kind=BindingRelationship.SELF,
        )
    )
    await session.commit()
    await apply_clinic_scope(session, ClinicScope(clinic_id=graph.clinic_id))
    yield scope_client(client, graph), graph


@pytest.mark.parametrize("typed", [PHONE, "3001112233", "300 111 2233", "+57 300 111 2233"])
async def test_the_sender_is_found_however_the_number_is_typed(
    scoped,  # type: ignore[no-untyped-def]
    typed: str,
) -> None:
    """Found in review: only the exact E.164 spelling resolved, so the same
    patient typed as "300 111 2233" was answered as an unknown number -- the
    empty result §7 warns reads as "no such patient"."""
    client, _ = scoped

    out = client.post(TURN, json={"sender_phone": typed, "text": "hola"})

    assert out.status_code == 200, out.text
    assert out.json()["patient_resolved"] is True, f"{typed!r} was not recognised"


async def test_a_number_that_is_not_a_phone_is_refused_not_treated_as_a_stranger(
    scoped,  # type: ignore[no-untyped-def]
) -> None:
    client, _ = scoped

    out = client.post(TURN, json={"sender_phone": "12345", "text": "hola"})

    assert out.status_code == 422


def _listen(clinic_id) -> asyncio.Queue[str]:  # type: ignore[no-untyped-def]
    """Register a dashboard's queue directly: `TestClient` cannot hold a stream
    open while making a second request, and the queue is where a notice is
    observable."""
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=live.QUEUE_LIMIT)
    live._listeners[clinic_id] = {queue}
    return queue


@pytest.mark.parametrize("text", ["Si, confirmo", "cancelar", "Quiero otra fecha"])
async def test_a_turn_that_changes_the_diary_tells_the_open_dashboards(
    scoped,  # type: ignore[no-untyped-def]
    session: AsyncSession,
    text: str,
) -> None:
    """Found in the delivery campaign: every other write route notifies, and the
    conversation, which confirms, cancels and moves appointments, did not, so a
    dashboard left open showed the old diary until its fallback refresh."""
    client, graph = scoped
    session.add(appointment(graph, at(3, 9)))
    await session.commit()
    queue = _listen(graph.clinic_id)
    try:
        out = client.post(TURN, json={"sender_phone": PHONE, "text": text})

        assert out.status_code == 200, out.text
        assert queue.qsize() >= 1, f"{text!r} changed the diary and told nobody"
        assert json.loads(queue.get_nowait())["topic"] == live.TOPIC_APPOINTMENTS
    finally:
        live._listeners.pop(graph.clinic_id, None)


async def test_a_turn_that_opens_a_case_tells_the_open_dashboards(
    scoped,  # type: ignore[no-untyped-def]
    session: AsyncSession,
) -> None:
    client, graph = scoped
    session.add(appointment(graph, at(3, 9)))
    await session.commit()
    queue = _listen(graph.clinic_id)
    try:
        out = client.post(TURN, json={"sender_phone": PHONE, "text": "Me duele mucho el pecho"})

        assert out.status_code == 200 and out.json()["escalation_id"], out.text
        topics = [json.loads(queue.get_nowait())["topic"] for _ in range(queue.qsize())]
        assert live.TOPIC_ESCALATIONS in topics, topics
        assert live.TOPIC_APPOINTMENTS not in topics, "a clinical message does not change the diary"
    finally:
        live._listeners.pop(graph.clinic_id, None)


async def test_a_message_that_changes_nothing_tells_nobody(
    scoped,  # type: ignore[no-untyped-def]
) -> None:
    client, graph = scoped
    queue = _listen(graph.clinic_id)
    try:
        client.post(TURN, json={"sender_phone": PHONE, "text": "hola"})

        assert queue.qsize() == 0
    finally:
        live._listeners.pop(graph.clinic_id, None)
