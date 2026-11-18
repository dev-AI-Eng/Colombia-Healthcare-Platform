"""The conversation route the dashboard's simulator calls, over HTTP.

The graph is tested node by node in `test_conversation_graph.py`; this is the
one place a typed phone number enters the pipeline, so it is where §7's rule
about canonicalising before a blind-index lookup has to hold.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.crypto import blind_index
from src.core.tenancy import ClinicScope, apply_clinic_scope
from src.identity.models import BindingRelationship, PhoneBinding
from tests.integration.factories import PHONE, create_graph, scope_client

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
