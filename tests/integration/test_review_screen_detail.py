"""The review screen tells the reviewer what happened and what to type.

Three findings from the M1 QA pass, all of them about a receptionist being left
without information the system already had.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.core.tenancy import ClinicScope, apply_clinic_scope
from tests.integration.factories import create_graph, scope_client

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "onboarding"


@pytest.fixture
async def scoped(session, client: TestClient) -> Iterator[TestClient]:  # type: ignore[no-untyped-def]
    graph = await create_graph(session)
    await session.commit()
    await apply_clinic_scope(session, ClinicScope(clinic_id=graph.clinic_id))
    yield scope_client(client, graph)


def _upload(client: TestClient, name: str) -> str:
    with (FIXTURES / name).open("rb") as handle:
        response = client.post("/onboarding/uploads", files={"file": (name, handle)})
    assert response.status_code == 201, response.text
    return str(response.json()["session_id"])


async def test_the_screen_says_what_the_import_did(scoped: TestClient) -> None:
    """ "Committed" alone does not distinguish 8 new patients from 8 overwritten."""
    session_id = _upload(scoped, "1_clean_ips.xlsx")
    scoped.post(f"/onboarding/uploads/{session_id}/review/validate", follow_redirects=False)
    scoped.post(f"/onboarding/uploads/{session_id}/review/commit", follow_redirects=False)

    page = scoped.get(f"/onboarding/uploads/{session_id}/review").text
    assert "has been committed" in page
    assert "new record(s) written" in page, "the screen did not say how many were written"

    # The same file again: now every row replaces one, which must be named.
    second = _upload(scoped, "1_clean_ips.xlsx")
    scoped.post(f"/onboarding/uploads/{second}/review/validate", follow_redirects=False)
    scoped.post(f"/onboarding/uploads/{second}/review/commit", follow_redirects=False)

    page = scoped.get(f"/onboarding/uploads/{second}/review").text
    assert "replaced" in page, "an overwrite was not reported to the reviewer"
    assert "overwritten" in page


async def test_a_name_question_says_which_order_to_type(scoped: TestClient) -> None:
    """An empty box is why a receptionist stalls: nothing said "Surnames, Given names"."""
    session_id = _upload(scoped, "2_receptionist.xlsx")
    scoped.post(
        f"/onboarding/uploads/{session_id}/review/AGOSTO (viejo)",
        data={"entity": "skip"},
        follow_redirects=False,
    )
    scoped.post(
        f"/onboarding/uploads/{session_id}/review/AGENDA",
        data={"entity": "patient", "exclude": "15, 16"},
        follow_redirects=False,
    )
    scoped.post(f"/onboarding/uploads/{session_id}/review/validate", follow_redirects=False)

    page = scoped.get(f"/onboarding/uploads/{session_id}/review").text
    assert "Rows needing an answer" in page
    assert "Surnames, Given names" in page, "the name field gave no hint of the format"


async def test_saving_a_sheet_returns_to_that_sheet(scoped: TestClient) -> None:
    """Every action used to return the reviewer to the top of a long page."""
    session_id = _upload(scoped, "2_receptionist.xlsx")
    response = scoped.post(
        f"/onboarding/uploads/{session_id}/review/AGENDA",
        data={"entity": "patient"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"].endswith("#sheet-agenda"), response.headers["location"]

    page = scoped.get(f"/onboarding/uploads/{session_id}/review").text
    assert 'id="sheet-agenda"' in page, "the anchor the redirect points at does not exist"
