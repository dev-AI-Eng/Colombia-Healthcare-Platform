"""Who sees a patient's identifiers, and what the log says about it.

Masked is the floor: an unauthenticated caller sees what the API has always
returned. Unmasking requires a signed session, and every unmasked read writes
its own audit action naming the fields uncovered, so the log can answer "who
has seen this patient's cédula" rather than only "who opened the screen".

The negative cases are the point. A forged cookie, an expired one, a tampered
role and an unset password must all land on masked, because every one of those
is a path somebody could take to unmask a clinic's patients.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.review.auth import (
    SESSION_COOKIE,
    UNMASKED_FIELDS,
    Role,
    issue_token,
    read_token,
    verify_password,
)
from src.audit.models import AccessLogEntry
from src.core.config import Settings
from tests.integration.factories import (
    DOCUMENT,
    PHONE,
    Graph,
    add_patient_records,
    create_graph,
    scope_client,
)

PASSWORD = "una-contraseña-de-prueba"


@pytest.fixture
def admin_settings(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Settings with an admin password configured, as a deployment would have."""
    monkeypatch.setattr(
        type(settings),
        "review_admin_password",
        property(lambda _self: _Secret(PASSWORD)),
        raising=False,
    )
    return settings


class _Secret:
    """Minimal stand-in for pydantic's SecretStr."""

    def __init__(self, value: str) -> None:
        self._value = value

    def get_secret_value(self) -> str:
        return self._value


@pytest.fixture
async def prepared(session: AsyncSession, client: TestClient) -> tuple[Graph, TestClient]:
    graph = await create_graph(session)
    await add_patient_records(session, graph)
    return graph, scope_client(client, graph)


async def _disclosures(session: AsyncSession) -> int:
    return int(
        await session.scalar(
            select(func.count())
            .select_from(AccessLogEntry)
            .where(AccessLogEntry.action == "disclose")
        )
        or 0
    )


# ----------------------------------------------------------- the default: masked


async def test_without_a_session_identifiers_are_masked(prepared) -> None:  # type: ignore[no-untyped-def]
    """The floor. This is what the API returned before roles existed."""
    _graph, client = prepared

    body = client.get("/review/patients").json()
    patient = body["items"][0]

    assert patient["identifiers_masked"] is True
    assert patient["document_number_masked"].startswith("*")
    assert DOCUMENT not in patient["document_number_masked"]
    assert PHONE not in str(patient["phone_masked"])


async def test_a_masked_read_records_no_disclosure(
    prepared,  # type: ignore[no-untyped-def]
    session: AsyncSession,
) -> None:
    """Only an unmasked read is a disclosure; otherwise the log means nothing."""
    _graph, client = prepared
    before = await _disclosures(session)

    client.get("/review/patients")

    assert await _disclosures(session) == before


# ------------------------------------------------------------------ signing in


async def test_the_right_password_returns_an_admin_session(
    prepared,  # type: ignore[no-untyped-def]
    admin_settings: Settings,
) -> None:
    _graph, client = prepared

    response = client.post("/review/session", json={"password": PASSWORD})

    assert response.status_code == 200, response.text
    assert response.json()["role"] == Role.ADMIN
    assert SESSION_COOKIE in response.cookies


async def test_the_wrong_password_is_refused_without_saying_why(
    prepared,  # type: ignore[no-untyped-def]
    admin_settings: Settings,
) -> None:
    _graph, client = prepared

    response = client.post("/review/session", json={"password": "incorrecta"})

    assert response.status_code == 401
    assert SESSION_COOKIE not in response.cookies
    # The same sentence whatever was wrong with it.
    assert "incorrecta" in response.json()["detail"].lower()


async def test_no_admin_password_configured_means_no_admin_sign_in(
    prepared,  # type: ignore[no-untyped-def]
) -> None:
    """A deployment that forgot to configure one must not get an open door."""
    _graph, client = prepared

    assert client.post("/review/session", json={"password": ""}).status_code in (401, 422)
    assert client.post("/review/session", json={"password": "cualquiera"}).status_code == 401


def test_an_empty_password_never_matches_an_unset_one(settings: Settings) -> None:
    """The case that makes the unset-password guard load bearing.

    `compare_digest(b"", b"")` is True, so without an explicit check for an
    unset password, submitting an empty one would unlock the admin role on any
    deployment that never configured it. The route's `min_length=1` stops an
    empty body reaching here over HTTP, but a second caller of
    `verify_password` would not have that -- so the guard belongs in the
    function and is tested at the function.
    """
    assert settings.review_admin_password.get_secret_value() == "", (
        "this test is only meaningful with no password configured"
    )

    assert verify_password("", settings) is False
    assert verify_password("cualquiera", settings) is False


async def test_signing_out_clears_the_session(
    prepared,  # type: ignore[no-untyped-def]
    admin_settings: Settings,
) -> None:
    _graph, client = prepared
    client.post("/review/session", json={"password": PASSWORD})
    assert client.get("/review/session").json()["role"] == Role.ADMIN

    client.delete("/review/session")

    assert client.get("/review/session").json()["role"] == Role.STAFF


# ------------------------------------------------------- what an admin can see


async def test_an_admin_sees_full_identifiers(
    prepared,  # type: ignore[no-untyped-def]
    admin_settings: Settings,
) -> None:
    graph, client = prepared
    client.post("/review/session", json={"password": PASSWORD})

    # The fixture creates a sibling too, and list order is by creation, so the
    # patient under test is found by id rather than assumed to be first.
    items = client.get("/review/patients").json()["items"]
    patient = next(p for p in items if p["id"] == str(graph.patient_id))

    assert patient["identifiers_masked"] is False
    assert patient["document_number_masked"] == DOCUMENT
    assert patient["phone_masked"] == PHONE
    assert "*" not in str(patient["email_masked"])
    # Every patient in the response is unmasked, not only the one checked.
    assert all(p["identifiers_masked"] is False for p in items)


async def test_an_unmasked_read_is_recorded_as_a_disclosure(
    prepared,  # type: ignore[no-untyped-def]
    session: AsyncSession,
    admin_settings: Settings,
) -> None:
    """ADR-13's requirement: its own audit action, naming the fields."""
    _graph, client = prepared
    client.post("/review/session", json={"password": PASSWORD})
    before = await _disclosures(session)

    client.get("/review/patients")

    assert await _disclosures(session) > before
    entry = (
        await session.scalars(
            select(AccessLogEntry)
            .where(AccessLogEntry.action == "disclose")
            .order_by(AccessLogEntry.id.desc())
            .limit(1)
        )
    ).one()
    assert entry.resource == "patients"
    assert entry.patient_id is not None
    assert set(entry.fields_disclosed or []) == set(UNMASKED_FIELDS)


async def test_the_patient_detail_screen_also_unmasks_and_records(
    prepared,  # type: ignore[no-untyped-def]
    session: AsyncSession,
    admin_settings: Settings,
) -> None:
    """Two routes return identifiers; both have to behave the same way."""
    graph, client = prepared
    client.post("/review/session", json={"password": PASSWORD})
    before = await _disclosures(session)

    body = client.get(f"/review/patients/{graph.patient_id}").json()

    assert body["identifiers_masked"] is False
    assert body["document_number_masked"] == DOCUMENT
    assert await _disclosures(session) > before


# --------------------------------------------------- the ways in that must fail


def test_a_forged_cookie_does_not_unmask(settings: Settings) -> None:
    """Signing is what makes the role a credential rather than a claim."""
    assert read_token("admin.9999999999.deadbeef", settings) is Role.STAFF
    assert read_token("admin", settings) is Role.STAFF
    assert read_token("admin.9999999999", settings) is Role.STAFF
    assert read_token("", settings) is Role.STAFF
    assert read_token(None, settings) is Role.STAFF


def test_an_expired_session_does_not_unmask(settings: Settings) -> None:
    long_ago = time.time() - 60 * 60 * 24
    token = issue_token(Role.ADMIN, settings, now=long_ago)

    assert read_token(token, settings) is Role.STAFF


def test_editing_the_role_in_a_valid_token_breaks_its_signature(
    settings: Settings,
) -> None:
    """The attack the signature exists to stop: take a staff token, say admin."""
    staff_token = issue_token(Role.STAFF, settings)
    _role, expiry, signature = staff_token.split(".")

    forged = f"{Role.ADMIN.value}.{expiry}.{signature}"

    assert read_token(forged, settings) is Role.STAFF


def test_a_signed_token_naming_an_unknown_role_does_not_unmask(
    settings: Settings,
) -> None:
    """Correctly signed, but a role this version cannot reason about."""
    from src.api.review.auth import _sign

    payload = f"superuser.{int(time.time()) + 3600}"
    token = f"{payload}.{_sign(payload, settings)}"

    assert read_token(token, settings) is Role.STAFF


async def test_a_staff_session_cannot_unmask_by_sending_a_cookie(
    prepared,  # type: ignore[no-untyped-def]
    settings: Settings,
) -> None:
    """End to end: a staff-signed cookie on a real request still sees masking."""
    _graph, client = prepared
    client.cookies.set(SESSION_COOKIE, issue_token(Role.STAFF, settings))

    patient = client.get("/review/patients").json()["items"][0]

    assert patient["identifiers_masked"] is True
    assert patient["document_number_masked"].startswith("*")
