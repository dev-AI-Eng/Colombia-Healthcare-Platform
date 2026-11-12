"""A doctor's phone, second phone and email: stored safely, shown, not blanked."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.tenancy import ClinicScope, apply_clinic_scope
from src.onboarding.canonical import Entity
from src.onboarding.repository import _apply_doctors
from src.onboarding.service import RowResult
from src.registry.models import Doctor
from tests.integration.factories import create_graph, scope_client

PHONE = "+573105550101"
SECOND = "+573105550102"
EMAIL = "dra.prueba@clinica.invalid"


@pytest.fixture
async def with_contact(session: AsyncSession, client: TestClient):  # type: ignore[no-untyped-def]
    graph = await create_graph(session)
    doctor = await session.get(Doctor, graph.doctor_id)
    assert doctor is not None
    doctor.phone, doctor.secondary_phone, doctor.email = PHONE, SECOND, EMAIL
    await session.commit()
    await apply_clinic_scope(session, ClinicScope(clinic_id=graph.clinic_id))
    yield scope_client(client, graph), graph


def test_the_doctors_list_shows_how_to_reach_them(with_contact) -> None:  # type: ignore[no-untyped-def]
    client, _ = with_contact
    row = client.get("/review/doctors").json()["items"][0]
    assert row["phone"] == PHONE
    assert row["secondary_phone"] == SECOND
    assert row["email"] == EMAIL


async def test_the_contact_is_ciphertext_in_the_database(
    with_contact,  # type: ignore[no-untyped-def]
    session: AsyncSession,
) -> None:
    """Direct identifiers are encrypted at rest, a doctor's included. Read the
    column the way the database holds it rather than through the model, which
    would decrypt it and prove nothing."""
    _, graph = with_contact
    await apply_clinic_scope(session, ClinicScope(clinic_id=graph.clinic_id))
    stored = (
        await session.execute(
            text("SELECT phone, secondary_phone, email FROM app.doctors WHERE id = :id"),
            {"id": graph.doctor_id},
        )
    ).one()
    for column in stored:
        raw = bytes(column)
        assert raw, "the column is empty"
        for plaintext in (PHONE, SECOND, EMAIL):
            assert plaintext.encode() not in raw, "a contact detail is stored in the clear"


def test_a_doctor_with_no_contact_details_returns_nulls(
    scoped_doctor,  # type: ignore[no-untyped-def]
) -> None:
    row = scoped_doctor.get("/review/doctors").json()["items"][0]
    assert row["phone"] is None
    assert row["secondary_phone"] is None
    assert row["email"] is None


@pytest.fixture
async def scoped_doctor(session: AsyncSession, client: TestClient):  # type: ignore[no-untyped-def]
    graph = await create_graph(session)
    await session.commit()
    await apply_clinic_scope(session, ClinicScope(clinic_id=graph.clinic_id))
    yield scope_client(client, graph)


async def test_importing_a_sheet_without_a_phone_column_keeps_the_number_we_hold(
    with_contact,  # type: ignore[no-untyped-def]
    session: AsyncSession,
) -> None:
    """A re-import is an update, and an update that omits a column must not blank
    it. A clinic re-uploading last month's doctors sheet, which never had a phone
    column, would otherwise erase every number staff had typed since."""
    _, graph = with_contact
    doctor = await session.get(Doctor, graph.doctor_id)
    assert doctor is not None
    doctor.external_ref = "M01"
    await session.flush()

    result = await _apply_doctors(
        session,
        clinic_id=graph.clinic_id,
        rows=[
            RowResult(
                row_number=2,
                entity=Entity.DOCTOR,
                raw={},
                values={"external_ref": "M01", "full_name": "Sara Betancur de la Cruz"},
            )
        ],
    )
    assert result.updated == 1

    await session.refresh(doctor)
    assert doctor.phone == PHONE
    assert doctor.secondary_phone == SECOND
    assert doctor.email == EMAIL


async def test_importing_a_sheet_with_a_phone_column_updates_it(
    with_contact,  # type: ignore[no-untyped-def]
    session: AsyncSession,
) -> None:
    _, graph = with_contact
    doctor = await session.get(Doctor, graph.doctor_id)
    assert doctor is not None
    doctor.external_ref = "M01"
    await session.flush()

    await _apply_doctors(
        session,
        clinic_id=graph.clinic_id,
        rows=[
            RowResult(
                row_number=2,
                entity=Entity.DOCTOR,
                raw={},
                values={
                    "external_ref": "M01",
                    "full_name": "Sara Betancur de la Cruz",
                    "phone": "+573105550199",
                },
            )
        ],
    )
    await session.refresh(doctor)
    assert doctor.phone == "+573105550199"
    assert doctor.email == EMAIL  # untouched: the file did not supply it
