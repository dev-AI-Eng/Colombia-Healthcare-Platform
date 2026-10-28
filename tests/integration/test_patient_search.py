"""Finding a patient by the number staff actually type.

The patients screen has no search box, and the API's lookup was exact-match on a
blind index: it worked for "+573001112233" and returned nothing for
"300 111 2233". A blind index of the two spellings is two unrelated values, so
the failure is silent -- an empty list, which a receptionist reads as "this
patient is not in the system". That is the one answer a search box must never
give for somebody who is in the clinic, so these tests are the attempts a
receptionist would make, not the one spelling the index happens to hold.

What is deliberately absent is name search. Names are encrypted, so there is no
index to ask; see the response to the client for the options.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.crypto import blind_index
from src.core.tenancy import ClinicScope, apply_clinic_scope
from src.registry.models import DocumentType, Patient
from tests.integration.factories import (
    DOCUMENT,
    PHONE,
    Graph,
    create_graph,
    scope_client,
)

PASSPORT = "AB1234567"


@pytest.fixture
async def scoped(session: AsyncSession, client: TestClient):  # type: ignore[no-untyped-def]
    graph = await create_graph(session)
    await session.commit()
    await apply_clinic_scope(session, ClinicScope(clinic_id=graph.clinic_id))
    yield scope_client(client, graph), graph


def _ids(response) -> set[str]:  # type: ignore[no-untyped-def]
    assert response.status_code == 200, response.text
    return {item["id"] for item in response.json()["items"]}


# ------------------------------------------------------------------ the phone


@pytest.mark.parametrize(
    "typed",
    [
        PHONE,  # +573001112233, the one spelling the index holds
        "3001112233",  # how people say it
        "300 111 2233",
        "(300) 111-2233",
        "+57 300 111 2233",
        "57 300 111 2233",
    ],
)
def test_the_phone_is_found_however_staff_type_it(scoped, typed: str) -> None:  # type: ignore[no-untyped-def]
    client, graph = scoped
    found = _ids(client.get("/review/patients", params={"phone": typed}))
    assert str(graph.patient_id) in found, f"{typed!r} found nobody"


def test_a_number_no_patient_has_is_an_empty_list_not_an_error(scoped) -> None:  # type: ignore[no-untyped-def]
    client, _ = scoped
    response = client.get("/review/patients", params={"phone": "+573159998877"})
    assert response.status_code == 200
    assert response.json()["items"] == []


@pytest.mark.parametrize("rubbish", ["abc", "12", "'; DROP TABLE patients; --"])
def test_something_that_is_not_a_phone_number_is_a_request_error(scoped, rubbish: str) -> None:  # type: ignore[no-untyped-def]
    """A 422, not an empty list: "no such patient" and "that is not a phone
    number" need different responses from the person typing."""
    client, _ = scoped
    response = client.get("/review/patients", params={"phone": rubbish})
    assert response.status_code == 422, response.text


def test_an_empty_search_box_lists_everyone(scoped) -> None:  # type: ignore[no-untyped-def]
    """A cleared box must go back to the full list, not to a 422."""
    client, graph = scoped
    found = _ids(client.get("/review/patients", params={"phone": "", "document_number": "  "}))
    assert str(graph.patient_id) in found


# ----------------------------------------------------------- the document number


@pytest.mark.parametrize(
    "typed",
    [
        DOCUMENT,  # 1020304050
        "1.020.304.050",  # the way Colombians write a cédula
        " 1020304050 ",  # Excel and copy-paste leave spaces
        "1020304050.0",  # a float that happens to be whole
    ],
)
def test_the_document_is_found_however_staff_type_it(scoped, typed: str) -> None:  # type: ignore[no-untyped-def]
    client, graph = scoped
    found = _ids(client.get("/review/patients", params={"document_number": typed}))
    assert str(graph.patient_id) in found, f"{typed!r} found nobody"


async def test_a_passport_is_found_in_whatever_case_it_is_typed(
    scoped,  # type: ignore[no-untyped-def]
    session: AsyncSession,
) -> None:
    """Passports carry letters. The importer stores them upper-case, so a
    receptionist typing the lower-case form must still reach the record."""
    client, graph = scoped
    traveller = Patient(
        clinic_id=graph.clinic_id,
        document_type=DocumentType.PA,
        document_number=PASSPORT,
        document_number_bidx=blind_index(PASSPORT),
        given_names="Lucas",
        family_names="Martin",
    )
    session.add(traveller)
    await session.commit()

    found = _ids(client.get("/review/patients", params={"document_number": "ab1234567"}))
    assert str(traveller.id) in found


def test_a_partial_number_matches_nobody_rather_than_everybody(scoped) -> None:  # type: ignore[no-untyped-def]
    """The match is exact by design -- the index cannot do prefixes. The screen
    says so; the API's part is to return nobody, not a guess."""
    client, _ = scoped
    assert client.get("/review/patients", params={"document_number": "10203"}).json()["items"] == []


# --------------------------------------------------------------- what stays hidden


async def test_a_search_never_reaches_another_clinics_patient(
    scoped,  # type: ignore[no-untyped-def]
    session: AsyncSession,
) -> None:
    """The same cédula in two clinics is two people. Searching one clinic for it
    must not surface the other's -- the clinic scope is the barrier, and the new
    canonicalisation must not have loosened it."""
    client, graph = scoped
    neighbour: Graph = await create_graph(session, name="Clínica Vecina")
    await session.commit()

    found = _ids(client.get("/review/patients", params={"document_number": DOCUMENT}))
    assert str(neighbour.patient_id) not in found
    assert str(graph.patient_id) in found


async def test_a_soft_deleted_patient_is_not_found(
    scoped,  # type: ignore[no-untyped-def]
    session: AsyncSession,
) -> None:
    import datetime as dt

    client, graph = scoped
    await apply_clinic_scope(session, ClinicScope(clinic_id=graph.clinic_id))
    patient = await session.get(Patient, graph.patient_id)
    assert patient is not None
    patient.deleted_at = dt.datetime.now(tz=dt.UTC)
    await session.commit()

    found = _ids(client.get("/review/patients", params={"phone": PHONE}))
    assert str(graph.patient_id) not in found


# ----------------------------------------------- EPS and the secondary contact


async def test_the_eps_and_secondary_contact_the_import_holds_are_shown(
    scoped,  # type: ignore[no-untyped-def]
    session: AsyncSession,
) -> None:
    """The importer has carried `eps` and the secondary contact since migration
    0005, and `PatientOut` never returned them, so the screen could not show
    what the import held. The client asked for both fields as if they were
    missing; the data was there and the response dropped it."""
    client, graph = scoped
    await apply_clinic_scope(session, ClinicScope(clinic_id=graph.clinic_id))
    patient = await session.get(Patient, graph.patient_id)
    assert patient is not None
    patient.eps = "Sanitas"
    patient.secondary_contact_name = "Camila Gómez"
    patient.secondary_contact_phone = "+573105550199"
    await session.commit()

    row = client.get("/review/patients", params={"document_number": DOCUMENT}).json()["items"][0]
    assert row["eps"] == "Sanitas"
    assert row["secondary_contact_name"] == "Camila Gómez"


async def test_the_secondary_phone_is_masked_like_every_other_identifier(
    scoped,  # type: ignore[no-untyped-def]
    session: AsyncSession,
) -> None:
    """A second phone is a second route to the patient and is no less an
    identifier for being a relative's. A staff session sees it masked, exactly
    as it sees the patient's own."""
    client, graph = scoped
    await apply_clinic_scope(session, ClinicScope(clinic_id=graph.clinic_id))
    patient = await session.get(Patient, graph.patient_id)
    assert patient is not None
    patient.secondary_contact_phone = "+573105550199"
    await session.commit()

    row = client.get("/review/patients", params={"document_number": DOCUMENT}).json()["items"][0]
    masked = row["secondary_contact_phone_masked"]
    assert masked is not None
    assert masked != "+573105550199"
    assert masked.endswith("0199")
    assert row["identifiers_masked"] is True


def test_a_patient_with_no_eps_or_secondary_contact_returns_nulls(scoped) -> None:  # type: ignore[no-untyped-def]
    client, _ = scoped
    row = client.get("/review/patients", params={"document_number": DOCUMENT}).json()["items"][0]
    assert row["eps"] is None
    assert row["secondary_contact_name"] is None
    assert row["secondary_contact_phone_masked"] is None
