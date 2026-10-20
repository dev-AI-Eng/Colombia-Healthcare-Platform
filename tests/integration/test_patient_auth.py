"""A patient may change their own appointment, and nobody else's.

The client asked why anyone could cancel or reschedule any appointment. The
answer was that nothing stopped them. These tests are the proof that something
does now, and they are written as the attack rather than as the feature: the
interesting cases are the ones where a caller presents a real, valid phone
number and asks to cancel an appointment that is not theirs.

What is deliberately *not* asserted here is staff authentication. That is M5
(`docs/ARCHITECTURE.md` §11.5), and a request with no patient header still
reaches the engine exactly as it did before -- which is why
`test_the_staff_path_is_unchanged_and_still_unauthenticated` exists and says
so out loud instead of leaving it to be discovered.
"""

from __future__ import annotations

import datetime as dt
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.review.patient_auth import PATIENT_PHONE_HEADER
from src.core.crypto import blind_index
from src.core.tenancy import ClinicScope, apply_clinic_scope
from src.identity.models import BindingRelationship, PhoneBinding
from src.registry.models import DocumentType, Patient
from src.scheduling.models import Appointment, AppointmentStatus
from tests.integration.factories import (
    PHONE,
    Graph,
    appointment,
    at,
    create_graph,
    scope_client,
)


@pytest.fixture
async def scoped(session: AsyncSession, client: TestClient):  # type: ignore[no-untyped-def]
    """A client pinned to one clinic, as every other API suite uses."""
    graph = await create_graph(session, hours=True)
    await session.commit()
    await apply_clinic_scope(session, ClinicScope(clinic_id=graph.clinic_id))
    yield scope_client(client, graph), graph


#: A second patient in the same clinic, on their own number.
OTHER_PHONE = "+573009998877"
OTHER_DOCUMENT = "7766554433"


async def _second_patient(session: AsyncSession, graph: Graph) -> uuid.UUID:
    """Another patient of the same clinic, bound to their own phone."""
    other = Patient(
        clinic_id=graph.clinic_id,
        document_type=DocumentType.CC,
        document_number=OTHER_DOCUMENT,
        document_number_bidx=blind_index(OTHER_DOCUMENT),
        given_names="Beatriz",
        family_names="Núñez Peña",
        phone_e164=OTHER_PHONE,
        phone_e164_bidx=blind_index(OTHER_PHONE),
    )
    session.add(other)
    await session.flush()
    session.add(
        PhoneBinding(
            clinic_id=graph.clinic_id,
            phone_e164_bidx=blind_index(OTHER_PHONE),
            patient_id=other.id,
            relationship_kind=BindingRelationship.SELF,
        )
    )
    await session.flush()
    return other.id


async def _bind(session: AsyncSession, graph: Graph, phone: str = PHONE) -> None:
    """Bind the factory's patient to `phone`, so they can prove who they are."""
    session.add(
        PhoneBinding(
            clinic_id=graph.clinic_id,
            phone_e164_bidx=blind_index(phone),
            patient_id=graph.patient_id,
            relationship_kind=BindingRelationship.SELF,
        )
    )
    await session.flush()


# ============================================================ the attack


async def test_a_patient_cannot_cancel_another_patients_appointment(
    scoped,  # type: ignore[no-untyped-def]
    session: AsyncSession,
) -> None:
    """The defect the client asked about, as the attack that exploited it.

    Beatriz presents her own real phone number -- she is a genuine patient of
    this clinic, so every check short of ownership passes -- and asks to cancel
    Ana's appointment. Before this guard existed the answer was 200 and Ana's
    appointment was gone.
    """
    client, graph = scoped
    await _bind(session, graph)
    await _second_patient(session, graph)
    ana = appointment(graph, at(14, 9, 0))
    session.add(ana)
    await session.flush()
    await session.commit()

    response = client.post(
        f"/review/appointments/{ana.id}/cancel",
        json={"reason": "no puedo asistir"},
        headers={PATIENT_PHONE_HEADER: OTHER_PHONE},
    )

    assert response.status_code == 403, response.text
    assert "no está a su nombre" in response.json()["detail"]

    # And the appointment is untouched: a refused request must change nothing.
    #
    # The scope has to be rebound first. `scoped` commits, which ends the
    # transaction that held `app.clinic_id`, so a read here sees no rows at all
    # -- row-level security working exactly as intended, and an assertion that
    # would otherwise pass for the wrong reason whatever the route did.
    await session.rollback()
    await apply_clinic_scope(session, ClinicScope(clinic_id=graph.clinic_id))
    still = await session.scalar(select(Appointment.status).where(Appointment.id == ana.id))
    assert still is AppointmentStatus.SCHEDULED


async def test_a_patient_cannot_reschedule_another_patients_appointment(
    scoped,  # type: ignore[no-untyped-def]
    session: AsyncSession,
) -> None:
    """Moving somebody else's appointment is the same violation as cancelling
    it, and worse in one way: the patient is never told it moved."""
    client, graph = scoped
    await _bind(session, graph)
    await _second_patient(session, graph)
    ana = appointment(graph, at(14, 11, 0))
    session.add(ana)
    await session.flush()
    await session.commit()

    moved_to = at(14, 16, 0)
    response = client.post(
        f"/review/appointments/{ana.id}/reschedule",
        json={"start": moved_to.isoformat(), "duration_minutes": 20},
        headers={PATIENT_PHONE_HEADER: OTHER_PHONE},
    )

    assert response.status_code == 403, response.text
    await session.rollback()
    await apply_clinic_scope(session, ClinicScope(clinic_id=graph.clinic_id))
    unchanged = await session.scalar(select(Appointment.status).where(Appointment.id == ana.id))
    assert unchanged is AppointmentStatus.SCHEDULED


async def test_a_patient_cannot_confirm_another_patients_hold(
    scoped,  # type: ignore[no-untyped-def]
    session: AsyncSession,
) -> None:
    """Confirming a stranger's hold commits them to an appointment they never
    asked for, and takes the slot."""
    client, graph = scoped
    await _bind(session, graph)
    await _second_patient(session, graph)
    held = appointment(
        graph,
        at(14, 13, 0),
        status=AppointmentStatus.HOLD,
        expires_at=dt.datetime.now(tz=dt.UTC) + dt.timedelta(minutes=10),
    )
    session.add(held)
    await session.flush()
    await session.commit()

    response = client.post(
        f"/review/appointments/{held.id}/confirm",
        headers={PATIENT_PHONE_HEADER: OTHER_PHONE},
    )
    assert response.status_code == 403, response.text


# ============================================================ the legitimate path


async def test_a_patient_can_cancel_their_own_appointment(
    scoped,  # type: ignore[no-untyped-def]
    session: AsyncSession,
) -> None:
    """The feature, not just the refusal. A guard that refused everything would
    pass every test above and be useless."""
    client, graph = scoped
    await _bind(session, graph)
    mine = appointment(graph, at(15, 9, 0))
    session.add(mine)
    await session.flush()
    await session.commit()

    response = client.post(
        f"/review/appointments/{mine.id}/cancel",
        json={"reason": "no puedo asistir"},
        headers={PATIENT_PHONE_HEADER: PHONE},
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == AppointmentStatus.CANCELLED


@pytest.mark.parametrize(
    "written",
    [
        PHONE,  # +573001112233
        "3001112233",  # national, as a patient would type it
        "300 111 2233",  # with spaces
        "(300) 111-2233",  # as a form might format it
        "+57 300 111 2233",  # with the country code spaced
    ],
)
async def test_the_number_resolves_however_the_patient_writes_it(
    scoped,  # type: ignore[no-untyped-def]
    session: AsyncSession,
    written: str,
) -> None:
    """Canonicalised to E.164 before the blind index is taken (§7).

    Without that, the index of "3001112233" and of "+573001112233" differ, the
    lookup finds nothing, and a patient cancelling their own appointment is
    told the number is not registered -- a correct request refused because of
    its formatting.
    """
    client, graph = scoped
    await _bind(session, graph)
    mine = appointment(graph, at(15, 11, 0))
    session.add(mine)
    await session.flush()
    await session.commit()

    response = client.post(
        f"/review/appointments/{mine.id}/cancel",
        json={"reason": "no puedo"},
        headers={PATIENT_PHONE_HEADER: written},
    )
    assert response.status_code == 200, f"{written!r} was not recognised: {response.text}"


# ============================================================ the edges


async def test_a_number_the_clinic_does_not_hold_is_refused(
    scoped,  # type: ignore[no-untyped-def]
    session: AsyncSession,
) -> None:
    """A valid Colombian number bound to nobody here. 403, and the same answer
    for every such number, so it reveals nothing about who is a patient."""
    client, graph = scoped
    await _bind(session, graph)
    mine = appointment(graph, at(15, 13, 0))
    session.add(mine)
    await session.flush()
    await session.commit()

    response = client.post(
        f"/review/appointments/{mine.id}/cancel",
        json={"reason": "x"},
        headers={PATIENT_PHONE_HEADER: "+573159998877"},
    )
    assert response.status_code == 403, response.text
    assert "no está registrado" in response.json()["detail"]


async def test_a_shared_handset_is_refused_rather_than_guessed(
    scoped,  # type: ignore[no-untyped-def]
    session: AsyncSession,
) -> None:
    """ADR-18, applied to the HTTP path as it is to the conversation path.

    A household number bound to two patients cannot say which of them is
    calling. Guessing would cancel the wrong family member's appointment, so
    the request is refused and the caller is told to phone the clinic -- the
    same answer M3's `identify` gives a shared handset.
    """
    client, graph = scoped
    await _bind(session, graph)
    sibling = await _second_patient(session, graph)
    # The sibling is also reachable on the first patient's number: a shared
    # handset, which is the ordinary case in a Colombian household.
    session.add(
        PhoneBinding(
            clinic_id=graph.clinic_id,
            phone_e164_bidx=blind_index(PHONE),
            patient_id=sibling,
            relationship_kind=BindingRelationship.GUARDIAN,
        )
    )
    mine = appointment(graph, at(15, 15, 0))
    session.add(mine)
    await session.flush()
    await session.commit()

    response = client.post(
        f"/review/appointments/{mine.id}/cancel",
        json={"reason": "x"},
        headers={PATIENT_PHONE_HEADER: PHONE},
    )
    assert response.status_code == 409, response.text
    assert "varias personas" in response.json()["detail"]


async def test_a_revoked_binding_no_longer_proves_anything(
    scoped,  # type: ignore[no-untyped-def]
    session: AsyncSession,
) -> None:
    """A number the patient no longer uses must stop working.

    Revocation is how a reassigned number is handled -- Colombian mobile
    numbers are recycled -- and a binding that kept working after revocation
    would hand the next owner of that number somebody else's appointments.
    """
    client, graph = scoped
    session.add(
        PhoneBinding(
            clinic_id=graph.clinic_id,
            phone_e164_bidx=blind_index(PHONE),
            patient_id=graph.patient_id,
            relationship_kind=BindingRelationship.SELF,
            revoked_at=dt.datetime.now(tz=dt.UTC) - dt.timedelta(days=1),
        )
    )
    mine = appointment(graph, at(16, 9, 0))
    session.add(mine)
    await session.flush()
    await session.commit()

    response = client.post(
        f"/review/appointments/{mine.id}/cancel",
        json={"reason": "x"},
        headers={PATIENT_PHONE_HEADER: PHONE},
    )
    assert response.status_code == 403, response.text


async def test_a_malformed_number_is_a_422_not_a_crash(
    scoped,  # type: ignore[no-untyped-def]
    session: AsyncSession,
) -> None:
    """Rubbish in the header is a request error, not a 500. It is also not a
    403: the caller has not failed to prove who they are, they have sent
    something that is not a phone number, and those need different fixes."""
    client, graph = scoped
    await _bind(session, graph)
    mine = appointment(graph, at(16, 11, 0))
    session.add(mine)
    await session.flush()
    await session.commit()

    for rubbish in ("", "not-a-number", "12", "'; DROP TABLE patients; --"):
        response = client.post(
            f"/review/appointments/{mine.id}/cancel",
            json={"reason": "x"},
            headers={PATIENT_PHONE_HEADER: rubbish},
        )
        assert response.status_code == 422, f"{rubbish!r} gave {response.status_code}"


async def test_a_patient_cannot_reach_another_clinics_appointment(
    scoped,  # type: ignore[no-untyped-def]
    session: AsyncSession,
) -> None:
    """The binding is looked up inside the request's clinic scope, so a number
    bound in clinic A proves nothing in clinic B -- and the appointment is not
    visible there either. Belt and braces, deliberately: the clinic scope is
    the barrier, and this asserts the new check does not accidentally bypass
    it by resolving a binding globally."""
    client, graph = scoped
    await _bind(session, graph)
    neighbour = await create_graph(session, name="Clínica Vecina", hours=True)
    theirs = appointment(neighbour, at(16, 13, 0))
    session.add(theirs)
    await session.flush()
    await session.commit()

    # `scoped` pins the client to `graph`'s clinic, so this asks clinic A to
    # cancel a row that lives in clinic B.
    response = client.post(
        f"/review/appointments/{theirs.id}/cancel",
        json={"reason": "x"},
        headers={PATIENT_PHONE_HEADER: PHONE},
    )
    assert response.status_code == 404, response.text


# ============================================================ what is NOT fixed


async def test_the_staff_path_is_unchanged_and_still_unauthenticated(
    scoped,  # type: ignore[no-untyped-def]
    session: AsyncSession,
) -> None:
    """Stated as a test so it cannot be mistaken for solved.

    Omitting the header is the staff path, and it still cancels any appointment
    in the clinic with no credential at all. That is M5's scope
    (`docs/ARCHITECTURE.md` §11.5: session auth, a second factor for any role
    that reads patient data, roles enforced in dependencies), and until then
    the controls on the staff path are the ones that were always there:
    synthetic-data mode only, a loopback-only Host, and the clinic scope.

    If this test ever fails because a credential became required, that is the
    M5 work landing -- update it rather than deleting it.
    """
    client, graph = scoped
    await _bind(session, graph)
    anyones = appointment(graph, at(16, 15, 0))
    session.add(anyones)
    await session.flush()
    await session.commit()

    response = client.post(
        f"/review/appointments/{anyones.id}/cancel",
        json={"reason": "cancelada en recepción"},
    )
    assert response.status_code == 200, response.text
