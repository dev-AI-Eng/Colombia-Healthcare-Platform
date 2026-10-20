"""Minimal record builders for integration tests."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

from fastapi.testclient import TestClient
from sqlalchemy.dialects.postgresql import Range
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.crypto import blind_index
from src.core.tenancy import ClinicScope, apply_clinic_scope
from src.core.timezones import BOGOTA
from src.identity.models import (
    BindingRelationship,
    Consent,
    ConsentChannel,
    ConsentPurpose,
    EvidenceKind,
    PhoneBinding,
)
from src.registry.models import Clinic, Doctor, DocumentType, Location, Patient
from src.scheduling.models import Appointment, AppointmentStatus, AppointmentType

PHONE = "+573001112233"
DOCUMENT = "1020304050"


@dataclass(frozen=True, slots=True)
class Graph:
    clinic_id: uuid.UUID
    doctor_id: uuid.UUID
    location_id: uuid.UUID
    appointment_type_id: uuid.UUID
    patient_id: uuid.UUID


async def create_graph(session: AsyncSession, *, name: str = "Clínica Prueba") -> Graph:
    """A clinic with one location, doctor, appointment type and patient. Flushed, not committed.

    The clinic scope is left bound to the session's transaction, because row-level
    security rejects every child insert without it.
    """
    clinic = Clinic(name=name)
    session.add(clinic)
    await session.flush()
    await apply_clinic_scope(session, ClinicScope(clinic_id=clinic.id))
    location = Location(clinic_id=clinic.id, name="Sede", address="Calle 1 # 2-3")
    doctor = Doctor(clinic_id=clinic.id, full_name="Sara Betancur de la Cruz", specialty="Medicina General")
    appointment_type = AppointmentType(clinic_id=clinic.id, name="Consulta", duration_minutes=20)
    patient = Patient(
        clinic_id=clinic.id,
        document_type=DocumentType.CC,
        document_number=DOCUMENT,
        document_number_bidx=blind_index(DOCUMENT),
        given_names="Ana",
        family_names="Gómez Ruiz",
        phone_e164=PHONE,
        phone_e164_bidx=blind_index(PHONE),
        email="ana.gomez@example.com",
    )
    session.add_all([location, doctor, appointment_type, patient])
    await session.flush()
    return Graph(
        clinic_id=clinic.id,
        doctor_id=doctor.id,
        location_id=location.id,
        appointment_type_id=appointment_type.id,
        patient_id=patient.id,
    )


def appointment(
    graph: Graph,
    start: datetime,
    *,
    minutes: int = 20,
    status: AppointmentStatus = AppointmentStatus.SCHEDULED,
    expires_at: datetime | None = None,
    patient_id: uuid.UUID | None = None,
) -> Appointment:
    return Appointment(
        clinic_id=graph.clinic_id,
        patient_id=patient_id or graph.patient_id,
        doctor_id=graph.doctor_id,
        location_id=graph.location_id,
        appointment_type_id=graph.appointment_type_id,
        during=Range(start, start + timedelta(minutes=minutes), bounds="[)"),
        status=status,
        expires_at=expires_at,
    )


def scope_client(client: TestClient, graph: Graph) -> TestClient:
    """Send `clinic_id` on every subsequent request. Scoped routes require it."""
    client.params = {"clinic_id": str(graph.clinic_id)}
    return client


def at(day: int, hour: int, minute: int = 0) -> datetime:
    """A Bogotá-local time in October 2026."""
    return datetime(2026, 10, day, hour, minute, tzinfo=BOGOTA)


async def add_patient_records(session: AsyncSession, graph: Graph) -> uuid.UUID:
    """Add a consent, a self binding, one appointment, and a second patient sharing
    the first patient's phone. Returns the appointment id. Commits."""
    booking = appointment(graph, at(5, 9))
    session.add(booking)
    session.add(
        Consent(
            clinic_id=graph.clinic_id,
            patient_id=graph.patient_id,
            purpose=ConsentPurpose.APPOINTMENT_MESSAGING,
            channel=ConsentChannel.WHATSAPP,
            granted_at=at(1, 8),
            evidence_kind=EvidenceKind.IMPORTED_DECLARATION,
            policy_version="test-v1",
        )
    )
    sibling = Patient(
        clinic_id=graph.clinic_id,
        document_type=DocumentType.TI,
        document_number="1122334455",
        document_number_bidx=blind_index("1122334455"),
        given_names="Tomás",
        family_names="Gómez Ruiz",
    )
    session.add(sibling)
    await session.flush()
    session.add_all(
        [
            PhoneBinding(
                clinic_id=graph.clinic_id,
                phone_e164_bidx=blind_index(PHONE),
                patient_id=graph.patient_id,
                relationship_kind=BindingRelationship.SELF,
            ),
            PhoneBinding(
                clinic_id=graph.clinic_id,
                phone_e164_bidx=blind_index(PHONE),
                patient_id=sibling.id,
                relationship_kind=BindingRelationship.GUARDIAN,
            ),
        ]
    )
    await session.commit()
    # The commit ended the transaction that held the scope; rebind it so the
    # caller can keep querying this clinic through the same session.
    await apply_clinic_scope(session, ClinicScope(clinic_id=graph.clinic_id))
    return booking.id
