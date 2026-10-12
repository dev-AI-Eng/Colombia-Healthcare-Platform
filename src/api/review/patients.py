"""Review routes for patients and appointments.

Every read is clinic-scoped and audited by the repositories.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, time, timedelta
from typing import Annotated

from fastapi import APIRouter, Query

from src.api.dependencies import ClinicScopeDep, PageDep, SessionDep, require_found
from src.api.review.auth import UNMASKED_FIELDS, RoleDep
from src.api.review.schemas import (
    AppointmentOut,
    ConsentOut,
    Page,
    PatientDetail,
    PatientOut,
    PhoneBindingOut,
    to_page,
)
from src.audit.context import AccessAction
from src.audit.service import Access, record_accesses
from src.core.timezones import BOGOTA
from src.identity import repository as identity
from src.registry import repository as registry
from src.scheduling import repository as scheduling
from src.scheduling.models import AppointmentStatus

router = APIRouter()


def _today() -> date:
    return datetime.now(BOGOTA).date()


async def _record_disclosure(session: SessionDep, patients: list[uuid.UUID]) -> None:
    """Record that full identifiers were shown to somebody.

    `disclose` rather than `read`, and naming the fields, because ADR-13's
    question is "who has seen a patient's cédula" and an ordinary read entry
    cannot answer it -- the log would show that somebody opened the patients
    screen, which every member of staff does.

    The repository has already recorded the read itself; this is the second,
    narrower entry that says what was uncovered.
    """
    if not patients:
        return
    await record_accesses(
        session,
        AccessAction.DISCLOSE,
        [
            Access(
                resource="patients",
                resource_id=str(patient_id),
                patient_id=patient_id,
                fields_disclosed=UNMASKED_FIELDS,
            )
            for patient_id in patients
        ],
    )


@router.get("/patients", response_model=Page[PatientOut])
async def list_patients(
    session: SessionDep,
    scope: ClinicScopeDep,
    role: RoleDep,
    page: PageDep,
    phone: Annotated[
        str | None, Query(description="Exact E.164 number, for example +573001112233")
    ] = None,
    document_number: Annotated[str | None, Query(description="Exact document number")] = None,
) -> Page[PatientOut]:
    """Active patients of the clinic, oldest first.

    Names and identifiers are encrypted at rest, so there is no name search.
    `phone` and `document_number` match exactly through the blind indexes.
    """
    result = await registry.list_patients(
        session,
        clinic_id=scope.clinic_id,
        page=page,
        phone_e164=phone,
        document_number=document_number,
    )
    today = _today()
    unmasked = role.sees_unmasked
    if unmasked:
        await _record_disclosure(session, [p.id for p in result.items])
    return to_page(
        result,
        [PatientOut.build(p, today=today, unmasked=unmasked) for p in result.items],
    )


@router.get("/patients/{patient_id}", response_model=PatientDetail)
async def get_patient(
    session: SessionDep,
    scope: ClinicScopeDep,
    role: RoleDep,
    patient_id: uuid.UUID,
) -> PatientDetail:
    """One patient with their consents, phone bindings and appointments."""
    patient = require_found(
        await registry.get_patient(session, clinic_id=scope.clinic_id, patient_id=patient_id),
        "Patient",
    )
    consents = await identity.list_patient_consents(
        session, clinic_id=scope.clinic_id, patient_id=patient_id
    )
    bindings = await identity.list_patient_bindings(
        session, clinic_id=scope.clinic_id, patient_id=patient_id
    )
    appointments = await scheduling.list_patient_appointments(
        session, clinic_id=scope.clinic_id, patient_id=patient_id
    )
    unmasked = role.sees_unmasked
    if unmasked:
        await _record_disclosure(session, [patient.id])
    return PatientDetail(
        **PatientOut.build(patient, today=_today(), unmasked=unmasked).model_dump(),
        consents=[ConsentOut.build(consent) for consent in consents],
        phone_bindings=[PhoneBindingOut.build(binding) for binding in bindings],
        appointments=[AppointmentOut.build(view) for view in appointments],
    )


@router.get("/appointments", response_model=Page[AppointmentOut])
async def list_appointments(
    session: SessionDep,
    scope: ClinicScopeDep,
    page: PageDep,
    doctor_id: uuid.UUID | None = None,
    patient_id: uuid.UUID | None = None,
    status_filter: Annotated[AppointmentStatus | None, Query(alias="status")] = None,
    date_from: Annotated[
        date | None, Query(description="First Bogotá calendar day to include")
    ] = None,
    date_to: Annotated[
        date | None, Query(description="Last Bogotá calendar day to include")
    ] = None,
) -> Page[AppointmentOut]:
    """Appointments ordered by start time. Date filters apply to the start time."""
    filters = scheduling.AppointmentFilter(
        clinic_id=scope.clinic_id,
        doctor_id=doctor_id,
        patient_id=patient_id,
        status=status_filter,
        starts_from=datetime.combine(date_from, time.min, tzinfo=BOGOTA) if date_from else None,
        starts_before=(
            datetime.combine(date_to + timedelta(days=1), time.min, tzinfo=BOGOTA)
            if date_to
            else None
        ),
    )
    result = await scheduling.list_appointments(session, page=page, filters=filters)
    return to_page(result, [AppointmentOut.build(view) for view in result.items])


@router.get("/appointments/{appointment_id}", response_model=AppointmentOut)
async def get_appointment(
    session: SessionDep, scope: ClinicScopeDep, appointment_id: uuid.UUID
) -> AppointmentOut:
    view = require_found(
        await scheduling.get_appointment(
            session, clinic_id=scope.clinic_id, appointment_id=appointment_id
        ),
        "Appointment",
    )
    return AppointmentOut.build(view)
