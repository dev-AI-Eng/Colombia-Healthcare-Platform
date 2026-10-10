"""Response models for the review API.

Appointment `start` and `end` are converted to Bogotá time (-05:00). Other
timestamps are serialized with whatever UTC offset the database session
returns them in; all of them are absolute instants.

Document numbers, phone numbers and emails are masked (see masking.py).
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, time
from typing import Any

from pydantic import BaseModel, ConfigDict

from src.api.review.masking import mask_email, mask_tail
from src.audit.models import AccessLogEntry
from src.audit.service import ChainVerification
from src.core.pagination import PageResult
from src.core.timezones import BOGOTA
from src.identity.models import Consent, PhoneBinding
from src.identity.repository import SharedHandset
from src.registry.models import Clinic, Doctor, Location, Patient
from src.scheduling.models import AppointmentType, AvailabilityException, AvailabilityRule
from src.scheduling.repository import AppointmentView, EscalationView

WEEKDAY_NAMES_ES = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")


class _Out(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class Page[T](BaseModel):
    items: list[T]
    total: int
    limit: int
    offset: int


def to_page[S, T](result: PageResult[S], items: list[T]) -> Page[T]:
    return Page[T](items=items, total=result.total, limit=result.limit, offset=result.offset)


class Summary(BaseModel):
    clinic_id: uuid.UUID
    app_env: str
    real_patient_data_allowed: bool
    row_counts: dict[str, int]
    appointments_by_status: dict[str, int]


class ClinicOut(_Out):
    id: uuid.UUID
    name: str
    timezone: str
    locale: str
    created_at: datetime

    @classmethod
    def build(cls, clinic: Clinic) -> ClinicOut:
        return cls.model_validate(clinic)


class LocationOut(_Out):
    id: uuid.UUID
    clinic_id: uuid.UUID
    name: str
    address: str

    @classmethod
    def build(cls, location: Location) -> LocationOut:
        return cls.model_validate(location)


class AppointmentTypeOut(_Out):
    id: uuid.UUID
    clinic_id: uuid.UUID
    name: str
    duration_minutes: int
    buffer_minutes: int
    sensitivity: str

    @classmethod
    def build(cls, appointment_type: AppointmentType) -> AppointmentTypeOut:
        return cls.model_validate(appointment_type)


class AvailabilityRuleOut(_Out):
    id: uuid.UUID
    location_id: uuid.UUID
    weekday: int
    weekday_name: str
    start_time: time
    end_time: time
    valid_from: date
    valid_until: date | None

    @classmethod
    def build(cls, rule: AvailabilityRule) -> AvailabilityRuleOut:
        return cls(
            id=rule.id,
            location_id=rule.location_id,
            weekday=rule.weekday,
            weekday_name=WEEKDAY_NAMES_ES[rule.weekday],
            start_time=rule.start_time,
            end_time=rule.end_time,
            valid_from=rule.valid_from,
            valid_until=rule.valid_until,
        )


class AvailabilityExceptionOut(_Out):
    id: uuid.UUID
    on_date: date
    kind: str
    start_time: time | None
    end_time: time | None

    @classmethod
    def build(cls, exception: AvailabilityException) -> AvailabilityExceptionOut:
        return cls.model_validate(exception)


class DoctorOut(_Out):
    id: uuid.UUID
    clinic_id: uuid.UUID
    full_name: str
    specialty: str
    active: bool

    @classmethod
    def build(cls, doctor: Doctor) -> DoctorOut:
        return cls.model_validate(doctor)


class DoctorDetail(DoctorOut):
    availability_rules: list[AvailabilityRuleOut]
    availability_exceptions: list[AvailabilityExceptionOut]
    upcoming_active_appointments: int


class PatientOut(BaseModel):
    id: uuid.UUID
    clinic_id: uuid.UUID
    document_type: str
    document_number_masked: str | None
    given_names: str
    family_names: str
    birth_date: date | None
    age: int | None
    is_minor: bool | None
    phone_masked: str | None
    email_masked: str | None
    created_at: datetime

    @classmethod
    def build(cls, patient: Patient, *, today: date) -> PatientOut:
        age = _age_on(patient.birth_date, today)
        return cls(
            id=patient.id,
            clinic_id=patient.clinic_id,
            document_type=patient.document_type,
            document_number_masked=mask_tail(patient.document_number),
            given_names=patient.given_names,
            family_names=patient.family_names,
            birth_date=patient.birth_date,
            age=age,
            is_minor=None if age is None else age < 18,
            phone_masked=mask_tail(patient.phone_e164),
            email_masked=mask_email(patient.email),
            created_at=patient.created_at,
        )


class ConsentOut(BaseModel):
    id: uuid.UUID
    patient_id: uuid.UUID
    purpose: str
    channel: str
    granted_at: datetime
    revoked_at: datetime | None
    active: bool
    evidence_kind: str
    policy_version: str

    @classmethod
    def build(cls, consent: Consent) -> ConsentOut:
        return cls(
            id=consent.id,
            patient_id=consent.patient_id,
            purpose=consent.purpose,
            channel=consent.channel,
            granted_at=consent.granted_at,
            revoked_at=consent.revoked_at,
            active=consent.revoked_at is None,
            evidence_kind=consent.evidence_kind,
            policy_version=consent.policy_version,
        )


class PhoneBindingOut(_Out):
    id: uuid.UUID
    patient_id: uuid.UUID
    relationship_kind: str
    verified_at: datetime | None
    revoked_at: datetime | None

    @classmethod
    def build(cls, binding: PhoneBinding) -> PhoneBindingOut:
        return cls.model_validate(binding)


class AppointmentOut(BaseModel):
    id: uuid.UUID
    clinic_id: uuid.UUID
    status: str
    start: datetime
    end: datetime
    duration_minutes: int
    expires_at: datetime | None
    doctor_id: uuid.UUID
    doctor_name: str
    specialty: str
    patient_id: uuid.UUID
    patient_name: str
    location_id: uuid.UUID
    location_name: str
    appointment_type: str

    @classmethod
    def build(cls, view: AppointmentView) -> AppointmentOut:
        appointment = view.appointment
        if appointment.during.lower is None or appointment.during.upper is None:
            # Unreachable for stored rows: a CHECK constraint requires both bounds.
            raise ValueError(f"Appointment {appointment.id} has an unbounded time range")
        start = appointment.during.lower.astimezone(BOGOTA)
        end = appointment.during.upper.astimezone(BOGOTA)
        return cls(
            id=appointment.id,
            clinic_id=appointment.clinic_id,
            status=appointment.status,
            start=start,
            end=end,
            duration_minutes=int((end - start).total_seconds() // 60),
            expires_at=appointment.expires_at,
            doctor_id=appointment.doctor_id,
            doctor_name=view.doctor_name,
            specialty=view.specialty,
            patient_id=appointment.patient_id,
            patient_name=f"{view.patient_given_names} {view.patient_family_names}",
            location_id=appointment.location_id,
            location_name=view.location_name,
            appointment_type=view.appointment_type_name,
        )


class PatientDetail(PatientOut):
    consents: list[ConsentOut]
    phone_bindings: list[PhoneBindingOut]
    appointments: list[AppointmentOut]


class HandsetMemberOut(BaseModel):
    patient_id: uuid.UUID
    patient_name: str
    relationship_kind: str


class SharedHandsetOut(BaseModel):
    reference: str
    patient_count: int
    members: list[HandsetMemberOut]

    @classmethod
    def build(cls, handset: SharedHandset) -> SharedHandsetOut:
        return cls(
            reference=handset.reference,
            patient_count=len(handset.members),
            members=[
                HandsetMemberOut(
                    patient_id=member.binding.patient_id,
                    patient_name=f"{member.given_names} {member.family_names}",
                    relationship_kind=member.binding.relationship_kind,
                )
                for member in handset.members
            ],
        )


class AuditEntryOut(BaseModel):
    id: int
    occurred_at: datetime
    actor_kind: str
    actor_id: str | None
    patient_id: uuid.UUID | None
    action: str
    resource: str
    resource_id: str | None
    purpose: str
    channel: str | None
    request_id: str | None
    processor: str | None
    processor_model: str | None
    fields_disclosed: list[str] | None
    zero_retention: bool | None
    row_hash_prefix: str

    @classmethod
    def build(cls, entry: AccessLogEntry) -> AuditEntryOut:
        return cls(
            id=entry.id,
            occurred_at=entry.occurred_at,
            actor_kind=entry.actor_kind,
            actor_id=entry.actor_id,
            patient_id=entry.patient_id,
            action=entry.action,
            resource=entry.resource,
            resource_id=entry.resource_id,
            purpose=entry.purpose,
            channel=entry.channel,
            request_id=entry.request_id,
            processor=entry.processor,
            processor_model=entry.processor_model,
            fields_disclosed=entry.fields_disclosed,
            zero_retention=entry.zero_retention,
            row_hash_prefix=entry.row_hash[:8].hex(),
        )


class ChainVerificationOut(BaseModel):
    rows_checked: int
    intact: bool
    first_broken_id: int | None
    truncated_after_id: int | None

    @classmethod
    def build(cls, verification: ChainVerification) -> ChainVerificationOut:
        return cls(
            rows_checked=verification.rows_checked,
            intact=verification.intact,
            first_broken_id=verification.first_broken_id,
            truncated_after_id=verification.truncated_after_id,
        )


def _age_on(birth_date: date | None, today: date) -> int | None:
    if birth_date is None:
        return None
    before_birthday = (today.month, today.day) < (birth_date.month, birth_date.day)
    return today.year - birth_date.year - before_birthday


class EscalationOut(BaseModel):
    """One thing a human has to deal with.

    `detail_es` is the Spanish sentence a receptionist reads and acts on; the
    reason code is what the queue is sorted and filtered by. The patient's name
    is shown in full, as on every other screen -- it is the document, phone and
    email that are masked, and an escalation carries none of those.
    """

    id: uuid.UUID
    clinic_id: uuid.UUID
    reason: str
    status: str
    detail_es: str
    context: dict[str, Any]
    created_at: datetime
    resolved_at: datetime | None
    patient_id: uuid.UUID | None
    #: None when the escalation names no patient, which is itself a reason to
    #: escalate: somebody messaged the clinic and nobody knows who.
    patient_name: str | None
    appointment_id: uuid.UUID | None

    @classmethod
    def build(cls, view: EscalationView) -> EscalationOut:
        row = view.escalation
        name = None
        if view.patient_given_names or view.patient_family_names:
            name = " ".join(
                part for part in (view.patient_given_names, view.patient_family_names) if part
            )
        return cls(
            id=row.id,
            clinic_id=row.clinic_id,
            reason=str(row.reason),
            status=str(row.status),
            detail_es=row.detail_es,
            context=dict(row.context or {}),
            created_at=row.created_at,
            resolved_at=row.resolved_at,
            patient_id=row.patient_id,
            patient_name=name,
            appointment_id=row.appointment_id,
        )
