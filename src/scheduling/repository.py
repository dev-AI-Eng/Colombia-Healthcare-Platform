"""Queries for appointment types, availability and appointments.

Every function takes the clinic explicitly; row-level security enforces the same
boundary in the database. Appointment results include the patient's name, so
every function returning appointments records one audit entry per appointment
returned, and none of them returns appointments of a soft-deleted patient.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.audit.context import AccessAction
from src.audit.service import Access, record_access, record_accesses
from src.core.pagination import PageRequest, PageResult, fetch_page
from src.registry.models import Doctor, Location, Patient
from src.scheduling.models import (
    ACTIVE_STATUSES,
    Appointment,
    AppointmentStatus,
    AppointmentType,
    AvailabilityException,
    AvailabilityRule,
    Escalation,
    EscalationStatus,
)


@dataclass(frozen=True, slots=True)
class AppointmentView:
    """An appointment with the display names of the records it references."""

    appointment: Appointment
    doctor_name: str
    specialty: str
    location_name: str
    appointment_type_name: str
    patient_given_names: str
    patient_family_names: str


@dataclass(frozen=True, slots=True)
class AppointmentFilter:
    """Which appointments to return. The clinic is required; the rest narrow further."""

    clinic_id: uuid.UUID
    doctor_id: uuid.UUID | None = None
    patient_id: uuid.UUID | None = None
    status: AppointmentStatus | None = None
    starts_from: datetime | None = None  # inclusive
    starts_before: datetime | None = None  # exclusive


def _appointment_view_statement(clinic_id: uuid.UUID) -> Select[Any]:
    return (
        select(
            Appointment,
            Doctor.full_name,
            Doctor.specialty,
            Location.name,
            AppointmentType.name,
            Patient.given_names,
            Patient.family_names,
        )
        .join(Doctor, Doctor.id == Appointment.doctor_id)
        .join(Location, Location.id == Appointment.location_id)
        .join(AppointmentType, AppointmentType.id == Appointment.appointment_type_id)
        .join(Patient, Patient.id == Appointment.patient_id)
        .where(Appointment.clinic_id == clinic_id, Patient.deleted_at.is_(None))
        .order_by(func.lower(Appointment.during), Appointment.id)
    )


def _to_view(row: Any) -> AppointmentView:
    appointment, doctor_name, specialty, location_name, type_name, given, family = row
    return AppointmentView(
        appointment=appointment,
        doctor_name=doctor_name,
        specialty=specialty,
        location_name=location_name,
        appointment_type_name=type_name,
        patient_given_names=given,
        patient_family_names=family,
    )


async def _record_views(session: AsyncSession, views: list[AppointmentView]) -> None:
    await record_accesses(
        session,
        AccessAction.READ,
        [
            Access(
                resource="appointments",
                resource_id=str(view.appointment.id),
                patient_id=view.appointment.patient_id,
            )
            for view in views
        ],
    )


async def list_appointment_types(
    session: AsyncSession, *, clinic_id: uuid.UUID
) -> list[AppointmentType]:
    statement = (
        select(AppointmentType)
        .where(AppointmentType.clinic_id == clinic_id)
        .order_by(AppointmentType.name)
    )
    return list((await session.scalars(statement)).all())


async def get_availability(
    session: AsyncSession, *, clinic_id: uuid.UUID, doctor_id: uuid.UUID
) -> tuple[list[AvailabilityRule], list[AvailabilityException]]:
    rules = await session.scalars(
        select(AvailabilityRule)
        .where(AvailabilityRule.clinic_id == clinic_id, AvailabilityRule.doctor_id == doctor_id)
        .order_by(AvailabilityRule.weekday, AvailabilityRule.start_time)
    )
    exceptions = await session.scalars(
        select(AvailabilityException)
        .where(
            AvailabilityException.clinic_id == clinic_id,
            AvailabilityException.doctor_id == doctor_id,
        )
        .order_by(AvailabilityException.on_date, AvailabilityException.start_time)
    )
    return list(rules.all()), list(exceptions.all())


async def count_upcoming_active(
    session: AsyncSession, *, clinic_id: uuid.UUID, doctor_id: uuid.UUID
) -> int:
    """Active appointments or holds for the doctor that start now or later."""
    count = await session.scalar(
        select(func.count())
        .select_from(Appointment)
        .where(
            Appointment.clinic_id == clinic_id,
            Appointment.doctor_id == doctor_id,
            Appointment.status.in_(ACTIVE_STATUSES),
            func.lower(Appointment.during) >= func.now(),
        )
    )
    return int(count or 0)


async def count_by_status(session: AsyncSession, *, clinic_id: uuid.UUID) -> dict[str, int]:
    rows = await session.execute(
        select(Appointment.status, func.count())
        .where(Appointment.clinic_id == clinic_id)
        .group_by(Appointment.status)
    )
    return {str(status): int(count) for status, count in rows.tuples()}


async def list_appointments(
    session: AsyncSession, *, page: PageRequest, filters: AppointmentFilter
) -> PageResult[AppointmentView]:
    """Appointments ordered by start time, then id."""
    statement = _appointment_view_statement(filters.clinic_id)
    if filters.doctor_id is not None:
        statement = statement.where(Appointment.doctor_id == filters.doctor_id)
    if filters.patient_id is not None:
        statement = statement.where(Appointment.patient_id == filters.patient_id)
    if filters.status is not None:
        statement = statement.where(Appointment.status == filters.status)
    if filters.starts_from is not None:
        statement = statement.where(func.lower(Appointment.during) >= filters.starts_from)
    if filters.starts_before is not None:
        statement = statement.where(func.lower(Appointment.during) < filters.starts_before)

    rows, total = await fetch_page(session, statement, page)
    views = [_to_view(row) for row in rows]
    await _record_views(session, views)
    return PageResult(items=views, total=total, limit=page.limit, offset=page.offset)


async def get_appointment(
    session: AsyncSession, *, clinic_id: uuid.UUID, appointment_id: uuid.UUID
) -> AppointmentView | None:
    row = (
        await session.execute(
            _appointment_view_statement(clinic_id).where(Appointment.id == appointment_id)
        )
    ).first()
    if row is None:
        return None
    view = _to_view(row)
    await record_access(
        session,
        AccessAction.READ,
        Access(
            resource="appointments",
            resource_id=str(view.appointment.id),
            patient_id=view.appointment.patient_id,
        ),
    )
    return view


async def list_patient_appointments(
    session: AsyncSession, *, clinic_id: uuid.UUID, patient_id: uuid.UUID
) -> list[AppointmentView]:
    rows = await session.execute(
        _appointment_view_statement(clinic_id).where(Appointment.patient_id == patient_id)
    )
    views = [_to_view(row) for row in rows.all()]
    await _record_views(session, views)
    return views


@dataclass(frozen=True, slots=True)
class EscalationView:
    """An escalation with the patient's name, when it has a patient.

    The name is denormalised here for the same reason appointments carry it: the
    queue is read as a list, and a screen that joined per row would issue one
    query per line.
    """

    escalation: Escalation
    patient_given_names: str | None
    patient_family_names: str | None


def _escalation_statement(clinic_id: uuid.UUID) -> Select[Any]:
    """Open escalations first, then newest first within each status.

    A receptionist works the open ones; the resolved ones are history. Patients
    are joined outer because an escalation may have none -- an unrecognised
    sender is itself a reason to escalate, and nobody knows who it was.
    Soft-deleted patients are excluded from the join, as everywhere else.
    """
    return (
        select(
            Escalation,
            Patient.given_names,
            Patient.family_names,
        )
        .outerjoin(
            Patient,
            (Patient.id == Escalation.patient_id) & Patient.deleted_at.is_(None),
        )
        .where(Escalation.clinic_id == clinic_id)
        .order_by(
            # `open` sorts before `resolved` alphabetically, which is the order
            # wanted, but relying on that would break if a status were renamed.
            (Escalation.status != EscalationStatus.OPEN),
            Escalation.created_at.desc(),
            Escalation.id,
        )
    )


def _to_escalation_view(row: Any) -> EscalationView:
    escalation, given, family = row
    return EscalationView(
        escalation=escalation,
        patient_given_names=given,
        patient_family_names=family,
    )


async def _record_escalation_views(session: AsyncSession, views: list[EscalationView]) -> None:
    """One audit entry per escalation that names a patient.

    An escalation row carries a patient's name and the reason a human is needed,
    so reading the queue is reading patient data. Rows with no patient are not
    patient data and are not recorded as such.
    """
    accesses = [
        Access(
            resource="escalations",
            resource_id=str(view.escalation.id),
            patient_id=view.escalation.patient_id,
        )
        for view in views
        if view.escalation.patient_id is not None
    ]
    if accesses:
        await record_accesses(session, AccessAction.READ, accesses)


async def list_escalations(
    session: AsyncSession,
    *,
    clinic_id: uuid.UUID,
    page: PageRequest,
    status: EscalationStatus | None = None,
) -> PageResult[EscalationView]:
    """The queue staff work from. Open first, newest first."""
    statement = _escalation_statement(clinic_id)
    if status is not None:
        statement = statement.where(Escalation.status == status)

    rows, total = await fetch_page(session, statement, page)
    views = [_to_escalation_view(row) for row in rows]
    await _record_escalation_views(session, views)
    return PageResult(items=views, total=total, limit=page.limit, offset=page.offset)


async def count_open_escalations(session: AsyncSession, *, clinic_id: uuid.UUID) -> int:
    """How many things are waiting for a person. For a badge, so no names load."""
    return (
        await session.scalar(
            select(func.count())
            .select_from(Escalation)
            .where(
                Escalation.clinic_id == clinic_id,
                Escalation.status == EscalationStatus.OPEN,
            )
        )
    ) or 0


async def resolve_escalation(
    session: AsyncSession,
    *,
    clinic_id: uuid.UUID,
    escalation_id: uuid.UUID,
    now: datetime,
) -> EscalationView | None:
    """Mark one escalation dealt with. Returns None if there is nothing open.

    Resolving is the only change a human can make to the row: it is never
    deleted, so nothing leaves the queue without a record of who closed it and
    when. Re-resolving an already-resolved row returns None rather than moving
    `resolved_at`, because the first person dealt with it and that is the time
    that matters.
    """
    escalation = await session.scalar(
        select(Escalation).where(
            Escalation.id == escalation_id,
            Escalation.clinic_id == clinic_id,
            Escalation.status == EscalationStatus.OPEN,
        )
    )
    if escalation is None:
        return None

    escalation.status = EscalationStatus.RESOLVED
    escalation.resolved_at = now
    await record_access(
        session,
        AccessAction.UPDATE,
        Access(
            resource="escalations",
            resource_id=str(escalation.id),
            patient_id=escalation.patient_id,
        ),
    )
    await session.flush()

    row = (
        await session.execute(
            _escalation_statement(clinic_id).where(Escalation.id == escalation_id)
        )
    ).first()
    return None if row is None else _to_escalation_view(row)
