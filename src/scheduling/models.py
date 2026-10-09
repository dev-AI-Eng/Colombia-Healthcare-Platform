"""Doctor availability and appointments.

Availability is stored as weekly rules plus dated exceptions, never as
precomputed slots (PDF M2: "real doctor availability model, not fixed slots").

Double-booking is prevented by the database (ADR-16). The initial migration adds
an exclusion constraint on `appointments` that rejects two rows for the same
doctor whose `during` ranges overlap while both are in an active status.

A slot hold (ADR-17) is an appointment row with status `hold` and an
`expires_at`. Holds and bookings therefore share the one exclusion constraint,
so a hold blocks a booking and a booking blocks a hold.

When several transactions insert conflicting rows at the same time, the losers
fail with `exclusion_violation` (SQLSTATE 23P01) or, because the inserts wait on
each other, `deadlock_detected` (40P01). Booking code must treat both as a lost
race for the slot. tests/integration/test_constraints.py exercises this.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, time
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Time,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSTZRANGE, ExcludeConstraint, Range
from sqlalchemy.orm import Mapped, mapped_column

from src.core.db import Base, string_enum


class AppointmentStatus(StrEnum):
    HOLD = "hold"
    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    CHECKED_IN = "checked_in"
    COMPLETED = "completed"
    NO_SHOW = "no_show"
    CANCELLED = "cancelled"
    RESCHEDULED = "rescheduled"


# Statuses that occupy the doctor's time and so take part in the overlap constraint.
ACTIVE_STATUSES: tuple[AppointmentStatus, ...] = (
    AppointmentStatus.HOLD,
    AppointmentStatus.SCHEDULED,
    AppointmentStatus.CONFIRMED,
    AppointmentStatus.CHECKED_IN,
)
_ACTIVE_SQL = "(" + ", ".join(f"'{status.value}'" for status in ACTIVE_STATUSES) + ")"


class Sensitivity(StrEnum):
    NORMAL = "normal"
    # Reserved for ADR-18: a second identity factor before disclosing details.
    HIGH = "high"


class ExceptionKind(StrEnum):
    UNAVAILABLE = "unavailable"
    EXTRA_HOURS = "extra_hours"


class AppointmentType(Base):
    __tablename__ = "appointment_types"
    __table_args__ = (
        CheckConstraint("duration_minutes > 0", name="duration_positive"),
        CheckConstraint("buffer_minutes >= 0", name="buffer_non_negative"),
        Index("ix_appointment_types_clinic_id", "clinic_id"),
        {"schema": "app"},
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    clinic_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("app.clinics.id"))
    name: Mapped[str] = mapped_column(String(100))
    duration_minutes: Mapped[int] = mapped_column(Integer)
    buffer_minutes: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    sensitivity: Mapped[Sensitivity] = mapped_column(
        string_enum(Sensitivity, "sensitivity_valid"), server_default=Sensitivity.NORMAL.value
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AvailabilityRule(Base):
    """A recurring weekly window during which a doctor sees patients."""

    __tablename__ = "availability_rules"
    __table_args__ = (
        CheckConstraint("weekday BETWEEN 0 AND 6", name="weekday_range"),
        CheckConstraint("end_time > start_time", name="time_window_ordered"),
        CheckConstraint(
            "valid_until IS NULL OR valid_until >= valid_from", name="validity_ordered"
        ),
        Index("ix_availability_rules_doctor_id", "doctor_id"),
        {"schema": "app"},
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    clinic_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("app.clinics.id"))
    doctor_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("app.doctors.id"))
    location_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("app.locations.id"))
    weekday: Mapped[int] = mapped_column(SmallInteger)  # 0 = Monday, as in date.weekday()
    start_time: Mapped[time] = mapped_column(Time)
    end_time: Mapped[time] = mapped_column(Time)
    valid_from: Mapped[date] = mapped_column(Date)
    valid_until: Mapped[date | None] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AvailabilityException(Base):
    """A dated change to a doctor's availability.

    `unavailable` with no times blocks the whole day; with times it blocks that
    window. `extra_hours` always has times and adds a window.
    """

    __tablename__ = "availability_exceptions"
    __table_args__ = (
        CheckConstraint("(start_time IS NULL) = (end_time IS NULL)", name="times_paired"),
        CheckConstraint("start_time IS NULL OR end_time > start_time", name="time_window_ordered"),
        CheckConstraint(
            "kind <> 'extra_hours' OR start_time IS NOT NULL", name="extra_hours_has_times"
        ),
        Index("ix_availability_exceptions_doctor_id_on_date", "doctor_id", "on_date"),
        {"schema": "app"},
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    clinic_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("app.clinics.id"))
    doctor_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("app.doctors.id"))
    on_date: Mapped[date] = mapped_column(Date)
    kind: Mapped[ExceptionKind] = mapped_column(string_enum(ExceptionKind, "kind_valid"))
    start_time: Mapped[time | None] = mapped_column(Time)
    end_time: Mapped[time | None] = mapped_column(Time)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Appointment(Base):
    __tablename__ = "appointments"
    __table_args__ = (
        ExcludeConstraint(
            ("doctor_id", "="),
            ("during", "&&"),
            name="no_overlapping_active_appointments",
            using="gist",
            where=text(f"status IN {_ACTIVE_SQL}"),
        ),
        CheckConstraint(
            "NOT isempty(during) AND NOT lower_inf(during) AND NOT upper_inf(during)",
            name="during_bounded",
        ),
        CheckConstraint(
            "(status = 'hold') = (expires_at IS NOT NULL)", name="expiry_only_for_holds"
        ),
        Index(
            "uq_appointments_clinic_id_external_ref",
            "clinic_id",
            "external_ref",
            unique=True,
            postgresql_where=text("external_ref IS NOT NULL"),
        ),
        Index("ix_appointments_patient_id", "patient_id"),
        Index("ix_appointments_clinic_id_start", "clinic_id", func.lower(text("during"))),
        {"schema": "app"},
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    clinic_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("app.clinics.id"))
    patient_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("app.patients.id"))
    doctor_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("app.doctors.id"))
    location_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("app.locations.id"))
    appointment_type_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("app.appointment_types.id"))
    # Half-open [start, end) interval, so back-to-back appointments do not overlap.
    during: Mapped[Range[datetime]] = mapped_column(TSTZRANGE)
    status: Mapped[AppointmentStatus] = mapped_column(
        string_enum(AppointmentStatus, "status_valid")
    )
    # Set exactly when status is `hold`; enforced by a CHECK constraint.
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Lets a retried request create at most one appointment.
    idempotency_key: Mapped[str | None] = mapped_column(String(120), unique=True)
    # Fields the clinic's own export carries (migration 0005).
    consultation_type: Mapped[str | None] = mapped_column(String(40))
    source: Mapped[str | None] = mapped_column(String(60))
    cancellation_reason: Mapped[str | None] = mapped_column(String(200))
    # The client's files model a reschedule as cancelling one row and creating
    # another that points back at it, so the link is an appointment, not a status.
    rescheduled_from_appointment_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("app.appointments.id")
    )
    preferred_channel: Mapped[str | None] = mapped_column(String(30))
    reminder_48h_sent: Mapped[bool] = mapped_column(Boolean, server_default=text("false"))
    reminder_24h_sent: Mapped[bool] = mapped_column(Boolean, server_default=text("false"))
    external_ref: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EscalationReason(StrEnum):
    """Why the system handed a patient to a human.

    A closed set, because an escalation queue sorted by free-form text is a
    queue nobody triages. `no_acceptable_slot` is the scheduling one and the
    only one M2 raises; the rest are declared here so M3's conversation graph
    writes into the same table rather than inventing a second one.
    """

    NO_ACCEPTABLE_SLOT = "no_acceptable_slot"
    MISSING_CONSENT = "missing_consent"
    UNRECOGNISED_SENDER = "unrecognised_sender"
    CLINICAL_CONCERN = "clinical_concern"
    EVALUATOR_EXHAUSTED = "evaluator_exhausted"
    PATIENT_ASKED_FOR_A_HUMAN = "patient_asked_for_a_human"


class EscalationStatus(StrEnum):
    OPEN = "open"
    RESOLVED = "resolved"


class Escalation(Base):
    """One thing a human has to deal with.

    Raised when the system reaches a state it must not resolve on its own.
    Scheduling raises exactly one kind -- no slot the patient can accept --
    and that is PDF criterion 2 for M2: the fallback has to produce a record
    staff can act on, not a silent dead end.

    `detail_es` is Spanish because a receptionist reads it. `context` carries
    whatever the raising code knows in machine-readable form, so a later
    screen can show the search that failed without re-running it.
    """

    __tablename__ = "escalations"
    __table_args__ = (
        Index("ix_escalations_clinic_id_status", "clinic_id", "status"),
        {"schema": "app"},
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    clinic_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("app.clinics.id"))
    # Null when the sender could not be identified, which is itself a reason.
    patient_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("app.patients.id"))
    appointment_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("app.appointments.id"))
    reason: Mapped[EscalationReason] = mapped_column(string_enum(EscalationReason, "reason_valid"))
    status: Mapped[EscalationStatus] = mapped_column(
        string_enum(EscalationStatus, "status_valid"), server_default=EscalationStatus.OPEN.value
    )
    detail_es: Mapped[str] = mapped_column(String(500))
    context: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
