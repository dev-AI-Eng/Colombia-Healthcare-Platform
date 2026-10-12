"""Booking, holding, rescheduling and cancelling, as one transaction each.

Every write to `appointments` goes through here. The exclusion constraint on
that table is what makes double-booking impossible -- not this module -- and
the job of this code is to work *with* it:

  * it writes inside the caller's transaction, so the appointment and its audit
    entry commit together or not at all;
  * it turns a lost race into a value the caller can act on, rather than
    letting a `DBAPIError` escape into a conversation flow or an HTTP handler;
  * it never checks for a conflict first and writes second. That check-then-act
    pattern is exactly what the constraint exists to replace: between the
    SELECT and the INSERT another transaction can commit, and the only thing
    that reliably notices is the database.

WHAT A LOST RACE LOOKS LIKE
---------------------------
Two transactions inserting overlapping appointments for one doctor produce
either `exclusion_violation` (23P01), when the loser's insert is rejected
outright, or `deadlock_detected` (40P01), when the two inserts wait on each
other and PostgreSQL breaks the cycle. Both mean the same thing to a caller --
somebody else took the time -- and `SAVEPOINT` is what lets the caller carry
on afterwards: without it the whole transaction is poisoned and the audit
entries written before the attempt are lost with it.

A unique violation on `idempotency_key` (23505) is a different outcome again,
and the one that keeps a redelivered webhook from creating a second booking.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import Range
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from src.audit.context import AccessAction
from src.audit.service import Access, record_access
from src.scheduling.models import (
    ACTIVE_STATUSES,
    Appointment,
    AppointmentStatus,
)

#: The SQLSTATEs that mean "another transaction took this time".
#:
#: 23P01 is the exclusion constraint rejecting the overlap. 40P01 is two
#: inserts having waited on each other until PostgreSQL broke the cycle; the
#: victim is chosen arbitrarily, so a deadlock here is a lost race and not a
#: fault. Treating either as an error the caller cannot handle would surface a
#: database code to a receptionist.
LOST_RACE_SQLSTATES: Final = frozenset({"23P01", "40P01"})

#: A redelivered webhook carrying an `idempotency_key` already used.
UNIQUE_VIOLATION: Final = "23505"

#: How long an offered slot is held while the patient decides. ADR-17 makes
#: this per-clinic eventually; until clinics can configure it, fifteen minutes
#: is the documented default.
DEFAULT_HOLD_MINUTES: Final = 15


class Outcome(StrEnum):
    """Why a booking attempt ended the way it did."""

    BOOKED = "booked"
    #: Somebody else holds or has booked an overlapping time.
    TAKEN = "taken"
    #: This exact request was already applied; the existing appointment is returned.
    DUPLICATE = "duplicate"


@dataclass(frozen=True, slots=True)
class BookingResult:
    """What happened, and the appointment if there is one.

    `appointment` is set for `BOOKED` and for `DUPLICATE` -- in the duplicate
    case it is the booking the earlier request created, which is what an
    idempotent caller wants back. It is None only when the time was taken.
    """

    outcome: Outcome
    appointment: Appointment | None = None

    @property
    def ok(self) -> bool:
        """Whether the caller ended up with an appointment."""
        return self.appointment is not None


def _sqlstate(error: DBAPIError) -> str | None:
    return getattr(error.orig, "sqlstate", None)


async def _existing_by_key(session: AsyncSession, idempotency_key: str) -> Appointment | None:
    """The appointment a previous request created under this key.

    Not scoped by clinic: the column is unique across the whole table, so the
    row that caused the violation is the row to return, and filtering by clinic
    would simply fail to find it. Row-level security still applies, so a key
    belonging to another clinic returns nothing rather than leaking a row.
    """
    stmt = select(Appointment).where(Appointment.idempotency_key == idempotency_key)
    return (await session.execute(stmt)).scalar_one_or_none()


async def _insert(session: AsyncSession, appointment: Appointment) -> BookingResult:
    """Insert one appointment, turning a lost race into an outcome.

    The insert runs inside a SAVEPOINT. A constraint violation rolls back only
    that savepoint, so the caller's transaction survives and whatever it wrote
    beforehand -- audit entries especially -- is still there to commit.
    """
    try:
        async with session.begin_nested():
            session.add(appointment)
            await session.flush()
    except DBAPIError as error:
        state = _sqlstate(error)
        if state in LOST_RACE_SQLSTATES:
            return BookingResult(Outcome.TAKEN)
        if state == UNIQUE_VIOLATION and appointment.idempotency_key:
            existing = await _existing_by_key(session, appointment.idempotency_key)
            if existing is not None:
                return BookingResult(Outcome.DUPLICATE, existing)
        raise
    return BookingResult(Outcome.BOOKED, appointment)


async def book(
    session: AsyncSession,
    *,
    clinic_id: uuid.UUID,
    patient_id: uuid.UUID,
    doctor_id: uuid.UUID,
    location_id: uuid.UUID,
    appointment_type_id: uuid.UUID,
    start: dt.datetime,
    end: dt.datetime,
    status: AppointmentStatus = AppointmentStatus.SCHEDULED,
    idempotency_key: str | None = None,
) -> BookingResult:
    """Book one appointment, or report that the time is taken.

    No availability check happens here. Whether the time is *offerable* is
    `availability.bookable_slots`; whether it is still *free* is the
    constraint, decided at the moment of writing. Asking first would be a
    check-then-act race, and it would also be slower.

    The audit entry is written only on success, inside the same transaction as
    the appointment, so the log never claims a booking that did not happen.
    """
    if end <= start:
        raise ValueError("an appointment must end after it starts")
    if status not in ACTIVE_STATUSES:
        raise ValueError(f"{status} does not occupy the doctor's time; use it for an update")
    if status is AppointmentStatus.HOLD:
        raise ValueError("use `hold` to create a hold, which needs an expiry")

    result = await _insert(
        session,
        Appointment(
            clinic_id=clinic_id,
            patient_id=patient_id,
            doctor_id=doctor_id,
            location_id=location_id,
            appointment_type_id=appointment_type_id,
            during=Range(start, end),
            status=status,
            idempotency_key=idempotency_key,
        ),
    )
    if result.outcome is Outcome.BOOKED and result.appointment is not None:
        await record_access(
            session,
            AccessAction.CREATE,
            Access(
                resource="appointments",
                resource_id=str(result.appointment.id),
                patient_id=patient_id,
            ),
        )
    return result


async def hold(
    session: AsyncSession,
    *,
    clinic_id: uuid.UUID,
    patient_id: uuid.UUID,
    doctor_id: uuid.UUID,
    location_id: uuid.UUID,
    appointment_type_id: uuid.UUID,
    start: dt.datetime,
    end: dt.datetime,
    now: dt.datetime,
    minutes: int = DEFAULT_HOLD_MINUTES,
    idempotency_key: str | None = None,
) -> BookingResult:
    """Hold a slot while the patient decides (ADR-17).

    A hold is an appointment in `hold` status with an expiry, so it takes part
    in the same exclusion constraint: a hold blocks a booking and a booking
    blocks a hold. That is the whole point -- offering a slot over a messaging
    channel means minutes pass before the reply, and without a hold the system
    either double-books or tells the patient their choice vanished.

    `now` is passed rather than read from the clock so a caller can test the
    boundary, and so every timestamp in one request agrees.
    """
    if minutes <= 0:
        raise ValueError("a hold must last a positive number of minutes")
    if end <= start:
        raise ValueError("an appointment must end after it starts")

    result = await _insert(
        session,
        Appointment(
            clinic_id=clinic_id,
            patient_id=patient_id,
            doctor_id=doctor_id,
            location_id=location_id,
            appointment_type_id=appointment_type_id,
            during=Range(start, end),
            status=AppointmentStatus.HOLD,
            expires_at=now + dt.timedelta(minutes=minutes),
            idempotency_key=idempotency_key,
        ),
    )
    if result.outcome is Outcome.BOOKED and result.appointment is not None:
        await record_access(
            session,
            AccessAction.CREATE,
            Access(
                resource="appointments",
                resource_id=str(result.appointment.id),
                patient_id=patient_id,
            ),
        )
    return result


async def confirm_hold(
    session: AsyncSession,
    *,
    clinic_id: uuid.UUID,
    appointment_id: uuid.UUID,
    now: dt.datetime,
) -> BookingResult:
    """Turn a live hold into a booking.

    The row is locked before its expiry is read, so a sweep cannot delete it
    between the check and the update. An expired hold is reported as `TAKEN`
    rather than silently confirmed: the time may since have gone to somebody
    else, and the conversation flow re-offers (ADR-17).

    Clearing `expires_at` is required, not cosmetic: a CHECK constraint ties
    the expiry to `hold` status, so a scheduled row carrying one is rejected.
    """
    stmt = (
        select(Appointment)
        .where(Appointment.clinic_id == clinic_id, Appointment.id == appointment_id)
        .with_for_update()
    )
    held = (await session.execute(stmt)).scalar_one_or_none()
    if held is None or held.status is not AppointmentStatus.HOLD:
        return BookingResult(Outcome.TAKEN)
    if held.expires_at is not None and held.expires_at <= now:
        return BookingResult(Outcome.TAKEN)

    held.status = AppointmentStatus.SCHEDULED
    held.expires_at = None
    await session.flush()
    await record_access(
        session,
        AccessAction.UPDATE,
        Access(resource="appointments", resource_id=str(held.id), patient_id=held.patient_id),
    )
    return BookingResult(Outcome.BOOKED, held)


async def release_hold(
    session: AsyncSession, *, clinic_id: uuid.UUID, appointment_id: uuid.UUID
) -> bool:
    """Give a held slot back, because the patient declined or moved on.

    Deletes rather than cancels. A hold that was never confirmed is not part of
    the patient's history -- it is an offer nobody accepted -- and leaving
    cancelled rows behind would fill the record with noise. A real booking is
    cancelled, never deleted; that is `cancel`.
    """
    stmt = (
        select(Appointment)
        .where(
            Appointment.clinic_id == clinic_id,
            Appointment.id == appointment_id,
            Appointment.status == AppointmentStatus.HOLD,
        )
        .with_for_update()
    )
    held = (await session.execute(stmt)).scalar_one_or_none()
    if held is None:
        return False
    patient_id = held.patient_id
    await session.delete(held)
    await session.flush()
    await record_access(
        session,
        AccessAction.DELETE,
        Access(resource="appointments", resource_id=str(appointment_id), patient_id=patient_id),
    )
    return True


async def sweep_expired_holds(
    session: AsyncSession, *, clinic_id: uuid.UUID, now: dt.datetime
) -> int:
    """Delete holds whose time has run out. Returns how many went.

    Housekeeping, not correctness: `availability` already treats an expired
    hold as free, so a slot is bookable the moment the hold lapses whether or
    not this has run. What the sweep buys is a table that does not grow
    without bound, and a schedule a human reading the database can follow.
    """
    stmt = (
        select(Appointment)
        .where(
            Appointment.clinic_id == clinic_id,
            Appointment.status == AppointmentStatus.HOLD,
            Appointment.expires_at <= now,
        )
        .with_for_update(skip_locked=True)
    )
    expired = list((await session.execute(stmt)).scalars())
    for held in expired:
        await session.delete(held)
    if expired:
        await session.flush()
    return len(expired)


async def cancel(
    session: AsyncSession,
    *,
    clinic_id: uuid.UUID,
    appointment_id: uuid.UUID,
    reason: str | None = None,
) -> bool:
    """Cancel a booking, freeing the doctor's time.

    The row stays. `cancelled` is outside the exclusion constraint's active
    statuses, so the time is immediately bookable again, while the patient's
    history keeps the fact that an appointment existed and was cancelled --
    which matters for a no-show pattern and for the audit trail.
    """
    stmt = (
        select(Appointment)
        .where(Appointment.clinic_id == clinic_id, Appointment.id == appointment_id)
        .with_for_update()
    )
    found = (await session.execute(stmt)).scalar_one_or_none()
    if found is None or found.status not in ACTIVE_STATUSES:
        return False

    found.status = AppointmentStatus.CANCELLED
    found.expires_at = None  # the CHECK ties an expiry to `hold` status
    if reason:
        found.cancellation_reason = reason
    await session.flush()
    await record_access(
        session,
        AccessAction.UPDATE,
        Access(resource="appointments", resource_id=str(found.id), patient_id=found.patient_id),
    )
    return True


async def reschedule(
    session: AsyncSession,
    *,
    clinic_id: uuid.UUID,
    appointment_id: uuid.UUID,
    start: dt.datetime,
    end: dt.datetime,
    idempotency_key: str | None = None,
) -> BookingResult:
    """Move an appointment to a new time, or report the new time is taken.

    The new appointment is written *before* the old one is released, which is
    the order that matters: releasing first would open the original time to
    another patient while this one might still fail, leaving the patient with
    no appointment at all. If the new time is taken, nothing changed.

    The old row becomes `rescheduled` rather than being deleted, so the history
    shows the move. It is locked for the whole operation, so two concurrent
    reschedules of the same appointment cannot both proceed.
    """
    stmt = (
        select(Appointment)
        .where(Appointment.clinic_id == clinic_id, Appointment.id == appointment_id)
        .with_for_update()
    )
    original = (await session.execute(stmt)).scalar_one_or_none()
    if original is None or original.status not in ACTIVE_STATUSES:
        return BookingResult(Outcome.TAKEN)

    result = await _insert(
        session,
        Appointment(
            clinic_id=original.clinic_id,
            patient_id=original.patient_id,
            doctor_id=original.doctor_id,
            location_id=original.location_id,
            appointment_type_id=original.appointment_type_id,
            during=Range(start, end),
            status=AppointmentStatus.SCHEDULED,
            rescheduled_from_appointment_id=original.id,
            idempotency_key=idempotency_key,
        ),
    )
    if result.outcome is not Outcome.BOOKED or result.appointment is None:
        return result  # the new time was taken; the original is untouched

    original.status = AppointmentStatus.RESCHEDULED
    original.expires_at = None
    await session.flush()
    await record_access(
        session,
        AccessAction.CREATE,
        Access(
            resource="appointments",
            resource_id=str(result.appointment.id),
            patient_id=original.patient_id,
        ),
    )
    return result
