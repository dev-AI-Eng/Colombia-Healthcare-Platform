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

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import Range
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from src.audit.context import AccessAction
from src.audit.service import Access, record_access
from src.core.timezones import BOGOTA
from src.scheduling.availability import Exception_, Rule, open_windows
from src.scheduling.events import ACTOR_STAFF, record_event
from src.scheduling.models import (
    ACTIVE_STATUSES,
    Appointment,
    AppointmentEventKind,
    AppointmentStatus,
    AppointmentType,
    AvailabilityException,
    AvailabilityRule,
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


#: The fields that make two requests under one key the same request.
#:
#: QA finding 5. The key alone was enough to return the earlier appointment,
#: so a key reused for a different operation -- a hold's key sent with a
#: booking, a key sent with a second patient's id -- answered with somebody
#: else's appointment and the API reported it as success. These are the fields
#: a caller is asserting are unchanged when it replays a request; `status`
#: covers the hold-versus-booking case, because a hold and a booking differ by
#: exactly that.
_IDENTITY: Final = (
    "clinic_id",
    "patient_id",
    "doctor_id",
    "location_id",
    "appointment_type_id",
    "status",
)


def _same_request(existing: Appointment, requested: Appointment) -> bool:
    """Whether `requested` is a replay of the request that wrote `existing`.

    The time span is compared separately: `during` is a Range, and the bounds
    are what a caller means by "the same appointment", not the object identity.
    """
    if any(getattr(existing, field) != getattr(requested, field) for field in _IDENTITY):
        return False
    return (
        existing.during.lower == requested.during.lower
        and existing.during.upper == requested.during.upper
    )


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
                if not _same_request(existing, appointment):
                    # QA finding 5: the key has been used for a different
                    # request. Returning the earlier appointment would report
                    # another operation's result as this one's success, and the
                    # caller would believe it had booked what it asked for.
                    raise KeyReused(appointment.idempotency_key) from error
                return BookingResult(Outcome.DUPLICATE, existing)
        raise
    return BookingResult(Outcome.BOOKED, appointment)


class KeyReused(Exception):
    """An idempotency key was sent with a different request than it first carried.

    QA finding 5. The key is the caller's assertion that this is the same
    operation it already sent; sending it with a different patient, doctor,
    time or status is either a bug in the caller or a collision, and both have
    to be loud. Silently returning the first appointment told the caller it had
    booked something it had not.
    """

    def __init__(self, idempotency_key: str) -> None:
        super().__init__(f"idempotency key {idempotency_key!r} was used for a different request")
        self.idempotency_key = idempotency_key
        self.detail_es = (
            "Esa clave de solicitud ya se usó para otra cita. "
            "Envíe la misma solicitud o use una clave nueva."
        )


class Ineligible(Exception):
    """The requested time is not one the clinic offers.

    QA finding 3. The exclusion constraint answers "is this time still free",
    which is a question only the database can answer at the moment of writing.
    It does not answer "is this a time the clinic works at all", and nothing
    else did either: a direct request booked 15:00 for a doctor whose hours end
    at 12:00, and 03:00 on a Sunday, because no appointment conflicted.

    Raised rather than returned as an `Outcome`, because the two are different
    kinds of answer. TAKEN means try another time; this means the request was
    never valid, and a caller that cannot tell them apart would offer the
    patient a retry that can never succeed.

    `detail_es` is what a receptionist reads. The exception carries both so the
    HTTP layer does not have to translate.
    """

    def __init__(self, reason: str, detail_es: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail_es = detail_es


async def check_eligible(
    session: AsyncSession,
    *,
    clinic_id: uuid.UUID,
    doctor_id: uuid.UUID,
    appointment_type_id: uuid.UUID,
    start: dt.datetime,
    end: dt.datetime,
    now: dt.datetime,
    location_id: uuid.UUID | None = None,
) -> None:
    """Raise `Ineligible` unless the clinic actually offers this time.

    Checks what the constraint cannot: the appointment type's duration, the
    doctor's weekly rules, the dated exceptions, and Colombian public holidays.
    What it deliberately does *not* check is whether the time is still free --
    that is the constraint's job, decided at the moment of writing, and asking
    here would be the check-then-act race this module exists to avoid.

    So `busy` is passed empty on purpose. A slot occupied by another
    appointment is eligible and then loses the race, which is the correct pair
    of answers: the clinic does work then, and somebody else got there first.
    """
    if start < now:
        # `now` was accepted by this function, threaded through three call
        # sites and the HTTP layer, and never read -- so nothing refused a
        # booking in the past. `availability.bookable_slots` drops a past slot
        # with the comment "offering a patient a time that has already gone is
        # worse than offering nothing", and the write path had no equivalent:
        # a slow conversation turn or a redelivered webhook could book a time
        # that had already passed, and `hold` could mint a hold whose expiry
        # was already behind the clock -- a dead row still occupying the
        # exclusion constraint until a sweep ran.
        #
        # Transcribing genuine history is what `override_reason` is for, which
        # is how the spreadsheet importer books; this refusal does not apply to
        # it, because the caller states that intent explicitly.
        local = start.astimezone(BOGOTA)
        raise Ineligible(
            f"{start.isoformat()} is before {now.isoformat()}",
            f"Esa hora ya pasó ({local.strftime('%d/%m/%Y %I:%M %p').lower()}). "
            f"Elija una hora futura.",
        )

    appointment_type = await session.get(AppointmentType, appointment_type_id)
    if appointment_type is None:
        raise Ineligible(
            f"no appointment type {appointment_type_id}",
            "El tipo de cita no existe. Verifique el servicio seleccionado.",
        )

    minutes = (end - start).total_seconds() / 60
    if minutes != appointment_type.duration_minutes:
        raise Ineligible(
            f"{appointment_type.name} lasts {appointment_type.duration_minutes} minutes, "
            f"not {minutes:g}",
            f"Una cita de {appointment_type.name} dura "
            f"{appointment_type.duration_minutes} minutos. "
            f"La solicitud pedía {minutes:g}.",
        )

    local = start.astimezone(BOGOTA)
    day = local.date()
    rules = await _rules_for(
        session, clinic_id=clinic_id, doctor_id=doctor_id, location_id=location_id
    )
    exceptions = await _exceptions_for(session, clinic_id=clinic_id, doctor_id=doctor_id, day=day)
    windows = open_windows(day, rules, exceptions)
    if not windows:
        raise Ineligible(
            f"the doctor has no hours on {day.isoformat()}",
            f"El profesional no atiende el {local.strftime('%d/%m/%Y')}.",
        )

    # The requested interval has to sit inside one open window, not merely
    # touch one: an appointment that starts at 11:50 in hours ending at 12:00
    # runs twenty minutes past closing.
    for window_start, window_end in (w.as_instants() for w in windows):
        if window_start <= start and end <= window_end:
            return

    offered = ", ".join(f"{w.start:%H:%M}-{w.end:%H:%M}" for w in windows)
    raise Ineligible(
        f"{start.isoformat()} to {end.isoformat()} falls outside the doctor's hours ({offered})",
        f"El profesional atiende el {local.strftime('%d/%m/%Y')} de {offered}. "
        f"La cita solicitada ({local.strftime('%I:%M %p').lstrip('0').lower()}) "
        f"queda fuera de ese horario.",
    )


async def _rules_for(
    session: AsyncSession,
    *,
    clinic_id: uuid.UUID,
    doctor_id: uuid.UUID,
    location_id: uuid.UUID | None = None,
) -> list[Rule]:
    """This doctor's weekly rules, narrowed to one site when given.

    The location filter is the write path's half of QA finding 9. `offers`
    gained it; this did not, so the eligibility check answered "does the doctor
    work at 09:00 on a Tuesday" from *every* site's hours and a booking was
    accepted at a sede where the doctor has no rules at all. Measured: a search
    at Sede Sur returned 0 slots while a booking at Sede Sur was accepted, on a
    doctor whose only rule was at Sede Norte.

    `None` means every site, which is right for a single-site clinic and is the
    behaviour a caller that cannot name a location should get.
    """
    stmt = select(AvailabilityRule).where(
        AvailabilityRule.clinic_id == clinic_id,
        AvailabilityRule.doctor_id == doctor_id,
    )
    if location_id is not None:
        stmt = stmt.where(AvailabilityRule.location_id == location_id)
    rows = await session.scalars(stmt)
    return [
        Rule(
            weekday=r.weekday,
            start=r.start_time,
            end=r.end_time,
            valid_from=r.valid_from,
            valid_until=r.valid_until,
        )
        for r in rows
    ]


async def _exceptions_for(
    session: AsyncSession, *, clinic_id: uuid.UUID, doctor_id: uuid.UUID, day: dt.date
) -> list[Exception_]:
    rows = await session.scalars(
        select(AvailabilityException).where(
            AvailabilityException.clinic_id == clinic_id,
            AvailabilityException.doctor_id == doctor_id,
            AvailabilityException.on_date == day,
        )
    )
    return [
        Exception_(on_date=e.on_date, kind=str(e.kind), start=e.start_time, end=e.end_time)
        for e in rows
    ]


async def _reclaim_expired_holds(
    session: AsyncSession,
    *,
    clinic_id: uuid.UUID,
    doctor_id: uuid.UUID,
    start: dt.datetime,
    end: dt.datetime,
    now: dt.datetime,
) -> int:
    """Delete the expired holds overlapping this time, in the caller's transaction.

    QA finding 2. `availability` stops counting a hold once it expires, while
    the exclusion constraint counts every `hold` row whatever its expiry -- so
    without this, a slot the availability API advertised as free was refused by
    the booking API until the five-minute sweep caught up.

    Fixed here rather than in the constraint because it cannot be fixed there:
    an exclusion predicate may only use IMMUTABLE functions, and `now()` is
    STABLE. PostgreSQL rejects it outright.

    Safe under concurrency for two reasons. The delete and the insert that
    follows share one transaction, so a rolled-back booking takes the reclaim
    with it and no hold is lost to a failed attempt. And `FOR UPDATE SKIP
    LOCKED` means two simultaneous bookings do not deadlock on the same expired
    row -- the second simply does not see it, and then loses to the constraint
    on the insert, which is the outcome it should have anyway.

    Only *expired* holds, and only ones overlapping the requested time. A live
    hold still blocks, which is the whole point of holding a slot.
    """
    overlapping = (
        select(Appointment.id)
        .where(
            Appointment.clinic_id == clinic_id,
            Appointment.doctor_id == doctor_id,
            Appointment.status == AppointmentStatus.HOLD,
            Appointment.expires_at.is_not(None),
            Appointment.expires_at <= now,
            Appointment.during.op("&&")(Range(start, end)),
        )
        .with_for_update(skip_locked=True)
    )
    expired = list((await session.scalars(overlapping)).all())
    if not expired:
        return 0
    await session.execute(delete(Appointment).where(Appointment.id.in_(expired)))
    await session.flush()
    return len(expired)


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
    now: dt.datetime | None = None,
    override_reason: str | None = None,
) -> BookingResult:
    """Book one appointment, or report that the time is taken.

    Two different questions are answered in two different places, and QA
    finding 3 was that only one of them was being asked. Whether the clinic
    *offers* this time -- the doctor's hours, the dated exceptions, the public
    holidays, the appointment type's duration -- is `check_eligible`, called
    here. Whether the time is still *free* is the exclusion constraint, decided
    at the moment of writing; this code never checks that first, because
    between a SELECT and an INSERT another transaction can commit.

    `override_reason` is the authorised manual booking the report asks for: a
    receptionist squeezing a patient in outside hours is a real thing a clinic
    does, so it is a named operation with a reason recorded in the audit entry,
    not a silent bypass. The reason is required, not optional, so an override
    cannot happen by forgetting an argument.

    The audit entry is written only on success, inside the same transaction as
    the appointment, so the log never claims a booking that did not happen.
    """
    if end <= start:
        raise ValueError("an appointment must end after it starts")
    if status not in ACTIVE_STATUSES:
        raise ValueError(f"{status} does not occupy the doctor's time; use it for an update")
    if status is AppointmentStatus.HOLD:
        raise ValueError("use `hold` to create a hold, which needs an expiry")

    moment = now or dt.datetime.now(tz=dt.UTC)
    if override_reason is None:
        await check_eligible(
            session,
            clinic_id=clinic_id,
            doctor_id=doctor_id,
            appointment_type_id=appointment_type_id,
            location_id=location_id,
            start=start,
            end=end,
            now=moment,
        )

    # QA finding 2: an expired hold still blocks the constraint, so it is
    # reclaimed here -- in this transaction, before the insert that would
    # otherwise be refused for a slot availability already called free.
    await _reclaim_expired_holds(
        session,
        clinic_id=clinic_id,
        doctor_id=doctor_id,
        start=start,
        end=end,
        now=moment,
    )

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
        if override_reason is not None:
            # A second entry, against its own resource, so an auditor can ask
            # "which bookings bypassed the clinic's hours" and get an answer
            # without reading every appointment. The reason is the staff
            # member's own words; it is not patient data and is stored as given.
            await record_access(
                session,
                AccessAction.CREATE,
                Access(
                    resource="appointment_overrides",
                    resource_id=str(result.appointment.id),
                    patient_id=patient_id,
                    fields_disclosed=(f"reason={override_reason}",),
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
    override_reason: str | None = None,
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

    # QA finding 3 applies to a hold as much as a booking: a hold the clinic
    # would never honour still blocks the slot for everyone else.
    if override_reason is None:
        await check_eligible(
            session,
            clinic_id=clinic_id,
            doctor_id=doctor_id,
            appointment_type_id=appointment_type_id,
            location_id=location_id,
            start=start,
            end=end,
            now=now,
        )

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
    actor: str = ACTOR_STAFF,
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
    # ADR-14, in this transaction: the event and the status change land
    # together or not at all.
    await record_event(
        session,
        appointment=found,
        kind=AppointmentEventKind.CANCELLED,
        actor=actor,
        reason=reason,
    )
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
    actor: str = ACTOR_STAFF,
    now: dt.datetime | None = None,
    override_reason: str | None = None,
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

    # The key is consulted before the status, and the order is the point. A
    # move that succeeded leaves the original `rescheduled`, so a retried
    # request -- a timeout, a redelivered webhook -- fell into the status guard
    # below and was told TAKEN. The caller then concluded the slot had been
    # lost and offered the patient another time, while the move it asked for
    # had in fact happened. Found by the QA finding 5 replay test.
    if idempotency_key is not None:
        replayed = await _existing_by_key(session, idempotency_key)
        if replayed is not None:
            if (
                original is None
                or replayed.rescheduled_from_appointment_id != original.id
                or replayed.during.lower != start
                or replayed.during.upper != end
            ):
                raise KeyReused(idempotency_key)
            return BookingResult(Outcome.DUPLICATE, replayed)

    if original is None or original.status not in ACTIVE_STATUSES:
        return BookingResult(Outcome.TAKEN)

    if end <= start:
        raise ValueError("an appointment must end after it starts")

    # The destination has to be a time the clinic offers, same as a new
    # booking: a move is a booking with an extra step, and the reviewer found
    # it was the looser of the two paths.
    if override_reason is None:
        await check_eligible(
            session,
            clinic_id=clinic_id,
            doctor_id=original.doctor_id,
            appointment_type_id=original.appointment_type_id,
            # A move keeps the site, so the destination has to be a time the
            # doctor works *there* -- the original's location, not any of them.
            location_id=original.location_id,
            start=start,
            end=end,
            now=now or dt.datetime.now(tz=dt.UTC),
        )

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
    if result.outcome is Outcome.DUPLICATE:
        # A replay is handled above, against the key, before anything is
        # written. Reaching here means the key belongs to a row that is not
        # this move's replacement -- `_same_request` compared the parties and
        # the time and found them equal, but the row does not point back at
        # this original. Returning it would report another operation's
        # appointment as this move's result (QA finding 5).
        raise KeyReused(idempotency_key or "")
    if result.outcome is not Outcome.BOOKED or result.appointment is None:
        return result  # the new time was taken; the original is untouched

    original.status = AppointmentStatus.RESCHEDULED
    original.expires_at = None
    await session.flush()
    # Against the original, not the replacement: the funnel counts "this
    # appointment was moved", and the new row's own `offered`/`confirmed`
    # events tell the rest of the story.
    await record_event(
        session,
        appointment=original,
        kind=AppointmentEventKind.RESCHEDULED,
        actor=actor,
    )
    # QA finding 11: both halves of the move, not only the new row. An auditor
    # reading the log saw an appointment appear and nothing explaining where
    # the patient's earlier time went -- the release is a write to patient data
    # and Rule 8 does not distinguish between the two directions.
    #
    # The event above and these entries answer different questions: the event
    # feeds the funnel, the audit entries record who read and wrote whose data.
    # One does not substitute for the other.
    await record_access(
        session,
        AccessAction.CREATE,
        Access(
            resource="appointments",
            resource_id=str(result.appointment.id),
            patient_id=original.patient_id,
        ),
    )
    await record_access(
        session,
        AccessAction.UPDATE,
        Access(
            resource="appointments",
            resource_id=str(original.id),
            patient_id=original.patient_id,
        ),
    )
    return result
