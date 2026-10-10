"""Write routes over the scheduling engine: book, hold, confirm, cancel, reschedule.

    Rev2, M2 deliverable: "Scheduling service with booking/cancel/reschedule
    APIs."

The engine itself is `src.scheduling.booking` and `src.scheduling.offers`, and
M3's conversation graph calls those directly in-process rather than over HTTP.
These routes exist so the deliverable is callable by a person -- from Swagger,
from a dashboard, or from a staff screen -- rather than only by Python.

They add no scheduling logic. Every one is a thin shell: validate the body,
call the engine function, map its outcome to a status code. Double-booking is
still refused by the database, holds still expire on their own, and audit
entries are still written by the repositories. A second caller of the same
engine function gets the same guarantees without re-implementing them here.

WHY THEY LIVE UNDER /review
---------------------------
Because that is where the access gate already is. `require_synthetic_review`
mounts this router only while synthetic-data mode is on, and `app.py` requires
a loopback Host for the whole surface. Authentication and the clinic-staff role
arrive in M5; until then an authenticated staff API does not exist to attach
these to, and mounting them outside the gate would make them the one
unauthenticated write path into patient data. The M11 staff API replaces this
prefix wholesale, per `review/__init__.py`.

STATUS CODES
------------
409 is the interesting one. `Outcome.TAKEN` means another patient committed the
same time first: the request was well formed and the answer is "no", which is a
conflict rather than a client error. `Outcome.DUPLICATE` means an idempotency
key has been seen before, so the original appointment is returned with 200 --
a retried request must not create a second booking.
"""

from __future__ import annotations

import datetime as dt
import uuid

from fastapi import APIRouter, HTTPException, status

from src.api.dependencies import ClinicScopeDep, SessionDep, require_found
from src.api.review.scheduling_schemas import (
    BookIn,
    CancelIn,
    HoldIn,
    RescheduleIn,
)
from src.api.review.schemas import AppointmentOut
from src.core.timezones import BOGOTA
from src.scheduling import booking, repository
from src.scheduling.booking import BookingResult, Outcome

router = APIRouter()

#: What a caller is told when the slot went to somebody else. Spanish, because
#: a receptionist reads it; the conversation graph maps the status code, not
#: this text.
TAKEN_DETAIL = "Ese horario ya está ocupado. Elija otro."

#: A hold that has already lapsed cannot be confirmed: its slot is free again
#: and may belong to another patient by now.
EXPIRED_DETAIL = "La reserva ya venció. Vuelva a buscar un horario."


def _now() -> dt.datetime:
    """The clock the engine is told about, in Colombia's zone.

    Taken here rather than inside the engine so a test can pass its own moment
    to the engine directly, and so every call in one request sees one instant.
    """
    return dt.datetime.now(BOGOTA)


async def _render(
    session: SessionDep, *, clinic_id: uuid.UUID, appointment_id: uuid.UUID
) -> AppointmentOut:
    """Read the written appointment back, in the shape every read route returns.

    The read goes through the repository rather than serialising the engine's
    own object, because the repository joins the doctor, location and type
    names the response carries and records the read in the audit log. Building
    the response here by hand would skip both.

    **It must happen before the transaction commits**, which is why nothing
    here commits: `get_session` does that when the endpoint returns.
    `apply_clinic_scope` sets a transaction-local setting, so a commit clears
    it, and a read afterwards sees no rows at all -- row-level security working
    exactly as intended, turning a successful write into a 404. An explicit
    commit here was removed for the same reason: it put the clearing of the
    scope in the middle of the request rather than at the end of it.
    """
    view = require_found(
        await repository.get_appointment(
            session, clinic_id=clinic_id, appointment_id=appointment_id
        ),
        "appointment",
    )
    return AppointmentOut.build(view)


async def _respond(
    session: SessionDep, *, clinic_id: uuid.UUID, result: BookingResult
) -> AppointmentOut:
    """Turn an engine outcome into a response or an HTTP error.

    `TAKEN` has no appointment to return, so it is the only outcome that
    raises. `BOOKED` and `DUPLICATE` both carry one, and both succeed: a retry
    that found its own earlier booking did not write anything the second time,
    which is the point of an idempotency key.
    """
    if result.outcome is Outcome.TAKEN:
        raise HTTPException(status.HTTP_409_CONFLICT, TAKEN_DETAIL)
    appointment = require_found(result.appointment, "appointment")
    return await _render(session, clinic_id=clinic_id, appointment_id=appointment.id)


@router.post(
    "/appointments",
    response_model=AppointmentOut,
    status_code=status.HTTP_201_CREATED,
    summary="Book an appointment",
    responses={409: {"description": "The slot was taken by another booking."}},
)
async def book_appointment(
    session: SessionDep, scope: ClinicScopeDep, body: BookIn
) -> AppointmentOut:
    """Write one appointment, inside the transaction the constraint guards.

    The end time is computed from the appointment type's duration when the body
    does not give one, so a caller cannot book a 20-minute consultation into a
    10-minute slot by sending its own end.
    """
    end = body.end or body.start + dt.timedelta(minutes=body.duration_minutes or 0)
    if end <= body.start:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "The appointment must end after it starts: send `end` or `duration_minutes`.",
        )

    result = await booking.book(
        session,
        clinic_id=scope.clinic_id,
        patient_id=body.patient_id,
        doctor_id=body.doctor_id,
        location_id=body.location_id,
        appointment_type_id=body.appointment_type_id,
        start=body.start,
        end=end,
        idempotency_key=body.idempotency_key,
    )
    return await _respond(session, clinic_id=scope.clinic_id, result=result)


@router.post(
    "/appointments/holds",
    response_model=AppointmentOut,
    status_code=status.HTTP_201_CREATED,
    summary="Hold a slot while a patient decides",
    responses={409: {"description": "The slot was taken by another booking."}},
)
async def hold_slot(session: SessionDep, scope: ClinicScopeDep, body: HoldIn) -> AppointmentOut:
    """Reserve a time for a few minutes.

    A hold sits in the same overlap constraint as a booking, so while it lives
    the time is genuinely the patient's. It releases itself when `expires_at`
    passes whether or not the sweeper has run.
    """
    end = body.end or body.start + dt.timedelta(minutes=body.duration_minutes or 0)
    if end <= body.start:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "The hold must end after it starts: send `end` or `duration_minutes`.",
        )

    result = await booking.hold(
        session,
        clinic_id=scope.clinic_id,
        patient_id=body.patient_id,
        doctor_id=body.doctor_id,
        location_id=body.location_id,
        appointment_type_id=body.appointment_type_id,
        start=body.start,
        end=end,
        now=_now(),
        minutes=body.minutes,
        idempotency_key=body.idempotency_key,
    )
    return await _respond(session, clinic_id=scope.clinic_id, result=result)


@router.post(
    "/appointments/{appointment_id}/confirm",
    response_model=AppointmentOut,
    summary="Turn a live hold into a booking",
    responses={409: {"description": "The hold had already expired."}},
)
async def confirm(
    session: SessionDep, scope: ClinicScopeDep, appointment_id: uuid.UUID
) -> AppointmentOut:
    """Promote a hold to `scheduled`, if it is still live.

    An expired hold answers 409 rather than being revived: its slot was free in
    the meantime and may already belong to somebody else.
    """
    result = await booking.confirm_hold(
        session,
        clinic_id=scope.clinic_id,
        appointment_id=appointment_id,
        now=_now(),
    )
    if result.outcome is Outcome.TAKEN:
        raise HTTPException(status.HTTP_409_CONFLICT, EXPIRED_DETAIL)
    require_found(result.appointment, "hold")
    return await _render(session, clinic_id=scope.clinic_id, appointment_id=appointment_id)


@router.post(
    "/appointments/{appointment_id}/cancel",
    response_model=AppointmentOut,
    summary="Cancel an appointment",
)
async def cancel_appointment(
    session: SessionDep,
    scope: ClinicScopeDep,
    appointment_id: uuid.UUID,
    body: CancelIn,
) -> AppointmentOut:
    """Mark it cancelled, which frees the doctor's time.

    The row is kept rather than deleted: a cancellation is part of the
    patient's history, and M4's reporting counts cancellations by reason.
    """
    changed = await booking.cancel(
        session,
        clinic_id=scope.clinic_id,
        appointment_id=appointment_id,
        reason=body.reason,
    )
    if not changed:
        # Either no such appointment in this clinic, or it is already in a
        # state that cannot be cancelled. Both are "nothing to cancel here".
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "No active appointment with that id in this clinic.",
        )
    return await _render(session, clinic_id=scope.clinic_id, appointment_id=appointment_id)


@router.post(
    "/appointments/{appointment_id}/reschedule",
    response_model=AppointmentOut,
    status_code=status.HTTP_201_CREATED,
    summary="Move an appointment to a new time",
    responses={409: {"description": "The new slot was taken by another booking."}},
)
async def reschedule_appointment(
    session: SessionDep,
    scope: ClinicScopeDep,
    appointment_id: uuid.UUID,
    body: RescheduleIn,
) -> AppointmentOut:
    """Cancel the old row and write a new one pointing back at it.

    Two rows rather than an edited one, because that is how the clinic's own
    exports model it and because the history is what M4 reports on. The
    response is the new appointment; 201 says a row was created.

    If the new time is taken the old appointment is left exactly as it was --
    the engine does the cancel and the insert in one transaction, so a refused
    insert rolls the cancel back with it. A patient never loses their existing
    appointment to a failed move.
    """
    end = body.end or body.start + dt.timedelta(minutes=body.duration_minutes or 0)
    if end <= body.start:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "The appointment must end after it starts: send `end` or `duration_minutes`.",
        )

    result = await booking.reschedule(
        session,
        clinic_id=scope.clinic_id,
        appointment_id=appointment_id,
        start=body.start,
        end=end,
        idempotency_key=body.idempotency_key,
    )
    if result.outcome is Outcome.TAKEN:
        raise HTTPException(status.HTTP_409_CONFLICT, TAKEN_DETAIL)
    moved = require_found(result.appointment, "appointment")
    return await _render(session, clinic_id=scope.clinic_id, appointment_id=moved.id)
