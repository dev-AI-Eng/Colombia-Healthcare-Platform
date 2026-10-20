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
from collections.abc import Awaitable
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from src.api.dependencies import ClinicScopeDep, PageDep, SessionDep, require_found
from src.api.review.scheduling_schemas import (
    BookIn,
    CancelIn,
    HoldIn,
    RescheduleIn,
)
from src.api.review.schemas import AppointmentOut, EscalationOut, Page, SlotOut, to_page
from src.core.timezones import BOGOTA
from src.scheduling import booking, offers, repository
from src.scheduling.booking import BookingResult, Outcome
from src.scheduling.models import AppointmentType, EscalationStatus

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


async def _attempt(call: Awaitable[BookingResult]) -> BookingResult:
    """Run a booking call, turning an ineligible time into a 422.

    QA finding 3. A time the clinic does not work is not a slot somebody else
    took: a 409 invites the receptionist to retry, and retrying 03:00 on a
    Sunday will fail forever. The engine's Spanish detail is what they read.
    """
    try:
        return await call
    except booking.Ineligible as ineligible:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, ineligible.detail_es
        ) from ineligible
    except booking.KeyReused as reused:
        # 409, not 422: the request is well formed and the conflict is with
        # state the server already holds (QA finding 5). A caller that meant to
        # replay has a bug; one that collided needs a new key.
        raise HTTPException(status.HTTP_409_CONFLICT, reused.detail_es) from reused


async def _span(
    session: SessionDep,
    *,
    appointment_type_id: uuid.UUID,
    start: dt.datetime,
    end: dt.datetime | None,
    duration_minutes: int | None,
) -> dt.datetime:
    """The end of the requested span, from the appointment type when not given.

    QA finding 3's second defect: this used `duration_minutes or 0` and never
    loaded the appointment type, while the docstring claimed the opposite --
    so omitting both fields produced an end equal to the start, and the type's
    configured duration was never consulted at all.

    A supplied `end` is still accepted and still checked, by
    `booking.check_eligible`, which refuses an interval that does not match the
    type's duration. Validating it here as well would put the rule in two
    places and let the engine's callers past it.
    """
    if end is not None:
        return end
    if duration_minutes is not None:
        return start + dt.timedelta(minutes=duration_minutes)

    appointment_type = await session.get(AppointmentType, appointment_type_id)
    if appointment_type is None:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "El tipo de cita no existe. Verifique el servicio seleccionado.",
        )
    return start + dt.timedelta(minutes=appointment_type.duration_minutes)


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
    does not give one, and an end the caller does send is checked against that
    duration by the engine -- so a caller cannot book a 20-minute consultation
    into a 10-minute slot by sending its own end.

    The engine also refuses a time the clinic does not work (QA finding 3),
    which arrives here as a 422 carrying the Spanish explanation: a time
    outside the doctor's hours is not a slot somebody else took, and a 409
    would tell the receptionist to try again at the same hour tomorrow.
    """
    end = await _span(
        session,
        appointment_type_id=body.appointment_type_id,
        start=body.start,
        end=body.end,
        duration_minutes=body.duration_minutes,
    )
    if end <= body.start:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "The appointment must end after it starts: send `end` or `duration_minutes`.",
        )

    result = await _attempt(
        booking.book(
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
    end = await _span(
        session,
        appointment_type_id=body.appointment_type_id,
        start=body.start,
        end=body.end,
        duration_minutes=body.duration_minutes,
    )
    if end <= body.start:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "The hold must end after it starts: send `end` or `duration_minutes`.",
        )

    result = await _attempt(
        booking.hold(
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

    result = await _attempt(
        booking.reschedule(
            session,
            clinic_id=scope.clinic_id,
            appointment_id=appointment_id,
            start=body.start,
            end=end,
            idempotency_key=body.idempotency_key,
        )
    )
    if result.outcome is Outcome.TAKEN:
        raise HTTPException(status.HTTP_409_CONFLICT, TAKEN_DETAIL)
    moved = require_found(result.appointment, "appointment")
    return await _render(session, clinic_id=scope.clinic_id, appointment_id=moved.id)


@router.get(
    "/availability/slots",
    response_model=list[SlotOut],
    summary="The slots a doctor can actually offer",
)
async def find_available_slots(
    session: SessionDep,
    scope: ClinicScopeDep,
    doctor_id: uuid.UUID,
    appointment_type_id: uuid.UUID,
    start_date: dt.date,
    end_date: dt.date,
    earliest: Annotated[
        dt.time | None, Query(description="Earliest hour the patient can attend.")
    ] = None,
    latest: Annotated[
        dt.time | None, Query(description="Latest hour the patient can attend.")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 20,
) -> list[SlotOut]:
    """What this doctor can genuinely offer in a date range.

    Computed from their weekly rules, their dated exceptions, Colombian public
    holidays and their existing diary -- every time it is asked. Nothing is
    stored, which is the whole of M2's availability decision: a stored slot
    table is correct until a doctor changes their hours or a holiday is
    declared, and then it is quietly wrong.

    An empty list is a normal answer, not a failure. It is what the
    conversation graph escalates on, and what a receptionist sees when a week
    is full.
    """
    if end_date < start_date:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "La fecha final no puede ser anterior a la inicial.",
        )

    try:
        slots = await offers.find_slots(
            session,
            offers.SlotSearch(
                clinic_id=scope.clinic_id,
                doctor_id=doctor_id,
                appointment_type_id=appointment_type_id,
                start_date=start_date,
                end_date=end_date,
                earliest=earliest,
                latest=latest,
                weekdays=frozenset(range(7)),
            ),
            now=_now(),
            limit=limit,
        )
    except ValueError:
        # The engine raises for an appointment type it cannot find, which is a
        # programming error from its own callers but an ordinary bad id here.
        # Answering 404 rather than letting it become a 500: the request named
        # something that does not exist, and a screen can say so.
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "No existe ese tipo de cita en esta clínica.",
        ) from None
    return [SlotOut.build(slot) for slot in slots]


@router.get(
    "/escalations",
    response_model=Page[EscalationOut],
    summary="The queue of things a human has to deal with",
)
async def list_escalations(
    session: SessionDep,
    scope: ClinicScopeDep,
    page: PageDep,
    status_filter: Annotated[
        EscalationStatus | None,
        Query(alias="status", description="Only open ones, or only resolved ones."),
    ] = None,
) -> Page[EscalationOut]:
    """Open escalations first, newest first within each status.

    This is what makes M2's fallback criterion visible: when no slot satisfies a
    patient, the request lands here rather than in a silent dead end. The
    attendance sweep writes here too.
    """
    result = await repository.list_escalations(
        session, clinic_id=scope.clinic_id, page=page, status=status_filter
    )
    return to_page(result, [EscalationOut.build(view) for view in result.items])


@router.post(
    "/escalations/{escalation_id}/resolve",
    response_model=EscalationOut,
    summary="Mark one escalation dealt with",
)
async def resolve_escalation(
    session: SessionDep, scope: ClinicScopeDep, escalation_id: uuid.UUID
) -> EscalationOut:
    """Close it, recording when.

    The row is never deleted, so nothing leaves the queue without a record.
    Resolving one that is already resolved answers 404 rather than moving the
    timestamp: the first person dealt with it, and that is the time that matters.
    """
    view = require_found(
        await repository.resolve_escalation(
            session,
            clinic_id=scope.clinic_id,
            escalation_id=escalation_id,
            now=_now(),
        ),
        "open escalation",
    )
    out = EscalationOut.build(view)
    return out
