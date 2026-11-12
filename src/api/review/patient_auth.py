"""Who is allowed to change this appointment.

The client asked why anyone could cancel or reschedule any appointment. The
honest answer was that nothing stopped them: the review API's only guard is
`require_synthetic_review`, which checks that synthetic-data mode is on and
404s otherwise. Clinic scope stopped one clinic touching another's rows, and
nothing stopped one *patient* touching another's.

This closes it for the patient-facing direction: a caller acting as a patient
may change their own appointment and nothing else.

WHO THE CALLER IS, AND WHY IT IS NOT WHAT THEY SAY
--------------------------------------------------
The caller proves which patient they are with the phone number the clinic
already holds for them, and the match is made against a `phone_bindings` blind
index -- never against a `patient_id` in the request body. That distinction is
the whole control. A request that named its own patient id would be asking the
attacker which patient they are, and rule 3's reasoning applies to an HTTP body
exactly as it does to a WhatsApp message: what the sender supplies is data, and
deterministic code resolves the row.

This is ADR-18's binding, the same resolution M3's `identify` node performs for
a conversation turn. Deliberately the same, because a patient who cancels by
WhatsApp and a patient who cancels through a screen are the same person with
the same claim, and two different answers to "is this your appointment" is how
one of them becomes wrong.

A shared handset is refused rather than guessed. A household number bound to
two patients cannot prove *which* of them is calling, and cancelling the wrong
family member's appointment is worse than making them call reception -- ADR-18
takes the same position for the conversation path.

WHAT THIS IS NOT
----------------
It is not staff authentication. `docs/ARCHITECTURE.md` §11.5 puts that in M5,
with a session, a second factor for any role that reads patient data, and roles
enforced in dependencies. Staff keep using the review routes unchanged, which
in synthetic-data mode means unauthenticated on a loopback socket.

So this does not make the API safe to expose. It makes the *patient* path
authorised, which is the part a patient can reach by design and the part the
client asked about. The remaining gap is stated plainly rather than implied
closed: until M5, a person who can reach the socket can still act as staff.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Final

import phonenumbers
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select

from src.api.dependencies import ClinicScopeDep, SessionDep
from src.core.crypto import blind_index
from src.identity.models import PhoneBinding
from src.scheduling.models import Appointment

#: The header a patient-facing caller presents. A header rather than a body
#: field so it cannot be confused with the appointment's own data, and so a
#: route that forgets to read it cannot silently accept the body's version.
PATIENT_PHONE_HEADER: Final = "X-Patient-Phone"

#: Colombia. `phonenumbers` carries libphonenumber's metadata, which tracks the
#: 2021 renumbering; a hand-rolled regex does not (CLAUDE.md §3).
REGION: Final = "CO"


class NotYourAppointment(Exception):
    """The caller is not the patient this appointment belongs to."""

    detail_es: Final = "Esa cita no está a su nombre. Si necesita cambiarla, llame a la clínica."


def canonical_phone(raw: str) -> str:
    """The number in E.164, or raise 422.

    Canonicalised before indexing, as §7 requires: the blind index of
    "3001112233" and of "+57 300 111 2233" must be the same value or the lookup
    silently finds nothing, which would read as "no appointment" rather than as
    a malformed request.
    """
    try:
        parsed = phonenumbers.parse(raw, REGION)
    except phonenumbers.NumberParseException as error:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "El número de teléfono no es válido.",
        ) from error
    if not phonenumbers.is_valid_number(parsed):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "El número de teléfono no es válido.",
        )
    return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)


async def caller_patient(
    session: SessionDep,
    scope: ClinicScopeDep,
    phone: Annotated[
        str | None,
        Header(
            alias=PATIENT_PHONE_HEADER,
            description=(
                "The patient's own phone number, as the clinic holds it. "
                "Present it to act as that patient; omit it to act as staff."
            ),
        ),
    ] = None,
) -> uuid.UUID | None:
    """The patient this caller has proved they are, or `None` for a staff caller.

    `None` is not "authorised as everyone": it means no patient claim was made,
    and the route then applies whatever staff rule is in force. Today that is
    `require_synthetic_review` alone, which is the M5 gap named in this module's
    docstring. Returning `None` rather than raising keeps the two paths in one
    dependency, so a route cannot serve the patient path while forgetting the
    staff one.

    An unknown number is a 403 and not a 404: the caller has told us a number
    the clinic does not hold, and saying so lets a patient who mistyped it fix
    the mistake. It reveals nothing -- the answer is the same for every number
    that is not bound in this clinic.
    """
    if phone is None:
        return None

    bindings = (
        await session.scalars(
            select(PhoneBinding.patient_id)
            .where(
                PhoneBinding.clinic_id == scope.clinic_id,
                PhoneBinding.phone_e164_bidx == blind_index(canonical_phone(phone)),
                PhoneBinding.revoked_at.is_(None),
            )
            .order_by(PhoneBinding.created_at)
        )
    ).all()
    patient_ids = tuple(dict.fromkeys(bindings))

    if not patient_ids:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Ese número no está registrado en esta clínica.",
        )
    if len(patient_ids) > 1:
        # A shared handset. ADR-18 refuses rather than guessing, for the same
        # reason the conversation path does: the number cannot say which family
        # member is calling, and cancelling the wrong one is worse than asking
        # them to phone reception.
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Ese número está registrado para varias personas. "
            "Llame a la clínica para confirmar de quién es la cita.",
        )
    return patient_ids[0]


CallerPatientDep = Annotated[uuid.UUID | None, Depends(caller_patient)]


async def require_owns(
    session: SessionDep,
    *,
    clinic_id: uuid.UUID,
    appointment_id: uuid.UUID,
    caller_patient_id: uuid.UUID | None,
) -> None:
    """Raise unless this caller may change this appointment.

    A staff caller (`caller_patient_id is None`) passes, because staff
    legitimately act for any patient and their own authorisation is M5's.

    A patient caller passes only for their own appointment. The check reads the
    row's `patient_id` from the database rather than trusting anything in the
    request, and it is a 403 rather than a 404: the appointment exists and the
    caller may not touch it. A 404 would be the more secretive answer, but it
    would also tell a patient whose own appointment was just moved that it no
    longer exists, which sends them to the wrong conclusion.
    """
    if caller_patient_id is None:
        return

    owner = await session.scalar(
        select(Appointment.patient_id).where(
            Appointment.clinic_id == clinic_id,
            Appointment.id == appointment_id,
        )
    )
    if owner is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "appointment not found")
    if owner != caller_patient_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, NotYourAppointment.detail_es)
