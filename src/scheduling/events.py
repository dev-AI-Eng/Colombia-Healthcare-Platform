"""Recording what happened to an appointment (ADR-14).

One function, called from inside the transaction that made the change. That is
the whole guarantee: `record_event` issues no commit, so the event and the
status change share a transaction and either both land or neither does. A
booking that commits without its event is impossible rather than unlikely.

**Why a separate table rather than reading status.** An appointment
rescheduled twice and then attended reads as `completed` today, and no report
built on current status can tell you it was ever rescheduled — which is
precisely the number a manager wants. Status answers "where is this now";
events answer "what happened", and only the second supports a funnel.

**Why the writer lives here and not in `booking.py`.** Two reasons. The
conversation graph and the staff routes both change appointments, and a
reporting table written in one of them would be missing the other's rows.
And a funnel whose completeness depends on remembering to add a line is a
funnel that is quietly wrong — so this is the only place that writes the
table, and `booking.py` calls it rather than reimplementing it.

**Actor is coarse on purpose.** `patient`, `staff` or `system`. A patient
cancelling and a receptionist cancelling are different facts about a clinic,
and a funnel that merges them hides the thing a manager is looking for. It is
not a user id: M5 brings accounts, and until then there is no id to record.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Final

from sqlalchemy.ext.asyncio import AsyncSession

from src.scheduling.models import Appointment, AppointmentEvent, AppointmentEventKind

#: Who caused an event. A closed set, because a funnel grouped by free-form
#: text is a funnel nobody can read.
ACTOR_PATIENT: Final = "patient"
ACTOR_STAFF: Final = "staff"
ACTOR_SYSTEM: Final = "system"

VALID_ACTORS: Final[frozenset[str]] = frozenset({ACTOR_PATIENT, ACTOR_STAFF, ACTOR_SYSTEM})


async def record_event(
    session: AsyncSession,
    *,
    appointment: Appointment,
    kind: AppointmentEventKind,
    actor: str,
    reason: str | None = None,
    occurred_at: dt.datetime | None = None,
) -> AppointmentEvent:
    """Record one event, in the caller's transaction.

    Takes the `Appointment` rather than its id, because the event denormalises
    the doctor and the location from it: passing an id would mean re-reading
    the row, and a second read is a second chance to read a different row.

    Flushes but never commits. ADR-14 requires the event and the change to
    share a transaction, and a commit here would break that in the one
    direction that matters -- the event landing while the change rolls back.
    """
    if actor not in VALID_ACTORS:
        # Raised rather than defaulted: an unrecognised actor silently filed as
        # `system` would make the patient-versus-staff split in the funnel
        # wrong, and wrong quietly.
        raise ValueError(f"actor must be one of {sorted(VALID_ACTORS)}, got {actor!r}")

    event = AppointmentEvent(
        clinic_id=appointment.clinic_id,
        appointment_id=appointment.id,
        doctor_id=appointment.doctor_id,
        location_id=appointment.location_id,
        kind=kind,
        actor=actor,
        reason=reason,
    )
    if occurred_at is not None:
        # Only set when the caller knows better than the server clock, which
        # is a backfill or a test. Left to the column default otherwise, so
        # the ordering of two events in one transaction is the database's.
        event.occurred_at = occurred_at
    session.add(event)
    await session.flush()
    return event


async def record_offer(
    session: AsyncSession,
    *,
    clinic_id: uuid.UUID,
    doctor_id: uuid.UUID,
    location_id: uuid.UUID,
    appointment_id: uuid.UUID,
    count: int,
) -> AppointmentEvent:
    """Record that slots were proposed to a patient.

    Separate from `record_event` because an offer has no appointment to read
    the doctor and location from -- it happens *before* anything is booked,
    against the appointment being rescheduled. Rev3 names `offered` in the
    funnel, and it is the denominator of the only metric that measures the bot
    rather than the clinic: slots proposed against slots accepted.

    `count` goes in `reason` because that column is the event's free text and a
    separate integer column would be dead weight on every other kind. Stored
    as a plain number so a report can read it back.
    """
    event = AppointmentEvent(
        clinic_id=clinic_id,
        appointment_id=appointment_id,
        doctor_id=doctor_id,
        location_id=location_id,
        kind=AppointmentEventKind.OFFERED,
        actor=ACTOR_SYSTEM,
        reason=str(count),
    )
    session.add(event)
    await session.flush()
    return event
