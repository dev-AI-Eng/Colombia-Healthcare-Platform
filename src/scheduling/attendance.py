"""Recording what actually happened: check-in, start, end, no-show.

Rev3 is explicit that "the system cannot detect on its own when a consultation
starts or ends", so every time here is written because a person pressed a
button. Nothing infers them, and nothing backfills them.

That constraint is why the punctuality report shows **coverage** beside every
number it derives. A clinic that records check-in reliably and start times
rarely produces a delay figure from a tenth of its appointments, and a mean
over a tenth of the rows is not a mean over the clinic -- it is a mean over the
appointments somebody happened to log. Rev3 asks for the numbers "computed only
where both times were recorded, with recording coverage shown", and that second
clause is the one that keeps the first honest.

**Order is enforced, not assumed.** A consultation cannot start before the
patient checked in, or end before it started. A clinic pressing the buttons out
of order is a data-entry mistake, and silently accepting it would put negative
delays into the report.

**Every transition records an event** (ADR-14) in the same transaction, so the
funnel and the appointment cannot disagree about what happened.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.audit.context import AccessAction
from src.audit.service import Access, record_access
from src.scheduling.events import ACTOR_STAFF, record_event
from src.scheduling.models import (
    ACTIVE_STATUSES,
    Appointment,
    AppointmentEventKind,
    AppointmentStatus,
)


class Step(StrEnum):
    """The four things a person can record about an appointment."""

    CHECK_IN = "check_in"
    START = "start"
    END = "end"
    NO_SHOW = "no_show"


class Refusal(StrEnum):
    """Why a step was refused. A closed set, so a screen can translate it."""

    NOT_FOUND = "not_found"
    #: Already cancelled, already completed, already missed.
    NOT_ACTIVE = "not_active"
    #: Starting before check-in, ending before start, and so on.
    OUT_OF_ORDER = "out_of_order"
    #: The same button pressed twice. Reported rather than raised: a
    #: receptionist double-clicking has done nothing wrong.
    ALREADY_RECORDED = "already_recorded"


@dataclass(frozen=True, slots=True)
class Result:
    ok: bool
    appointment: Appointment | None = None
    refusal: Refusal | None = None


#: Which step writes which column, and which event it records.
_COLUMN: Final[dict[Step, str]] = {
    Step.CHECK_IN: "checked_in_at",
    Step.START: "started_at",
    Step.END: "ended_at",
}

_EVENT: Final[dict[Step, AppointmentEventKind]] = {
    Step.CHECK_IN: AppointmentEventKind.CHECKED_IN,
    Step.START: AppointmentEventKind.STARTED,
    Step.END: AppointmentEventKind.COMPLETED,
    Step.NO_SHOW: AppointmentEventKind.NO_SHOW,
}

#: What must already be recorded before each step. Start needs a check-in; end
#: needs a start. Check-in needs nothing, because it is the first thing that
#: happens.
_REQUIRES: Final[dict[Step, str | None]] = {
    Step.CHECK_IN: None,
    Step.START: "checked_in_at",
    Step.END: "started_at",
    # A no-show requires nothing recorded -- it is the statement that nothing
    # happened. Present rather than absent so the lookup cannot raise: every
    # Step must have an entry, which is what makes adding one a visible
    # change instead of a KeyError at runtime.
    Step.NO_SHOW: None,
}

#: Statuses each step may act on. A cancelled appointment cannot be checked in,
#: and a completed one cannot be marked missed.
_ALLOWED_FROM: Final[dict[Step, frozenset[AppointmentStatus]]] = {
    Step.CHECK_IN: frozenset(ACTIVE_STATUSES),
    Step.START: frozenset({AppointmentStatus.CHECKED_IN}),
    Step.END: frozenset({AppointmentStatus.CHECKED_IN}),
    Step.NO_SHOW: frozenset({AppointmentStatus.SCHEDULED, AppointmentStatus.CONFIRMED}),
}

#: What the appointment becomes. `END` is the only one that leaves the active
#: set, which is why it frees the doctor's time for a later booking.
_BECOMES: Final[dict[Step, AppointmentStatus | None]] = {
    Step.CHECK_IN: AppointmentStatus.CHECKED_IN,
    Step.START: None,  # still checked in; only the time is recorded
    Step.END: AppointmentStatus.COMPLETED,
    Step.NO_SHOW: AppointmentStatus.NO_SHOW,
}


async def record(
    session: AsyncSession,
    *,
    clinic_id: uuid.UUID,
    appointment_id: uuid.UUID,
    step: Step,
    now: dt.datetime,
) -> Result:
    """Record one step, or report why it was refused.

    Refuses rather than raises for every outcome a receptionist can cause: a
    stale screen, a double click, the buttons pressed out of order. Those are
    ordinary and the screen has an answer for each. `now` is passed rather than
    read, as everywhere else in this package, so a test can construct "ended
    before it started".
    """
    found = (
        await session.execute(
            select(Appointment)
            .where(Appointment.clinic_id == clinic_id, Appointment.id == appointment_id)
            .with_for_update()
        )
    ).scalar_one_or_none()

    if found is None:
        return Result(ok=False, refusal=Refusal.NOT_FOUND)
    if found.status not in _ALLOWED_FROM[step]:
        return Result(ok=False, refusal=Refusal.NOT_ACTIVE)

    column = _COLUMN.get(step)
    if column is not None and getattr(found, column) is not None:
        return Result(ok=False, appointment=found, refusal=Refusal.ALREADY_RECORDED)

    required = _REQUIRES[step]
    if required is not None and getattr(found, required) is None:
        # A consultation that starts before the patient checked in would put a
        # negative delay into the punctuality report.
        return Result(ok=False, appointment=found, refusal=Refusal.OUT_OF_ORDER)

    if column is not None:
        setattr(found, column, now)
    becomes = _BECOMES[step]
    if becomes is not None:
        found.status = becomes
        # The CHECK ties an expiry to `hold`, and none of these statuses is a
        # hold, so a stale expiry has to go with it.
        found.expires_at = None

    await session.flush()
    await record_event(
        session, appointment=found, kind=_EVENT[step], actor=ACTOR_STAFF, occurred_at=now
    )
    await record_access(
        session,
        AccessAction.UPDATE,
        Access(resource="appointments", resource_id=str(found.id), patient_id=found.patient_id),
    )
    return Result(ok=True, appointment=found)
