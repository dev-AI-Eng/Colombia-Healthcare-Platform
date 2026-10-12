"""The two scheduled sweeps over appointments: reminders due, and attendance.

    Rev2, M2 scope: "Background job scaffolding: daily reminder sweep;
    end-of-day attendance sweep that flags appointments with no recorded
    check-in for staff to confirm as no-show (software cannot observe
    attendance, only record it)."

Both are queries plus a bounded write. Neither sends a message and neither
decides what happened to a patient.

WHY THE REMINDER SWEEP DOES NOT SEND
------------------------------------
Sending needs the channel layer, WhatsApp template approval and the consent
gate, all of which are M4. What M2 owns is the question "which appointments
are due a reminder and have not had one", and the flag that stops a patient
being messaged twice. `reminders_due` answers it; `mark_sent` records the
answer once a dispatcher has acted. Putting a dispatch call here now would be
a code path that cannot run, which is worse than no code path.

The two flags already exist on the appointment (`reminder_48h_sent`,
`reminder_24h_sent`), so the sweep is idempotent against a timer: a run every
few minutes returns the same work until somebody dispatches it, and nothing
afterwards.

WHY THE ATTENDANCE SWEEP DOES NOT SET no_show
---------------------------------------------
Because it does not know. A patient who attended while the receptionist was
busy looks exactly like a patient who never arrived: in both cases nobody
pressed Check-in. Writing `no_show` would be the system inventing a clinical
fact, and a wrongly recorded no-show follows a patient around -- some clinics
charge for it.

So the sweep leaves the status alone and raises an escalation instead, which
is the queue a human already works from. Rev2's own wording is the rule here:
software cannot observe attendance, only record it.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import InstrumentedAttribute

from src.audit.context import AccessAction
from src.audit.service import Access, record_accesses
from src.core.timezones import BOGOTA
from src.scheduling.models import (
    ACTIVE_STATUSES,
    Appointment,
    AppointmentStatus,
    Escalation,
    EscalationReason,
    EscalationStatus,
)

#: Which reminder a due row is for. The two the appointment carries flags for.
ReminderKind = Literal["48h", "24h"]

#: How far ahead each reminder looks. A 48h reminder becomes due once the
#: appointment is inside 48 hours; the 24h one once it is inside 24. Both are
#: upper bounds -- an appointment three days out is due neither.
WINDOWS: dict[ReminderKind, dt.timedelta] = {
    "48h": dt.timedelta(hours=48),
    "24h": dt.timedelta(hours=24),
}

#: Statuses a reminder may be sent for. A hold is an offer nobody accepted, so
#: it is excluded even though it occupies the doctor's time: reminding somebody
#: to attend a slot they never confirmed is confusing, and it expires anyway.
REMINDABLE: tuple[AppointmentStatus, ...] = (
    AppointmentStatus.SCHEDULED,
    AppointmentStatus.CONFIRMED,
)

#: Statuses that mean somebody has already recorded what happened. Anything
#: else, in the past, is what the attendance sweep asks about.
SETTLED: tuple[AppointmentStatus, ...] = (
    AppointmentStatus.CHECKED_IN,
    AppointmentStatus.COMPLETED,
    AppointmentStatus.NO_SHOW,
    AppointmentStatus.CANCELLED,
    AppointmentStatus.RESCHEDULED,
)


@dataclass(frozen=True, slots=True)
class DueReminder:
    """One reminder a dispatcher should send.

    Carries everything a message needs, so M4's dispatcher does not query
    again per row: the appointment to mark afterwards, who it is for, and when
    the appointment starts.
    """

    appointment_id: uuid.UUID
    patient_id: uuid.UUID
    doctor_id: uuid.UUID
    starts_at: dt.datetime
    kind: ReminderKind


def _flag(kind: ReminderKind) -> InstrumentedAttribute[bool]:
    """The column that records this reminder as sent."""
    return Appointment.reminder_48h_sent if kind == "48h" else Appointment.reminder_24h_sent


async def reminders_due(
    session: AsyncSession,
    *,
    clinic_id: uuid.UUID,
    now: dt.datetime,
) -> list[DueReminder]:
    """Appointments inside a reminder window that have not had that reminder.

    The 24h reminder is reported in preference to the 48h one when both are
    unsent: an appointment booked a few hours before it falls inside both
    windows, and sending two messages in one sweep reads as a glitch. The 48h
    flag is left alone rather than backfilled, because a flag means "we sent
    this", and we did not.

    Ordered by start time so a dispatcher works through the soonest first,
    which is the order that matters if it is rate-limited.
    """
    due: list[DueReminder] = []
    seen: set[uuid.UUID] = set()

    # 24h first, so it wins for an appointment inside both windows.
    kinds: tuple[ReminderKind, ...] = ("24h", "48h")
    for kind in kinds:
        horizon = now + WINDOWS[kind]
        rows = (
            await session.scalars(
                select(Appointment)
                .where(
                    Appointment.clinic_id == clinic_id,
                    Appointment.status.in_(REMINDABLE),
                    _flag(kind).is_(False),
                    # Half-open: inside the window, and not already past.
                    func.lower(Appointment.during) > now,
                    func.lower(Appointment.during) <= horizon,
                )
                .order_by(func.lower(Appointment.during))
            )
        ).all()
        for row in rows:
            if row.id in seen:
                continue
            start = row.during.lower
            if start is None:
                # An appointment with no start cannot be reminded about, and
                # the column is NOT NULL with a bounded range, so this is
                # unreachable rather than tolerated. Skipping keeps the sweep
                # running if it ever happens; a crash here would stop every
                # other clinic's reminders too.
                continue
            seen.add(row.id)
            due.append(
                DueReminder(
                    appointment_id=row.id,
                    patient_id=row.patient_id,
                    doctor_id=row.doctor_id,
                    starts_at=start,
                    kind=kind,
                )
            )

    if due:
        # Reading an appointment is reading a patient's data, whatever it is
        # read for. The repository convention is that the data layer records
        # the access, not the caller.
        await record_accesses(
            session,
            AccessAction.READ,
            [
                Access(
                    resource="appointments",
                    resource_id=str(d.appointment_id),
                    patient_id=d.patient_id,
                )
                for d in due
            ],
        )

    due.sort(key=lambda d: d.starts_at)
    return due


async def mark_sent(
    session: AsyncSession,
    *,
    clinic_id: uuid.UUID,
    appointment_id: uuid.UUID,
    kind: ReminderKind,
) -> bool:
    """Record that this reminder went out. Returns False if nothing matched.

    Called by a dispatcher once the message is genuinely away, never before:
    the flag's whole job is to stop a second message, so setting it ahead of a
    send that then fails loses the reminder silently.

    Flushes but does not commit; the caller owns the transaction, so a failed
    dispatch rolls the flag back with it.
    """
    row = await session.scalar(
        select(Appointment).where(
            Appointment.id == appointment_id,
            Appointment.clinic_id == clinic_id,
        )
    )
    if row is None:
        return False

    if kind == "48h":
        row.reminder_48h_sent = True
    else:
        row.reminder_24h_sent = True

    await record_accesses(
        session,
        AccessAction.UPDATE,
        [
            Access(
                resource="appointments",
                resource_id=str(row.id),
                patient_id=row.patient_id,
            )
        ],
    )
    await session.flush()
    return True


async def flag_unrecorded_attendance(
    session: AsyncSession,
    *,
    clinic_id: uuid.UUID,
    now: dt.datetime,
) -> int:
    """Raise an escalation per past appointment with no outcome recorded.

    "Past" means the appointment ended before today began, in Bogotá. A sweep
    that used "ended before now" would flag an appointment that finished
    twenty minutes ago, while the patient is still at the desk and the
    receptionist has not caught up. End of day is the earliest point at which
    an unmarked appointment is genuinely a question.

    Returns the number of escalations raised. Running twice raises none the
    second time: an open escalation for that appointment is itself the record
    that the question has been asked.

    Flushes but does not commit, so the caller can sweep several clinics in
    one transaction.
    """
    today = now.astimezone(BOGOTA).date()
    start_of_today = dt.datetime.combine(today, dt.time.min, tzinfo=BOGOTA)

    # Appointments that finished before today and that nobody has settled.
    candidates = (
        await session.scalars(
            select(Appointment)
            .where(
                Appointment.clinic_id == clinic_id,
                Appointment.status.in_(REMINDABLE),
                func.upper(Appointment.during) <= start_of_today,
            )
            .order_by(func.lower(Appointment.during))
        )
    ).all()
    if not candidates:
        return 0

    # One escalation per appointment, ever. `resolved` counts: a receptionist
    # who answered the question should not be asked it again tomorrow.
    already = set(
        (
            await session.scalars(
                select(Escalation.appointment_id).where(
                    Escalation.clinic_id == clinic_id,
                    Escalation.reason == EscalationReason.ATTENDANCE_UNRECORDED,
                    Escalation.appointment_id.in_([c.id for c in candidates]),
                )
            )
        ).all()
    )

    raised = 0
    flagged_patients: list[tuple[uuid.UUID, uuid.UUID | None]] = []
    for row in candidates:
        if row.id in already:
            continue
        start = row.during.lower
        if start is None:  # unreachable: the column is NOT NULL and bounded
            continue
        local = start.astimezone(BOGOTA)
        session.add(
            Escalation(
                clinic_id=clinic_id,
                patient_id=row.patient_id,
                appointment_id=row.id,
                reason=EscalationReason.ATTENDANCE_UNRECORDED,
                status=EscalationStatus.OPEN,
                detail_es=(
                    f"Nadie registró si el paciente asistió a la cita del "
                    f"{local.strftime('%d/%m/%Y')} a las "
                    f"{local.strftime('%I:%M %p').lstrip('0').lower()}. "
                    f"Confirme si asistió o no asistió."
                ),
                context={
                    "appointment_start": start.isoformat(),
                    "status_at_sweep": str(row.status),
                },
            )
        )
        flagged_patients.append((row.id, row.patient_id))
        raised += 1

    if raised:
        # One entry per escalation actually written, taken from the loop rather
        # than recomputed from `candidates`: a filter that drifted would log an
        # access that never happened, or miss one that did.
        await record_accesses(
            session,
            AccessAction.CREATE,
            [
                Access(
                    resource="escalations",
                    resource_id=str(appointment_id),
                    patient_id=patient_id,
                )
                for appointment_id, patient_id in flagged_patients
            ],
        )
        await session.flush()
    return raised


__all__ = [
    "ACTIVE_STATUSES",
    "REMINDABLE",
    "SETTLED",
    "WINDOWS",
    "DueReminder",
    "ReminderKind",
    "flag_unrecorded_attendance",
    "mark_sent",
    "reminders_due",
]
