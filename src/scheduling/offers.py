"""Finding slots to offer a patient, and what to do when there are none.

This is the surface M3's conversation graph calls: `find_slots` is its
`find_slots` tool, `offer` is `hold_slots`, and `escalate` is what happens
when neither can help. Keeping it here rather than in the graph means the
scheduling rules are testable without a conversation, and the graph cannot
reach around them.

Two things it exists to get right.

**A slot is only offerable if it can still be booked.** `availability`
computes what the rules allow; this loads what is actually taken and asks
`availability` to subtract it. Offering a time that the constraint will then
refuse is the failure mode patients notice: they pick a slot and are told it
has gone.

**No slot is not an error.** A patient who needs an appointment and cannot be
given one is the case the scope asks about, and the honest answer is a person.
`escalate` writes a row staff work from rather than sending a message into the
void -- a fallback that only replies is a dead end, because nobody is told the
patient is still waiting.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass
from typing import Final

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import Range
from sqlalchemy.ext.asyncio import AsyncSession

from src.audit.context import AccessAction
from src.audit.service import Access, record_access
from src.core.timezones import BOGOTA
from src.scheduling import booking
from src.scheduling.availability import (
    Busy,
    Exception_,
    Rule,
    Slot,
    bookable_slots,
)
from src.scheduling.models import (
    ACTIVE_STATUSES,
    Appointment,
    AppointmentType,
    AvailabilityException,
    AvailabilityRule,
    Escalation,
    EscalationReason,
)

#: How many slots to offer at once. Three is enough to feel like a choice and
#: few enough to read on a phone; every one offered is also a slot held away
#: from somebody else, so more is not better.
DEFAULT_OFFER_COUNT: Final = 3


@dataclass(frozen=True, slots=True)
class SlotSearch:
    """What the patient needs. Every field narrows the search."""

    clinic_id: uuid.UUID
    doctor_id: uuid.UUID
    appointment_type_id: uuid.UUID
    start_date: dt.date
    end_date: dt.date
    #: Clinic-local times the patient can manage, e.g. "mornings only".
    earliest: dt.time | None = None
    latest: dt.time | None = None
    #: Clinic-local weekdays the patient can manage; empty means any.
    weekdays: frozenset[int] = frozenset()


async def _load_rules(
    session: AsyncSession, *, clinic_id: uuid.UUID, doctor_id: uuid.UUID
) -> list[Rule]:
    stmt = select(AvailabilityRule).where(
        AvailabilityRule.clinic_id == clinic_id, AvailabilityRule.doctor_id == doctor_id
    )
    return [
        Rule(
            weekday=r.weekday,
            start=r.start_time,
            end=r.end_time,
            valid_from=r.valid_from,
            valid_until=r.valid_until,
        )
        for r in (await session.execute(stmt)).scalars()
    ]


async def _load_exceptions(
    session: AsyncSession,
    *,
    clinic_id: uuid.UUID,
    doctor_id: uuid.UUID,
    start_date: dt.date,
    end_date: dt.date,
) -> list[Exception_]:
    stmt = select(AvailabilityException).where(
        AvailabilityException.clinic_id == clinic_id,
        AvailabilityException.doctor_id == doctor_id,
        AvailabilityException.on_date >= start_date,
        AvailabilityException.on_date <= end_date,
    )
    return [
        Exception_(on_date=e.on_date, kind=str(e.kind), start=e.start_time, end=e.end_time)
        for e in (await session.execute(stmt)).scalars()
    ]


async def _load_busy(
    session: AsyncSession,
    *,
    clinic_id: uuid.UUID,
    doctor_id: uuid.UUID,
    start_date: dt.date,
    end_date: dt.date,
) -> list[Busy]:
    """Appointments and holds occupying this doctor across the window.

    No patient data is read -- only the spans -- so this writes no audit entry.
    A doctor's calendar is not a disclosure about any particular patient, and
    recording one access per existing appointment every time somebody asks for
    a slot would bury the log that matters.
    """
    window_start = dt.datetime.combine(start_date, dt.time.min, tzinfo=BOGOTA)
    window_end = dt.datetime.combine(end_date + dt.timedelta(days=1), dt.time.min, tzinfo=BOGOTA)
    stmt = select(Appointment).where(
        Appointment.clinic_id == clinic_id,
        Appointment.doctor_id == doctor_id,
        Appointment.status.in_(ACTIVE_STATUSES),
        # Overlap, not containment: an appointment starting before the window
        # and ending inside it still occupies time the patient could be offered.
        Appointment.during.op("&&")(Range(window_start, window_end)),
    )
    out: list[Busy] = []
    for row in (await session.execute(stmt)).scalars():
        lower, upper = row.during.lower, row.during.upper
        if lower is None or upper is None:
            continue  # a bounded range is enforced by CHECK; skip defensively
        out.append(Busy(lower, upper, expires_at=row.expires_at))
    return out


def _acceptable(slot: Slot, search: SlotSearch) -> bool:
    """Whether a slot fits what the patient said they can manage.

    Filtering happens on clinic-local time, because "mornings" means mornings
    where the patient lives, not in UTC.
    """
    local = slot.local_start
    if search.weekdays and local.weekday() not in search.weekdays:
        return False
    if search.earliest is not None and local.time() < search.earliest:
        return False
    return not (search.latest is not None and local.time() >= search.latest)


async def find_slots(
    session: AsyncSession,
    search: SlotSearch,
    *,
    now: dt.datetime,
    limit: int = DEFAULT_OFFER_COUNT,
) -> list[Slot]:
    """The next few slots this patient could actually take.

    Returns an empty list when nothing fits, which is a normal outcome and the
    caller's cue to escalate. It is never an exception: "no appointment is
    available" is an answer, not a failure.
    """
    appointment_type = await session.get(AppointmentType, search.appointment_type_id)
    if appointment_type is None:
        raise ValueError(f"no appointment type {search.appointment_type_id}")

    rules = await _load_rules(session, clinic_id=search.clinic_id, doctor_id=search.doctor_id)
    if not rules:
        return []  # the doctor has no schedule; nothing to offer

    exceptions = await _load_exceptions(
        session,
        clinic_id=search.clinic_id,
        doctor_id=search.doctor_id,
        start_date=search.start_date,
        end_date=search.end_date,
    )
    busy = await _load_busy(
        session,
        clinic_id=search.clinic_id,
        doctor_id=search.doctor_id,
        start_date=search.start_date,
        end_date=search.end_date,
    )

    slots = bookable_slots(
        start_date=search.start_date,
        end_date=search.end_date,
        rules=rules,
        exceptions=exceptions,
        busy=busy,
        duration_minutes=appointment_type.duration_minutes,
        buffer_minutes=appointment_type.buffer_minutes,
        now=now,
    )
    return [slot for slot in slots if _acceptable(slot, search)][:limit]


async def escalate(
    session: AsyncSession,
    *,
    clinic_id: uuid.UUID,
    reason: EscalationReason,
    detail_es: str,
    patient_id: uuid.UUID | None = None,
    appointment_id: uuid.UUID | None = None,
    context: dict[str, object] | None = None,
) -> Escalation:
    """Put something in front of a human, and record that it happened.

    Audited as a write against the patient when there is one: an escalation
    names them and what they wanted, so it is patient data and the log has to
    say it was created.
    """
    row = Escalation(
        clinic_id=clinic_id,
        patient_id=patient_id,
        appointment_id=appointment_id,
        reason=reason,
        detail_es=detail_es,
        context=dict(context or {}),
    )
    session.add(row)
    await session.flush()
    if patient_id is not None:
        await record_access(
            session,
            AccessAction.CREATE,
            Access(resource="escalations", resource_id=str(row.id), patient_id=patient_id),
        )
    return row


@dataclass(frozen=True, slots=True)
class OfferResult:
    """What came of trying to give the patient a time."""

    held: list[booking.BookingResult]
    escalation: Escalation | None = None

    @property
    def offered_any(self) -> bool:
        return any(result.ok for result in self.held)


async def offer(
    session: AsyncSession,
    search: SlotSearch,
    *,
    patient_id: uuid.UUID,
    location_id: uuid.UUID,
    now: dt.datetime,
    limit: int = DEFAULT_OFFER_COUNT,
    hold_minutes: int = booking.DEFAULT_HOLD_MINUTES,
) -> OfferResult:
    """Find slots, hold them, and escalate if none could be held.

    Every offered slot is held, because an offer the patient cannot accept is
    worse than no offer (ADR-17). Holding can still lose a race -- another
    conversation may take a slot between the search and the hold -- and a slot
    lost that way is simply dropped from the offer; the remaining ones stand.

    Escalation happens when *nothing* could be held, whether because the search
    found nothing or because every candidate was taken on the way. From the
    patient's side those are the same situation, and both need a person.
    """
    slots = await find_slots(session, search, now=now, limit=limit)

    held: list[booking.BookingResult] = []
    for slot in slots:
        result = await booking.hold(
            session,
            clinic_id=search.clinic_id,
            patient_id=patient_id,
            doctor_id=search.doctor_id,
            location_id=location_id,
            appointment_type_id=search.appointment_type_id,
            start=slot.start,
            end=slot.end,
            now=now,
            minutes=hold_minutes,
        )
        if result.ok:
            held.append(result)

    if held:
        return OfferResult(held)

    escalation = await escalate(
        session,
        clinic_id=search.clinic_id,
        patient_id=patient_id,
        reason=EscalationReason.NO_ACCEPTABLE_SLOT,
        detail_es=(
            "No hay citas disponibles que el paciente pueda tomar entre el "
            f"{search.start_date:%d/%m/%Y} y el {search.end_date:%d/%m/%Y}. "
            "Contacte al paciente para acordar una hora."
        ),
        context={
            "doctor_id": str(search.doctor_id),
            "appointment_type_id": str(search.appointment_type_id),
            "start_date": search.start_date.isoformat(),
            "end_date": search.end_date.isoformat(),
            "earliest": search.earliest.isoformat() if search.earliest else None,
            "latest": search.latest.isoformat() if search.latest else None,
            "weekdays": sorted(search.weekdays),
        },
    )
    return OfferResult([], escalation)
