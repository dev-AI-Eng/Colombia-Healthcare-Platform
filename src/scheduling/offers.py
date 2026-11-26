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

import dataclasses
import datetime as dt
import uuid
from collections.abc import Iterable
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
    OfferRound,
)

#: How many slots to offer at once. Three is enough to feel like a choice and
#: few enough to read on a phone; every one offered is also a slot held away
#: from somebody else, so more is not better.
DEFAULT_OFFER_COUNT: Final = 3

#: How far through a day's free times each successive date's offer falls, when an
#: offer spreads its times: the first date gets the day's earliest time, the next
#: a late one, then an early-middle one, and so on. A fraction of the day's own
#: free times rather than a fixed hour, so it fits a doctor who works only
#: mornings as well as one who works a split day. Fixed, not random: the same
#: search gives the same offer.
TIME_OF_DAY_SPREAD: Final = (0.0, 0.7, 0.35, 0.9, 0.5)


@dataclass(frozen=True, slots=True)
class SlotSearch:
    """What the patient needs. Every field narrows the search."""

    clinic_id: uuid.UUID
    doctor_id: uuid.UUID
    appointment_type_id: uuid.UUID
    start_date: dt.date
    end_date: dt.date
    #: Which site. QA finding 9: without it, a slot derived from one location's
    #: hours was held against a different location's id, so a multi-site clinic
    #: could offer a patient a time at a building the doctor is not in that
    #: day. `None` means every location, which is the right answer for a
    #: single-site clinic and the only behaviour that existed before.
    location_id: uuid.UUID | None = None
    #: Clinic-local times the patient can manage, e.g. "mornings only".
    earliest: dt.time | None = None
    latest: dt.time | None = None
    #: Clinic-local weekdays the patient can manage; empty means any.
    weekdays: frozenset[int] = frozenset()
    #: Clinic-local dates already offered and turned down; never offered again.
    excluded_dates: frozenset[dt.date] = frozenset()
    #: Clinic-local dates to search on; empty means any. An offer repeated
    #: after its holds expired keeps to the dates the patient was shown.
    only_dates: frozenset[dt.date] = frozenset()
    #: At most one slot per clinic-local day, so an offer is a choice between
    #: dates rather than five times on the same morning.
    one_per_day: bool = False
    #: With `one_per_day`, vary where in the day each date's slot falls (see
    #: `TIME_OF_DAY_SPREAD`) instead of always taking the earliest. The client's
    #: condition on the date offer: every date at 8:00 a.m. is not a choice of times.
    spread_times: bool = False


async def _load_rules(
    session: AsyncSession,
    *,
    clinic_id: uuid.UUID,
    doctor_id: uuid.UUID,
    location_id: uuid.UUID | None = None,
) -> list[Rule]:
    stmt = select(AvailabilityRule).where(
        AvailabilityRule.clinic_id == clinic_id, AvailabilityRule.doctor_id == doctor_id
    )
    if location_id is not None:
        stmt = stmt.where(AvailabilityRule.location_id == location_id)
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
    """The spans occupying this doctor across the window, and nothing else.

    Three columns, not the row. The comment below used to say "no patient data
    is read" while `select(Appointment)` loaded every column including
    `patient_id` -- QA finding 13, and the reviewer was right that the claim
    and the query disagreed. Now the query is what the comment says it is, so
    the reasoning that follows actually holds.

    No audit entry: a doctor's calendar is not a disclosure about any
    particular patient, and recording one access per existing appointment every
    time somebody asks for a slot would bury the log that matters. That
    argument depends on no patient identifier leaving the database, which is
    now enforced by the SELECT rather than asserted in a comment.
    """
    window_start = dt.datetime.combine(start_date, dt.time.min, tzinfo=BOGOTA)
    window_end = dt.datetime.combine(end_date + dt.timedelta(days=1), dt.time.min, tzinfo=BOGOTA)
    stmt = select(Appointment.during, Appointment.expires_at).where(
        Appointment.clinic_id == clinic_id,
        Appointment.doctor_id == doctor_id,
        Appointment.status.in_(ACTIVE_STATUSES),
        # Overlap, not containment: an appointment starting before the window
        # and ending inside it still occupies time the patient could be offered.
        Appointment.during.op("&&")(Range(window_start, window_end)),
    )
    out: list[Busy] = []
    for during, expires_at in await session.execute(stmt):
        lower, upper = during.lower, during.upper
        if lower is None or upper is None:
            continue  # a bounded range is enforced by CHECK; skip defensively
        out.append(Busy(lower, upper, expires_at=expires_at))
    return out


def _acceptable(slot: Slot, search: SlotSearch) -> bool:
    """Whether a slot fits what the patient said they can manage.

    Filtering happens on clinic-local time, because "mornings" means mornings
    where the patient lives, not in UTC.
    """
    local = slot.local_start
    if local.date() in search.excluded_dates:
        return False
    if search.only_dates and local.date() not in search.only_dates:
        return False
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

    rules = await _load_rules(
        session,
        clinic_id=search.clinic_id,
        doctor_id=search.doctor_id,
        location_id=search.location_id,
    )
    exceptions = await _load_exceptions(
        session,
        clinic_id=search.clinic_id,
        doctor_id=search.doctor_id,
        start_date=search.start_date,
        end_date=search.end_date,
    )
    # QA finding 8: this used to return [] on an empty rule list, before the
    # exceptions were loaded at all. An `extra_hours` exception is a schedule
    # in its own right -- a locum covering one Saturday, a doctor opening a
    # clinic day that no weekly rule describes -- and it was silently ignored.
    # `open_windows` already returns nothing when neither source offers a
    # window, so the early exit was not needed, only early.
    if not rules and not exceptions:
        return []  # the doctor has no schedule at all; nothing to offer

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
    if search.one_per_day:
        return _one_per_day(slots, search, limit)
    chosen: list[Slot] = []
    for slot in slots:
        if not _acceptable(slot, search):
            continue
        chosen.append(slot)
        if len(chosen) == limit:
            break
    return chosen


def _one_per_day(slots: Iterable[Slot], search: SlotSearch, limit: int) -> list[Slot]:
    """One slot on each of the first `limit` acceptable dates, soonest date first.

    The slots arrive in time order, so a date's slots are complete as soon as the
    next date starts, and the walk stops there rather than reading the whole range.
    """
    by_day: dict[dt.date, list[Slot]] = {}
    for slot in slots:
        if not _acceptable(slot, search):
            continue
        day = slot.local_start.date()
        if day not in by_day and len(by_day) == limit:
            break
        by_day.setdefault(day, []).append(slot)
    return [
        _spread(day_slots, position) if search.spread_times else day_slots[0]
        for position, day_slots in enumerate(by_day.values())
    ]


def _spread(day_slots: list[Slot], position: int) -> Slot:
    """The slot that fraction of the way through this date's free times that
    `TIME_OF_DAY_SPREAD` gives the `position`-th date of the offer."""
    fraction = TIME_OF_DAY_SPREAD[position % len(TIME_OF_DAY_SPREAD)]
    return day_slots[round(fraction * (len(day_slots) - 1))]


async def recent_rounds(
    session: AsyncSession,
    *,
    clinic_id: uuid.UUID,
    appointment_id: uuid.UUID,
    since: dt.datetime,
) -> list[OfferRound]:
    """The rounds of dates offered to move this appointment since `since`,
    oldest first. Each is patient data read, so each is recorded (rule 8)."""
    rows = list(
        (
            await session.scalars(
                select(OfferRound)
                .where(
                    OfferRound.clinic_id == clinic_id,
                    OfferRound.appointment_id == appointment_id,
                    OfferRound.offered_at > since,
                )
                .order_by(OfferRound.offered_at)
            )
        ).all()
    )
    for row in rows:
        await record_access(
            session,
            AccessAction.READ,
            Access(resource="offer_rounds", resource_id=str(row.id), patient_id=row.patient_id),
        )
    return rows


async def record_round(
    session: AsyncSession,
    *,
    clinic_id: uuid.UUID,
    appointment_id: uuid.UUID,
    patient_id: uuid.UUID,
    offered_dates: tuple[dt.date, ...],
    now: dt.datetime,
) -> OfferRound:
    """Remember which dates one round offered, so the next round offers
    others and the conversation knows how many rounds it has made.

    Kept apart from the holds on purpose: a hold is deleted when released and
    swept when it expires, and a patient who answers "ninguna" twenty minutes
    later must still get dates they have not already turned down.
    """
    row = OfferRound(
        clinic_id=clinic_id,
        appointment_id=appointment_id,
        patient_id=patient_id,
        offered_dates=list(offered_dates),
        offered_at=now,
    )
    session.add(row)
    await session.flush()
    await record_access(
        session,
        AccessAction.CREATE,
        Access(resource="offer_rounds", resource_id=str(row.id), patient_id=patient_id),
    )
    return row


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
    # QA finding 9: the hold used `location_id` while the search ignored it, so
    # the two could disagree. The search now carries the location, and this is
    # where they are reconciled -- a caller that passes both must pass the same
    # one, because a slot found at one site cannot be held at another.
    if search.location_id is not None and search.location_id != location_id:
        raise ValueError(
            f"the search is for location {search.location_id} but the hold would be "
            f"placed at {location_id}; a slot found at one site cannot be held at another"
        )
    slots = await find_slots(
        session, dataclasses.replace(search, location_id=location_id), now=now, limit=limit
    )

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
