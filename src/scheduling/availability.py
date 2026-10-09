"""Which times a doctor is actually free, computed from rules.

Slots are never stored. The scope asks for "a real availability model, not
fixed slots", and the reason is drift: a materialised slot table is correct
only until a doctor changes their hours, takes a morning off, or a public
holiday is declared, and nothing in the table knows that happened. Rules
survive all three, so this module derives the bookable times on demand from

  * the doctor's recurring weekly windows (`AvailabilityRule`),
  * dated changes to them (`AvailabilityException`),
  * Colombian public holidays,
  * and the appointments and holds already occupying the doctor's time.

Nothing here touches the database. It takes plain values and returns plain
values, so every branch is testable without a session, and the one place that
decides whether a patient can be offered a time is the one place with no I/O in
it.

TIME AND TIME ZONES
-------------------
A rule says "Tuesdays, 07:00 to 19:00". That is a *clinic-local* wall-clock
window, and it stays 07:00 on the clock whatever happens elsewhere. Slots are
returned as UTC-aware instants, because that is what `appointments.during`
stores and what the exclusion constraint compares. Colombia is UTC-5 with no
daylight saving (`core.timezones.BOGOTA`), so the conversion is exact.

WHAT COUNTS AS BUSY
-------------------
Any appointment in an active status occupies the doctor, and a hold is an
active status -- that is ADR-17: a hold blocks a booking exactly as a booking
does, because they share one exclusion constraint. An expired hold does not,
and this module treats it as free the moment it expires rather than waiting for
the sweep to delete it: the sweep is housekeeping, not the source of truth.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Final, Protocol

from src.core.timezones import BOGOTA

#: How far ahead a caller may ask for slots. A request for "the next two years"
#: is a mistake or an attack, not a scheduling question, and computing it would
#: walk 700 days of rules per doctor.
MAX_HORIZON_DAYS: Final = 120


class _HolidayCalendar(Protocol):
    """What this module needs of a holiday source: a date-in test."""

    def __contains__(self, value: object, /) -> bool: ...


@lru_cache(maxsize=8)
def colombian_holidays(year: int) -> frozenset[dt.date]:
    """Public holidays in Colombia for one year.

    From the `holidays` package rather than a list written here. Ley 51 de 1983
    (the "Emiliani" law) moves most Colombian holidays to the following Monday,
    and which ones move is not derivable -- Reyes Magos moves, Jueves Santo does
    not -- so the rule has to be maintained by somebody who tracks it.

    Cached per year because it is pure and a horizon spans at most two years.
    """
    import holidays

    # `country_holidays` rather than the `Colombia` class: it is the documented
    # entry point, it is typed, and the ISO code will not be renamed.
    return frozenset(holidays.country_holidays("CO", years=year).keys())


@dataclass(frozen=True, slots=True)
class Window:
    """A clinic-local wall-clock window on one date.

    Both the weekly rules and the dated exceptions reduce to this before any
    arithmetic happens, so the overlap logic is written once.
    """

    on_date: dt.date
    start: dt.time
    end: dt.time

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"window must be ordered: {self.start} to {self.end}")

    def as_instants(self) -> tuple[dt.datetime, dt.datetime]:
        """The window as UTC instants, which is what the database compares."""
        start = dt.datetime.combine(self.on_date, self.start, tzinfo=BOGOTA)
        end = dt.datetime.combine(self.on_date, self.end, tzinfo=BOGOTA)
        return start.astimezone(dt.UTC), end.astimezone(dt.UTC)


@dataclass(frozen=True, slots=True)
class Busy:
    """A span the doctor is already committed to, as UTC instants.

    Built from appointments and holds. A hold carries `expires_at`; once that
    has passed the span no longer occupies anything, which `free_windows`
    applies rather than trusting the row to have been swept.
    """

    start: dt.datetime
    end: dt.datetime
    expires_at: dt.datetime | None = None

    def occupies(self, *, now: dt.datetime) -> bool:
        return self.expires_at is None or self.expires_at > now


@dataclass(frozen=True, slots=True)
class Slot:
    """One bookable time, as UTC instants."""

    start: dt.datetime
    end: dt.datetime

    @property
    def local_start(self) -> dt.datetime:
        """The start in clinic-local time, for anything a patient reads."""
        return self.start.astimezone(BOGOTA)


@dataclass(frozen=True, slots=True)
class Rule:
    """A recurring weekly window, as `AvailabilityRule` records one."""

    weekday: int  # 0 = Monday, matching date.weekday()
    start: dt.time
    end: dt.time
    valid_from: dt.date
    valid_until: dt.date | None = None

    def applies_on(self, day: dt.date) -> bool:
        if day.weekday() != self.weekday or day < self.valid_from:
            return False
        return self.valid_until is None or day <= self.valid_until


@dataclass(frozen=True, slots=True)
class Exception_:
    """A dated change to a doctor's availability.

    `unavailable` with no times closes the whole day; with times it closes that
    window. `extra_hours` always carries times and opens one.

    Named with a trailing underscore because `Exception` is a builtin, and
    shadowing it in a module that raises would be a trap for the next reader.
    """

    on_date: dt.date
    kind: str  # "unavailable" | "extra_hours"
    start: dt.time | None = None
    end: dt.time | None = None


def _subtract(window: Window, blocks: Iterable[Window]) -> list[Window]:
    """What remains of `window` once every overlapping block is removed.

    A block may split a window in two -- a lunch break in the middle of a
    morning -- so this returns a list, and the result is ordered.
    """
    remaining = [window]
    for block in blocks:
        if block.on_date != window.on_date:
            continue
        nxt: list[Window] = []
        for part in remaining:
            if block.end <= part.start or block.start >= part.end:
                nxt.append(part)  # no overlap
                continue
            if block.start > part.start:
                nxt.append(Window(part.on_date, part.start, block.start))
            if block.end < part.end:
                nxt.append(Window(part.on_date, block.end, part.end))
        remaining = nxt
    return remaining


def open_windows(
    day: dt.date,
    rules: Sequence[Rule],
    exceptions: Sequence[Exception_] = (),
    *,
    holidays_for: _HolidayCalendar | None = None,
) -> list[Window]:
    """The clinic-local windows a doctor is open on one day.

    A public holiday closes the day outright: the clinic is shut, so no rule
    applies and no exception reopens it. `extra_hours` on a holiday would be a
    clinic saying it works that day, which is a real thing, so that one case is
    honoured -- the holiday closes the recurring rules, not an explicit
    instruction to open.
    """
    calendar = colombian_holidays(day.year) if holidays_for is None else holidays_for

    closed_all_day = any(
        e.on_date == day and e.kind == "unavailable" and e.start is None for e in exceptions
    )
    if closed_all_day:
        windows: list[Window] = []
    elif day in calendar:
        windows = []  # a public holiday closes the recurring schedule
    else:
        windows = [Window(day, r.start, r.end) for r in rules if r.applies_on(day)]

    extra = [
        Window(day, e.start, e.end)
        for e in exceptions
        if e.on_date == day and e.kind == "extra_hours" and e.start and e.end
    ]
    windows.extend(extra)

    partial_blocks = [
        Window(day, e.start, e.end)
        for e in exceptions
        if e.on_date == day and e.kind == "unavailable" and e.start and e.end
    ]
    out: list[Window] = []
    for window in windows:
        out.extend(_subtract(window, partial_blocks))
    return sorted(out, key=lambda w: w.start)


def free_windows(
    windows: Sequence[Window],
    busy: Sequence[Busy],
    *,
    now: dt.datetime,
) -> list[tuple[dt.datetime, dt.datetime]]:
    """The open windows as UTC spans, with occupied time removed.

    Expired holds are skipped: a hold past `expires_at` releases the time
    immediately, rather than when the background sweep next runs.
    """
    taken = [(b.start, b.end) for b in busy if b.occupies(now=now)]
    out: list[tuple[dt.datetime, dt.datetime]] = []
    for window in windows:
        parts = [window.as_instants()]
        for block_start, block_end in taken:
            nxt: list[tuple[dt.datetime, dt.datetime]] = []
            for start, end in parts:
                if block_end <= start or block_start >= end:
                    nxt.append((start, end))
                    continue
                if block_start > start:
                    nxt.append((start, block_start))
                if block_end < end:
                    nxt.append((block_end, end))
            parts = nxt
        out.extend(parts)
    return sorted(out)


def slots_in(
    spans: Sequence[tuple[dt.datetime, dt.datetime]],
    *,
    duration_minutes: int,
    buffer_minutes: int = 0,
    step_minutes: int | None = None,
) -> list[Slot]:
    """Cut free spans into bookable slots.

    `duration_minutes` is the appointment; `buffer_minutes` is the gap the
    clinic wants after it, which must fit inside the span too -- a 30-minute
    appointment with a 10-minute buffer needs 40 free minutes, or the next
    patient starts before the doctor is ready.

    `step_minutes` is how far apart offered start times are, defaulting to the
    appointment length so slots do not overlap each other. A clinic offering
    every 15 minutes for a 30-minute appointment passes 15 and gets overlapping
    candidates, which is correct: only one of them will survive booking.
    """
    if duration_minutes <= 0:
        raise ValueError("duration_minutes must be positive")
    if buffer_minutes < 0:
        raise ValueError("buffer_minutes cannot be negative")

    # `or` would swallow a zero: step_minutes=0 is a caller mistake that would
    # loop forever, and defaulting it to the duration hides that.
    if step_minutes is not None and step_minutes <= 0:
        raise ValueError("step_minutes must be positive")
    step = dt.timedelta(minutes=duration_minutes if step_minutes is None else step_minutes)
    needed = dt.timedelta(minutes=duration_minutes + buffer_minutes)
    appointment = dt.timedelta(minutes=duration_minutes)

    out: list[Slot] = []
    for span_start, span_end in spans:
        start = span_start
        while start + needed <= span_end:
            out.append(Slot(start, start + appointment))
            start += step
    return out


def bookable_slots(
    *,
    start_date: dt.date,
    end_date: dt.date,
    rules: Sequence[Rule],
    exceptions: Sequence[Exception_] = (),
    busy: Sequence[Busy] = (),
    duration_minutes: int,
    buffer_minutes: int = 0,
    step_minutes: int | None = None,
    now: dt.datetime,
    holidays_for: _HolidayCalendar | None = None,
) -> list[Slot]:
    """Every bookable slot for one doctor across a date range.

    The whole module in one call: rules and exceptions give the open windows,
    appointments and holds remove what is taken, and the remainder is cut into
    slots. A slot that starts in the past is dropped -- offering a patient a
    time that has already gone is worse than offering nothing.

    `end_date` is inclusive, because a receptionist asking for "the 1st to the
    7th" means the 7th.
    """
    if end_date < start_date:
        raise ValueError("end_date cannot be before start_date")
    span_days = (end_date - start_date).days + 1
    if span_days > MAX_HORIZON_DAYS:
        raise ValueError(f"horizon of {span_days} days exceeds the {MAX_HORIZON_DAYS}-day maximum")

    out: list[Slot] = []
    for offset in range(span_days):
        day = start_date + dt.timedelta(days=offset)
        windows = open_windows(day, rules, exceptions, holidays_for=holidays_for)
        if not windows:
            continue
        spans = free_windows(windows, busy, now=now)
        out.extend(
            slot
            for slot in slots_in(
                spans,
                duration_minutes=duration_minutes,
                buffer_minutes=buffer_minutes,
                step_minutes=step_minutes,
            )
            if slot.start >= now
        )
    return out
