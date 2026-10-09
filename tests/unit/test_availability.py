"""The availability engine: which times a doctor is actually free.

No database. The engine takes plain values and returns plain values precisely
so these can be exhaustive, because every branch here decides whether a patient
is offered a time that does not exist or refused one that does.

Dates are concrete rather than relative to today, so a test cannot start
passing or failing because of when it runs. 2026-11-03 is a Tuesday and
2026-11-04 a Wednesday; both are chosen for being ordinary days with no
Colombian holiday near them.
"""

from __future__ import annotations

import datetime as dt

import pytest

from src.core.timezones import BOGOTA
from src.scheduling.availability import (
    MAX_HORIZON_DAYS,
    Busy,
    Exception_,
    Rule,
    Window,
    bookable_slots,
    colombian_holidays,
    free_windows,
    open_windows,
    slots_in,
)

TUESDAY = dt.date(2026, 11, 3)
WEDNESDAY = dt.date(2026, 11, 4)
#: A Monday that is a public holiday: Día de San José, shifted there by Emiliani.
SAN_JOSE = dt.date(2026, 3, 23)

#: A doctor who works Tuesday mornings, 08:00 to 12:00 clinic time.
MORNINGS = Rule(weekday=1, start=dt.time(8), end=dt.time(12), valid_from=dt.date(2026, 1, 1))


def _utc(day: dt.date, hour: int, minute: int = 0) -> dt.datetime:
    """A clinic-local wall-clock time, as the UTC instant the database stores."""
    return dt.datetime.combine(day, dt.time(hour, minute), tzinfo=BOGOTA).astimezone(dt.UTC)


#: Well before any slot in these tests, so "in the past" never interferes.
EARLY = _utc(dt.date(2026, 1, 1), 0)


# ------------------------------------------------------------------ holidays
def test_a_public_holiday_closes_the_day() -> None:
    """A clinic is shut, so the recurring rules do not apply.

    23 March 2026 is the Monday on which Día de San José is observed; a doctor
    who works Mondays is not available on it.
    """
    assert SAN_JOSE in colombian_holidays(2026)
    mondays = Rule(weekday=0, start=dt.time(8), end=dt.time(12), valid_from=dt.date(2026, 1, 1))
    assert open_windows(SAN_JOSE, [mondays]) == []


def test_the_holiday_list_follows_the_emiliani_shift() -> None:
    """Ley 51 de 1983 moves most Colombian holidays to the following Monday.

    Which ones move is not derivable -- Reyes Magos moves, Jueves Santo does
    not -- so this asserts the library tracks it rather than that we do. In
    2026 Reyes Magos falls on Tuesday 6 January and is observed on Monday 12.
    """
    year = colombian_holidays(2026)
    assert dt.date(2026, 1, 12) in year, "Reyes Magos should be observed on the Monday"
    assert dt.date(2026, 1, 6) not in year, "and not on the date it nominally falls"
    assert dt.date(2026, 4, 2) in year, "Jueves Santo does not shift"


def test_extra_hours_can_open_a_holiday() -> None:
    """A clinic saying "we work that day" is an instruction, not an accident.

    The holiday closes the recurring schedule; it does not override a dated
    decision to open.
    """
    mondays = Rule(weekday=0, start=dt.time(8), end=dt.time(12), valid_from=dt.date(2026, 1, 1))
    opened = Exception_(SAN_JOSE, "extra_hours", dt.time(9), dt.time(11))
    assert open_windows(SAN_JOSE, [mondays], [opened]) == [
        Window(SAN_JOSE, dt.time(9), dt.time(11))
    ]


# -------------------------------------------------------------- rules in force
def test_a_rule_applies_only_on_its_weekday() -> None:
    assert open_windows(TUESDAY, [MORNINGS]) == [Window(TUESDAY, dt.time(8), dt.time(12))]
    assert open_windows(WEDNESDAY, [MORNINGS]) == []


def test_a_rule_does_not_apply_before_it_starts_or_after_it_ends() -> None:
    """`valid_from` and `valid_until` bound a rule, and both are inclusive."""
    bounded = Rule(
        weekday=1,
        start=dt.time(8),
        end=dt.time(12),
        valid_from=dt.date(2026, 11, 3),
        valid_until=dt.date(2026, 11, 3),
    )
    assert open_windows(dt.date(2026, 11, 3), [bounded]), "valid_from is inclusive"
    assert open_windows(dt.date(2026, 10, 27), [bounded]) == [], "the Tuesday before"
    assert open_windows(dt.date(2026, 11, 10), [bounded]) == [], "the Tuesday after"


# -------------------------------------------------------------- exceptions
def test_a_whole_day_closure_removes_every_window() -> None:
    off = Exception_(TUESDAY, "unavailable")
    assert open_windows(TUESDAY, [MORNINGS], [off]) == []


def test_a_partial_closure_splits_the_window() -> None:
    """A doctor out from 09:00 to 10:00 leaves two windows, not one."""
    lunch = Exception_(TUESDAY, "unavailable", dt.time(9), dt.time(10))
    assert open_windows(TUESDAY, [MORNINGS], [lunch]) == [
        Window(TUESDAY, dt.time(8), dt.time(9)),
        Window(TUESDAY, dt.time(10), dt.time(12)),
    ]


def test_an_exception_on_another_day_is_ignored() -> None:
    elsewhere = Exception_(WEDNESDAY, "unavailable")
    assert open_windows(TUESDAY, [MORNINGS], [elsewhere]) == [
        Window(TUESDAY, dt.time(8), dt.time(12))
    ]


# ------------------------------------------------------------- busy time
def test_an_appointment_removes_its_own_time_and_no_more() -> None:
    window = Window(TUESDAY, dt.time(8), dt.time(12))
    booked = Busy(_utc(TUESDAY, 9), _utc(TUESDAY, 10))
    assert free_windows([window], [booked], now=EARLY) == [
        (_utc(TUESDAY, 8), _utc(TUESDAY, 9)),
        (_utc(TUESDAY, 10), _utc(TUESDAY, 12)),
    ]


def test_a_live_hold_occupies_the_time_exactly_as_a_booking_does() -> None:
    """ADR-17: a hold and a booking share one exclusion constraint.

    If a hold did not block here, the engine would offer a slot the database
    then refuses -- the patient picks a time and is told it is gone.
    """
    window = Window(TUESDAY, dt.time(8), dt.time(12))
    held = Busy(_utc(TUESDAY, 9), _utc(TUESDAY, 10), expires_at=_utc(TUESDAY, 9, 15))
    free = free_windows([window], [held], now=_utc(TUESDAY, 9, 5))
    assert free == [
        (_utc(TUESDAY, 8), _utc(TUESDAY, 9)),
        (_utc(TUESDAY, 10), _utc(TUESDAY, 12)),
    ]


def test_an_expired_hold_releases_its_time_without_waiting_for_the_sweep() -> None:
    """The sweep is housekeeping; expiry is what decides.

    A hold whose `expires_at` has passed must free the slot immediately, or
    every offer is unavailable for as long as the sweep interval.
    """
    window = Window(TUESDAY, dt.time(8), dt.time(12))
    stale = Busy(_utc(TUESDAY, 9), _utc(TUESDAY, 10), expires_at=_utc(TUESDAY, 9, 15))
    free = free_windows([window], [stale], now=_utc(TUESDAY, 9, 20))
    assert free == [(_utc(TUESDAY, 8), _utc(TUESDAY, 12))], "the whole morning is free again"


# ------------------------------------------------------------------- slots
def test_slots_fill_a_span_without_overlapping() -> None:
    span = [(_utc(TUESDAY, 8), _utc(TUESDAY, 9))]
    slots = slots_in(span, duration_minutes=30)
    assert [s.start for s in slots] == [_utc(TUESDAY, 8), _utc(TUESDAY, 8, 30)]
    assert slots[0].end == slots[1].start, "back to back, not overlapping"


def test_a_buffer_must_fit_inside_the_span_too() -> None:
    """A 30-minute appointment with a 10-minute buffer needs 40 free minutes.

    Counting only the appointment would schedule the next patient before the
    doctor is ready, which is the whole reason the clinic set a buffer.
    """
    span = [(_utc(TUESDAY, 8), _utc(TUESDAY, 9))]
    slots = slots_in(span, duration_minutes=30, buffer_minutes=10)
    assert [s.start for s in slots] == [_utc(TUESDAY, 8)], "08:30 + 40 min would run past 09:00"
    assert slots[0].end == _utc(TUESDAY, 8, 30), "the buffer is not part of the appointment"


def test_a_span_shorter_than_one_appointment_yields_nothing() -> None:
    span = [(_utc(TUESDAY, 8), _utc(TUESDAY, 8, 20))]
    assert slots_in(span, duration_minutes=30) == []


def test_a_step_shorter_than_the_appointment_offers_overlapping_candidates() -> None:
    """Only one survives booking; offering every 15 minutes is a clinic's choice."""
    span = [(_utc(TUESDAY, 8), _utc(TUESDAY, 9))]
    starts = [s.start for s in slots_in(span, duration_minutes=30, step_minutes=15)]
    assert starts == [_utc(TUESDAY, 8), _utc(TUESDAY, 8, 15), _utc(TUESDAY, 8, 30)]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"duration_minutes": 0}, "duration_minutes must be positive"),
        ({"duration_minutes": 30, "buffer_minutes": -1}, "buffer_minutes cannot be negative"),
        ({"duration_minutes": 30, "step_minutes": 0}, "step_minutes must be positive"),
    ],
)
def test_nonsense_durations_are_refused(kwargs: dict[str, int], message: str) -> None:
    """A zero-length appointment would generate slots forever."""
    with pytest.raises(ValueError, match=message):
        slots_in([(_utc(TUESDAY, 8), _utc(TUESDAY, 9))], **kwargs)  # type: ignore[arg-type]


# ----------------------------------------------------------- the whole call
def test_bookable_slots_puts_the_pieces_together() -> None:
    slots = bookable_slots(
        start_date=TUESDAY,
        end_date=TUESDAY,
        rules=[MORNINGS],
        busy=[Busy(_utc(TUESDAY, 9), _utc(TUESDAY, 10))],
        duration_minutes=60,
        now=EARLY,
    )
    assert [s.start for s in slots] == [
        _utc(TUESDAY, 8),
        _utc(TUESDAY, 10),
        _utc(TUESDAY, 11),
    ], "08:00 before the booking, 10:00 and 11:00 after it"


def test_a_slot_already_in_the_past_is_never_offered() -> None:
    """Offering a time that has gone is worse than offering nothing."""
    slots = bookable_slots(
        start_date=TUESDAY,
        end_date=TUESDAY,
        rules=[MORNINGS],
        duration_minutes=60,
        now=_utc(TUESDAY, 10, 30),
    )
    assert [s.start for s in slots] == [_utc(TUESDAY, 11)]


def test_the_local_time_a_patient_reads_is_clinic_time() -> None:
    """Slots are UTC instants; what a patient is told is 08:00."""
    slots = bookable_slots(
        start_date=TUESDAY,
        end_date=TUESDAY,
        rules=[MORNINGS],
        duration_minutes=60,
        now=EARLY,
    )
    assert slots[0].local_start.hour == 8
    assert slots[0].start.hour == 13, "08:00 in Bogotá is 13:00 UTC"


def test_an_unbounded_horizon_is_refused() -> None:
    """A request for two years of slots is a mistake, not a question."""
    with pytest.raises(ValueError, match="exceeds the"):
        bookable_slots(
            start_date=TUESDAY,
            end_date=TUESDAY + dt.timedelta(days=MAX_HORIZON_DAYS),
            rules=[MORNINGS],
            duration_minutes=30,
            now=EARLY,
        )


def test_a_backwards_range_is_refused() -> None:
    with pytest.raises(ValueError, match="end_date cannot be before start_date"):
        bookable_slots(
            start_date=WEDNESDAY,
            end_date=TUESDAY,
            rules=[MORNINGS],
            duration_minutes=30,
            now=EARLY,
        )


def test_a_window_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="window must be ordered"):
        Window(TUESDAY, dt.time(12), dt.time(8))
