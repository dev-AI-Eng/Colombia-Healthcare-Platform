"""`verified_facts`: the only facts a generated message may contain.

ADR-02 and ADR-22. Every fact in an outbound message traces to this object
(rule 4), and the provenance stage enforces that by **whitelist match** rather
than by asking a model whether a sentence looks true. That is the whole reason
this type exists: fabrication detection becomes a set-membership test on values
deterministic code put here, which is a question code can answer completely.

Two consequences shape the design:

- **Nothing here is free text from a patient.** Every field is written by a tool
  call against verified database state. Patient text never reaches this object,
  because rule 3 makes it data rather than a fact.
- **Every renderable value has to be enumerable.** The provenance check walks
  the draft and resolves each date, time, name and number against `tokens()`.
  A fact that cannot be listed as a string cannot be checked, so it cannot be
  allowed into a message.

Spanish normalisation lives in `normalise` rather than in the checker, so the
same rules apply when a fact is stored and when a draft is checked. Doing it in
two places is how "10:30 a.m." passes and "10:30 AM" is called a fabrication.
"""

from __future__ import annotations

import datetime as dt
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Final

from src.core.timezones import BOGOTA

#: Spanish month and day names, lower case and unaccented, as a reader writes
#: them. Rendered text uses these; the checker resolves against them.
MONTHS_ES: Final[tuple[str, ...]] = (
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
)

#: Monday-first, matching `date.weekday()`.
WEEKDAYS_ES: Final[tuple[str, ...]] = (
    "lunes",
    "martes",
    "miercoles",
    "jueves",
    "viernes",
    "sabado",
    "domingo",
)


def normalise(value: str) -> str:
    """Casefold, strip accents, and collapse whitespace and punctuation runs.

    Applied identically when a fact is registered and when a draft token is
    checked. "Miércoles" and "miercoles" are the same fact; so are "10:30 a.m."
    and "10:30 a. m.", which is how a Colombian receptionist often writes it.
    """
    folded = unicodedata.normalize("NFKD", value.casefold())
    without_accents = "".join(ch for ch in folded if not unicodedata.combining(ch))
    collapsed = re.sub(r"\s+", " ", without_accents).strip()
    # "a. m." and "a.m." are the same meridiem, and Colombian clinics write
    # both. Collapsed to one form here rather than in the checker, so storing
    # and checking cannot disagree about it.
    return re.sub(r"\b([ap])\.?\s?m\.?", r"\1.m.", collapsed)


def spell_date(value: dt.date) -> str:
    """`2026-10-23` -> `jueves 23 de octubre`. Patient-facing Spanish."""
    return f"{WEEKDAYS_ES[value.weekday()]} {value.day} de {MONTHS_ES[value.month - 1]}"


def spell_time(value: dt.datetime) -> str:
    """`10:30` -> `10:30 a.m.`, in Bogotá time and 12-hour form (§7).

    Rendered by code, never by a model: ADR-02's "appointment facts rendered by
    code" is precisely this function, because a model writing "10:30" for an
    11:30 appointment is the failure the whole pipeline exists to prevent.
    """
    local = value.astimezone(BOGOTA)
    hour = local.hour % 12 or 12
    meridiem = "a.m." if local.hour < 12 else "p.m."
    return f"{hour}:{local.minute:02d} {meridiem}"


@dataclass(frozen=True, slots=True)
class SlotFact:
    """One offered time. Both halves are rendered, so both are checkable."""

    starts_at: dt.datetime
    duration_minutes: int

    @property
    def date_text(self) -> str:
        return spell_date(self.starts_at.astimezone(BOGOTA).date())

    @property
    def time_text(self) -> str:
        return spell_time(self.starts_at)


@dataclass(frozen=True, slots=True)
class VerifiedFacts:
    """What a generated message is allowed to say.

    Built by tool calls before generation (ADR-02). The generator receives this
    and nothing else that looks like a fact, so the provenance stage can treat
    any unresolvable value in the draft as fabricated.
    """

    patient_given_name: str | None = None
    doctor_name: str | None = None
    specialty: str | None = None
    location_name: str | None = None
    location_address: str | None = None
    clinic_name: str | None = None
    clinic_phone: str | None = None
    appointment_starts_at: dt.datetime | None = None
    appointment_duration_minutes: int | None = None
    offered_slots: tuple[SlotFact, ...] = ()
    #: Arrival advice and similar clinic policy: a number a message may state
    #: only because a tool put it here, not because a model knew it.
    arrive_minutes_early: int | None = None

    def tokens(self) -> frozenset[str]:
        """Every value a draft may contain, normalised.

        The provenance check resolves against this set. Numbers appear in every
        form a message might render them in — a date as `23`, `23 de octubre`
        and the full weekday phrasing — because a draft saying "el 23" is
        stating a fact that is in here, and calling that a fabrication would
        make the gate unusable.
        """
        out: set[str] = set()

        def add(value: str | None) -> None:
            if value:
                out.add(normalise(value))

        for simple in (
            self.patient_given_name,
            self.doctor_name,
            self.specialty,
            self.location_name,
            self.location_address,
            self.clinic_name,
            self.clinic_phone,
        ):
            add(simple)

        # A name is also resolvable by its parts: a message addressing "Sara"
        # when the fact is "Sara Betancur" is not inventing a person.
        for compound in (self.patient_given_name, self.doctor_name):
            for part in (compound or "").split():
                if len(part) > 2:
                    add(part)

        if self.clinic_phone:
            digits = re.sub(r"\D", "", self.clinic_phone)
            add(digits)
            # A phone is written in groups, and the checker sees each group as
            # its own number: "601 555 0101" tokenises to three. Registering
            # the groups individually is what stops a clinic's own number being
            # reported as three fabrications. Groups of one or two digits are
            # left out: they are too short to be distinctive and would let an
            # unrelated "45 minutos" resolve.
            for group in re.findall(r"\d{3,}", self.clinic_phone):
                add(group)
            # Also the national form without the country code, which is how a
            # message usually prints it.
            if digits.startswith("57") and len(digits) > 8:
                add(digits[2:])

        if self.arrive_minutes_early is not None:
            add(str(self.arrive_minutes_early))

        for slot in self.all_times():
            local = slot.starts_at.astimezone(BOGOTA)
            add(slot.date_text)
            add(slot.time_text)
            add(f"{local.day}")
            add(f"{local.day} de {MONTHS_ES[local.month - 1]}")
            add(MONTHS_ES[local.month - 1])
            add(WEEKDAYS_ES[local.date().weekday()])
            add(local.strftime("%d/%m/%Y"))
            add(local.strftime("%Y-%m-%d"))
            # Bare clock forms a draft may legitimately use.
            add(f"{local.hour % 12 or 12}:{local.minute:02d}")
            add(f"{local.hour:02d}:{local.minute:02d}")
            add(str(slot.duration_minutes))

        return frozenset(out)

    def all_times(self) -> tuple[SlotFact, ...]:
        booked = (
            (SlotFact(self.appointment_starts_at, self.appointment_duration_minutes or 0),)
            if self.appointment_starts_at is not None
            else ()
        )
        return booked + self.offered_slots

    def is_empty(self) -> bool:
        """No facts at all.

        A generated message with no verified facts behind it has nothing to say
        and nothing to check, so the graph routes it to a static template rather
        than letting a model improvise.
        """
        return not self.tokens()


@dataclass
class FactsBuilder:
    """Accumulates facts from tool calls, then freezes them.

    Mutable while tools run, immutable once generation starts: a fact added
    after the provenance set was computed would be a fact nobody checked.
    """

    _values: dict[str, object] = field(default_factory=dict)
    _slots: list[SlotFact] = field(default_factory=list)

    def set(self, **values: object) -> FactsBuilder:
        self._values.update({k: v for k, v in values.items() if v is not None})
        return self

    def offer(self, starts_at: dt.datetime, duration_minutes: int) -> FactsBuilder:
        self._slots.append(SlotFact(starts_at, duration_minutes))
        return self

    def build(self) -> VerifiedFacts:
        return VerifiedFacts(offered_slots=tuple(self._slots), **self._values)  # type: ignore[arg-type]
