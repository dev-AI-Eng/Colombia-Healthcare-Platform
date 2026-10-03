"""Propose which canonical field each column of a clinic file holds.

The stages run cheapest first and stop as soon as one is confident, so a file
written in ordinary Spanish resolves entirely here and **never reaches a model**:

    1. exact      the header equals a field name or a known alias
    2. alias      the header contains an alias as whole words
    3. fuzzy      rapidfuzz against every alias, above a threshold
    4. (later)    a model reranks only what is still ambiguous

Two rules shape everything below.

**The matcher proposes, it never converts.** A proposal says "this column looks
like `birth_date`". Which normalizer runs is fixed in `canonical.py`, so no
transform is ever inferred from the file's contents (ADR-08a).

**Confidence decides who decides.** A strong match is pre-ticked for the human
to glance at; a weak one is shown as a question. Nothing is ever applied without
someone confirming the screen.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from rapidfuzz import fuzz, process

from src.onboarding.canonical import ALL_FIELDS, Entity, Field, Requirement
from src.onboarding.normalizers import strip_accents

# Above this, a fuzzy match is proposed as confident. Below the floor it is not
# proposed at all: a bad suggestion costs more attention than a blank one,
# because a reviewer has to work out why it is wrong.
FUZZY_CONFIDENT: Final = 90.0
FUZZY_FLOOR: Final = 72.0


class Confidence(StrEnum):
    EXACT = "exact"  # the header is the field's name or a known alias
    STRONG = "strong"  # contained as whole words, or a very close fuzzy match
    WEAK = "weak"  # plausible, but a person should look
    NONE = "none"  # nothing matched


@dataclass(frozen=True, slots=True)
class Proposal:
    """One column, and the canonical field it appears to hold."""

    column: str
    field: Field | None
    confidence: Confidence
    score: float
    #: Why this was proposed, shown on the confirmation screen so a reviewer can
    #: judge the suggestion instead of trusting it.
    reason: str

    @property
    def auto(self) -> bool:
        """Whether the screen pre-ticks this proposal."""
        return self.confidence in (Confidence.EXACT, Confidence.STRONG)


@dataclass(frozen=True, slots=True)
class SheetMapping:
    entity: Entity
    proposals: tuple[Proposal, ...]
    #: Required fields no column was proposed for. The import cannot proceed
    #: until a human maps these or the file is re-exported.
    missing_required: tuple[str, ...]

    @property
    def needs_review(self) -> tuple[Proposal, ...]:
        return tuple(p for p in self.proposals if not p.auto)


def normalize_header(header: str) -> str:
    """Reduce a header to comparable words.

    Accents go because unaccented spelling is the norm in real exports;
    punctuation goes because `No.`, `N°` and `#` are all written for the same
    column; and runs of spaces collapse because hand-made files are untidy.
    """
    # camelCase before casefolding, which is what destroys the word boundary:
    # RIPS names its own fields `primerNombre` and `numDocumentoIdentificacion`
    # (Resolución 1036 de 2022), so a clinic exporting in the shape the
    # government asked for would otherwise match nothing at all.
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", header)
    text = strip_accents(text)
    # Underscores are word separators in exported headers ("id_type",
    # "doctor_id"), but `\w` keeps them, so they are replaced explicitly.
    # Without this an exported column never matches the words it is made of.
    text = re.sub(r"[^\w\s]+|_", " ", text)
    return " ".join(text.split())


#: Fields more than one column may legitimately fill, because the columns hold
#: parts of one value rather than rival versions of it. Everything else keeps the
#: rule that two columns mapping to one field means one silently wins.
SHARED_FIELDS: Final = frozenset({"given_names", "family_names"})

# Headers that mean "no column", not "a column called nothing".
_EMPTY_HEADERS: Final = frozenset({"", "n a", "na", "sin nombre", "column1", "unnamed 0"})

# `fecha` alone is genuinely ambiguous: a clinic file may use it for the
# appointment or for the date of birth. These words decide it, and they are
# checked before any general match so a birth date is never imported as an
# appointment date.
_BIRTH_MARKERS: Final = ("nacimiento", "nacim", "fdn", "birth")
_APPOINTMENT_MARKERS: Final = ("cita", "atencion", "programada", "appointment", "consulta")


def _candidates(entity: Entity) -> tuple[Field, ...]:
    return tuple(f for f in ALL_FIELDS if f.entity is entity)


def _alias_forms(field: Field) -> tuple[str, ...]:
    """Every string that should match this field, normalized once."""
    return tuple({normalize_header(a) for a in (field.name.replace("_", " "), *field.aliases)})


def _disambiguate_date(header: str, entity: Entity) -> Field | None:
    """Resolve a bare date header before anything else looks at it."""
    words = normalize_header(header)
    if "fecha" not in words and "date" not in words:
        return None
    if any(marker in words for marker in _BIRTH_MARKERS):
        return next((f for f in _candidates(entity) if f.name == "birth_date"), None)
    if any(marker in words for marker in _APPOINTMENT_MARKERS):
        return next((f for f in _candidates(entity) if f.name == "appointment_date"), None)
    return None


def match_column(header: str, entity: Entity, *, taken: set[str] | None = None) -> Proposal:
    """Propose a canonical field for one column header."""
    taken = taken or set()
    words = normalize_header(header)
    #: The header with its spaces removed. Splitting camelCase is what lets a
    #: RIPS heading like `primerNombre` match, but it also splits a product name
    #: a clinic writes as one word: "WhatsApp" becomes "whats app", which the
    #: alias `whatsapp` no longer matches. Comparing the joined form too handles
    #: every such name without a list of brands to maintain.
    joined = words.replace(" ", "")

    if words in _EMPTY_HEADERS:
        return Proposal(header, None, Confidence.NONE, 0.0, "The column has no heading.")

    available = tuple(f for f in _candidates(entity) if f.name not in taken)
    if not available:
        return Proposal(header, None, Confidence.NONE, 0.0, "Every field is already mapped.")

    # A date column is decided by its own words before general matching, so
    # "fecha de nacimiento" can never be taken for the appointment date.
    if (decided := _disambiguate_date(header, entity)) and decided.name not in taken:
        return Proposal(
            header,
            decided,
            Confidence.STRONG,
            98.0,
            f"The heading says {'birth' if decided.name == 'birth_date' else 'appointment'}.",
        )

    # 1. exact
    for field in available:
        forms = _alias_forms(field)
        if words in forms or joined in forms:
            return Proposal(header, field, Confidence.EXACT, 100.0, "The heading is a known name.")

    # 2. whole-word containment, longest alias first so "tipo de documento"
    #    beats "documento" for the same header.
    #
    #    The alias must also account for most of the header. "rescheduled from
    #    appointment id" contains "appointment id" but means something else
    #    entirely, and importing it as the appointment's own code would link
    #    every rescheduled appointment to the wrong row.
    best_contained: tuple[int, Field, str] | None = None
    header_word_count = len(words.split())
    for field in available:
        for alias in _alias_forms(field):
            if not alias:
                continue
            if not re.search(rf"\b{re.escape(alias)}\b", words):
                continue
            if len(alias.split()) < header_word_count - 1:
                continue  # the header says more than the alias accounts for
            if best_contained is None or len(alias) > best_contained[0]:
                best_contained = (len(alias), field, alias)
    if best_contained:
        _, field, alias = best_contained
        return Proposal(header, field, Confidence.STRONG, 95.0, f"The heading contains {alias!r}.")

    # 3. fuzzy, against every alias of every candidate
    lookup: dict[str, Field] = {}
    for field in available:
        for alias in _alias_forms(field):
            lookup.setdefault(alias, field)

    # token_set_ratio ignores tokens the header has and the alias does not, so
    # "rescheduled from appointment id" scores 81 against "appointment date" on
    # the strength of one shared word. token_sort_ratio compares the whole
    # strings, and the lower of the two keeps a long header from matching every
    # alias that shares a single token with it.
    def _score(header_words: str, alias: str, **_: object) -> float:
        # rapidfuzz passes score_cutoff to a custom scorer, hence **_.
        return min(
            fuzz.token_set_ratio(header_words, alias),
            fuzz.token_sort_ratio(header_words, alias),
        )

    found = process.extractOne(words, lookup.keys(), scorer=_score)
    if found:
        alias, score, _ = found
        field = lookup[alias]
        if score >= FUZZY_CONFIDENT:
            return Proposal(
                header, field, Confidence.STRONG, float(score), f"Very close to {alias!r}."
            )
        if score >= FUZZY_FLOOR:
            return Proposal(header, field, Confidence.WEAK, float(score), f"Similar to {alias!r}.")

    return Proposal(header, None, Confidence.NONE, 0.0, "No canonical field resembles this column.")


def match_sheet(headers: tuple[str, ...], entity: Entity) -> SheetMapping:
    """Propose a field for every column of a sheet.

    Columns are matched in order of how clearly they match, not left to right,
    so a confident column claims its field before a vaguer one can take it. Two
    columns are never proposed for the same field: one of them would silently
    overwrite the other.
    """
    ranked = sorted(
        (match_column(h, entity) for h in headers),
        key=lambda p: (-p.score, p.column),
    )

    taken: set[str] = set()
    decided: dict[str, Proposal] = {}
    for proposal in ranked:
        if proposal.field is None:
            decided[proposal.column] = proposal
            continue
        if proposal.field.name in SHARED_FIELDS:
            # RIPS splits a name into four fields (Resolución 1036 de 2022:
            # primerNombre, segundoNombre, primerApellido, segundoApellido) and
            # a clinic exporting in that shape sends four columns for our two.
            # They are parts of one value, not two columns competing for it, so
            # they share the field and are joined in file order on conversion.
            taken.add(proposal.field.name)
            decided[proposal.column] = proposal
            continue
        if proposal.field.name in taken:
            # Re-match without the field that has already been claimed.
            decided[proposal.column] = match_column(proposal.column, entity, taken=taken)
            if (again := decided[proposal.column].field) is not None:
                taken.add(again.name)
            continue
        taken.add(proposal.field.name)
        decided[proposal.column] = proposal

    missing = tuple(
        f.name
        for f in _candidates(entity)
        if f.requirement is Requirement.REQUIRED and f.name not in taken
    )
    # A file that separates names satisfies the name requirement differently
    # from one that supplies a single column, so neither is required alone.
    if entity is Entity.PATIENT and (
        "full_name" in taken or {"given_names", "family_names"} <= taken
    ):
        missing = tuple(m for m in missing if m not in {"full_name", "given_names", "family_names"})

    return SheetMapping(
        entity=entity,
        proposals=tuple(decided[h] for h in headers),
        missing_required=missing,
    )


# Which entity a sheet holds, guessed from its name and headers. A person
# confirms this too: a sheet named "Hoja1" says nothing.
_SHEET_HINTS: Final[dict[Entity, tuple[str, ...]]] = {
    Entity.PATIENT: ("paciente", "patient", "usuario", "afiliado"),
    # A clinic names this sheet after the practitioner it employs, so the word
    # is the specialty as often as it is "medico": a dental suite exports
    # "Dentistas", and a sheet of doctors guessed as patients maps none of its
    # columns, because a patient has no specialty or consulting room.
    Entity.DOCTOR: (
        "medico",
        "doctor",
        "profesional",
        "especialista",
        "dentista",
        "odontolog",
        "prestador",
        "terapeuta",
    ),
    Entity.SPECIALTY: ("especialidad", "specialt", "servicio"),
    Entity.AVAILABILITY: ("disponibilidad", "availability", "horario", "agenda medico"),
    Entity.APPOINTMENT: ("cita", "appointment", "agenda", "consulta", "turno"),
}


def guess_entity(sheet_name: str, headers: tuple[str, ...]) -> tuple[Entity, str]:
    """Guess what a sheet holds, and say why."""
    name = normalize_header(sheet_name)
    # The longest matching hint wins. "Doctor_Availability" contains both
    # "doctor" and "availability", and "Outpatient_Appointments" contains both
    # "patient" and "appointment"; taking the first hit in declaration order
    # would read an appointments sheet as a list of patients.
    hits = [
        (len(hint), entity)
        for entity, hints in _SHEET_HINTS.items()
        for hint in hints
        if hint in name
    ]
    if hits:
        _, entity = max(hits)
        return entity, f"The sheet is called {sheet_name!r}."

    # Fall back to whichever entity the headers satisfy best.
    scores: list[tuple[float, Entity]] = []
    for entity in Entity:
        mapping = match_sheet(headers, entity)
        matched = sum(1 for p in mapping.proposals if p.auto)
        scores.append((matched / max(len(headers), 1), entity))
    best_share, best_entity = max(scores)
    if best_share == 0.0:
        # Nothing matched anything, which happens when the headings carry no
        # meaning at all: a file whose columns are named by position because it
        # had no heading row. Saying "0% of the columns match specialty fields"
        # while reading it as specialties states a conclusion the evidence does
        # not support, and `max` over an all-zero tie picks arbitrarily. Patients
        # are what a clinic imports, so that is the default, and the reason says
        # plainly that the reviewer has to choose.
        return Entity.PATIENT, (
            "No column matched any known field, so this is a guess: choose what the sheet holds."
        )
    return best_entity, f"{best_share:.0%} of the columns match {best_entity.value} fields."
