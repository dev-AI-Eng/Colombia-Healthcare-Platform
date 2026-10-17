"""The deterministic gate. First rejecting stage wins (ADR-22).

Four stages, in this order, because the cheap certain checks must run before
the expensive uncertain one:

1. **Provenance** — every fact-shaped token in the draft resolves to
   `verified_facts`. Rule 4, enforced by code rather than by prompt.
2. **Forbidden content** — Spanish medical-advice lexicon. Rule 5.
3. **Omission** — the facts this message type must carry are present. The
   dominant patient-facing model error in the 2025 patient-safety analysis was
   omission at 42%, ahead of fabrication, and a model judge cannot be trusted
   to measure completeness (DeLucia et al. put LLM judges at AUC 0.49-0.66 on
   exactly that question, near chance).
4. **Model judge** — paraphrased advice and tone, which code cannot catch. Not
   in this module: stages 1-3 are deterministic and this one is not, and the
   point of ADR-22 is that the control which does not degrade runs first.

Stages 1-3 carry M3's hardest exit criterion. Rev3 asks for 95% of
fabricated-fact cases rejected; this is deterministic code, so it either
implements the rule or has a bug, and the suite asserts **100%**. Exceeding the
contract is safer than encoding a 5% tolerance that has no meaning here.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from src.conversation.facts import VerifiedFacts, normalise
from src.core.timezones import BOGOTA


class Reason(StrEnum):
    """Why a draft was rejected. A closed set, so retries can be fed codes
    rather than free-form critique (ADR-23)."""

    FABRICATED_FACT = "fabricated_fact"
    MEDICAL_ADVICE = "medical_advice"
    MISSING_REQUIRED_FACT = "missing_required_fact"
    #: Stage 4's code, raised by the model judge rather than here.
    TONE = "tone"


class MessageKind(StrEnum):
    """What a message is for. Decides which facts stage 3 requires."""

    CONFIRMATION = "confirmation"
    CANCELLATION = "cancellation"
    SLOT_OFFER = "slot_offer"
    REMINDER = "reminder"
    #: "A staff member will contact you" — carries no appointment facts, so
    #: stage 3 requires nothing of it.
    FALLBACK = "fallback"


#: Facts each message type must state. A confirmation that omits the time is
#: worse than no confirmation: the patient believes they are booked and does
#: not know when.
REQUIRED: Final[dict[MessageKind, tuple[str, ...]]] = {
    MessageKind.CONFIRMATION: ("appointment_starts_at", "doctor_name"),
    MessageKind.REMINDER: ("appointment_starts_at",),
    MessageKind.SLOT_OFFER: ("offered_slots",),
    MessageKind.CANCELLATION: (),
    MessageKind.FALLBACK: (),
}

# --------------------------------------------------------------- stage 2 lexicon
#
# Spanish for Colombia (rule 10). Patterns, not a word list, because "tome dos
# tabletas" and "tome 2 tabletas" are the same instruction and a bare word list
# catches neither reliably. Each pattern is anchored on a *verb of instruction*
# or a dosage shape rather than on a medical noun: a message may legitimately
# name a specialty ("cardiologia") or say "traiga sus examenes", and rejecting
# those would make the gate unusable.

#: Quantities a dosage instruction uses. Spelled-out numbers matter as much as
#: digits: "tome dos tabletas" is the same instruction as "tome 2 tabletas",
#: and a pattern that only sees digits misses the more natural phrasing.
_QUANTITY: Final[str] = (
    r"(?:\d+|un|una|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|media|medio)"
)

_ADVICE_PATTERNS: Final[tuple[tuple[str, str], ...]] = (
    # Instruction to take, apply, increase or stop a medication or dose. Verbs
    # are listed in every form a draft might use -- imperative, infinitive and
    # the "puede/debe + infinitive" construction a hedging model prefers.
    (
        r"\b(tome|tomar|toma|tomese|tomese)\b.{0,20}"
        rf"{_QUANTITY}?.{{0,20}}\b(tableta|tabletas|pastilla|pastillas|capsula|capsulas"
        r"|dosis|mg|ml|gotas?|jarabe|sobres?)\b",
        "dosage",
    ),
    (
        r"\b(aplique|aplicar|inyecte|inyectar|poner|ponerse)\b.{0,40}\b(dosis|mg|ml|unidades?)\b",
        "dosage",
    ),
    (
        r"\b(aumente|aumentar|disminuya|disminuir|suspenda|suspender|reduzca|reducir"
        r"|deje de tomar|dejar de tomar|cambie|cambiar)\b.{0,30}"
        r"\b(dosis|medicamento|medicina|tratamiento|pastillas?)\b",
        "dosage",
    ),
    (r"\b\d+\s?(mg|ml|mcg|g)\b", "dosage"),
    # Diagnosis stated to the patient.
    (r"\b(usted|ud\.?)\s+(tiene|padece|sufre de)\b", "diagnosis"),
    (r"\b(es|son)\s+(sintomas?|signos?)\s+de\b", "diagnosis"),
    (r"\bprobablemente\s+(sea|tenga|es)\b", "diagnosis"),
    (r"\b(su|el)\s+diagnostico\s+(es|seria)\b", "diagnosis"),
    # Treatment recommendation.
    (
        r"\b(le recomiendo|recomendamos|deberia)\b.{0,30}\b(tomar|tratamiento|medicamento|cirugia)\b",
        "treatment",
    ),
    (r"\b(no se preocupe|no es grave|es normal)\b", "reassurance"),
    (r"\bno necesita\b.{0,30}\b(cita|consulta|medico|examen)\b", "reassurance"),
)

_COMPILED: Final[tuple[tuple[re.Pattern[str], str], ...]] = tuple(
    (re.compile(pattern), label) for pattern, label in _ADVICE_PATTERNS
)

# ------------------------------------------------------------- stage 1 tokenising
#
# What counts as a fact-shaped token: anything a message could get wrong in a
# way that matters. Prose is not checked -- "su cita ha sido confirmada" states
# no fact -- so the checker looks only for values.

#: A time: 10:30, 10:30 a.m., 10 a.m. The meridiem group is optional, so the
#: trailing separator has to be inside it -- otherwise a bare "10:30 " matches
#: with the space attached and normalises to a token the fact set never holds.
_TIME = re.compile(
    r"\b\d{1,2}:\d{2}(?:\s?(?:a\.?\s?m\.?|p\.?\s?m\.?))?"
    r"|\b\d{1,2}\s?(?:a\.?\s?m\.?|p\.?\s?m\.?)"
)
#: A date: 23/10/2026, 2026-10-23, "23 de octubre".
_DATE = re.compile(
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b"
    r"|\b\d{1,2}\s+de\s+[a-z]+\b"
)
#: A bare number of two or more digits: a duration, a phone, a room.
_NUMBER = re.compile(r"\b\d{2,}\b")
#: A capitalised run: a person, a place. Lower-cased sentence starts are not
#: names, so the pattern requires the capital to be mid-sentence or a title.
_PROPER = re.compile(
    r"\b(?:Dr\.?|Dra\.?|Doctor|Doctora)\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ]+)*"
)

#: Words that look like facts but state none, so they never need to resolve.
#: Kept deliberately short: every entry here is a hole in the gate.
_STOPWORDS: Final[frozenset[str]] = frozenset(
    normalise(w)
    for w in (
        # Relative time a templated message may legitimately use, because the
        # clinic's own policy text says it rather than a model inventing it.
        "manana",
        "hoy",
        "ayer",
    )
)


@dataclass(frozen=True, slots=True)
class Finding:
    """One reason a draft failed, with the span that caused it.

    The span matters: ADR-23 feeds retries "reason codes and quoted offending
    spans, not free-form critique", and a retry told only "fabricated fact"
    has nothing to correct.
    """

    reason: Reason
    span: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class Verdict:
    approved: bool
    findings: tuple[Finding, ...] = ()

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(f.reason.value for f in self.findings))


def _fact_tokens(draft: str) -> list[str]:
    """Every fact-shaped span in the draft, longest patterns first.

    Dates before numbers, so "23 de octubre" is checked as a date rather than
    as the bare number 23 — the two resolve differently and the date is the
    stronger claim.
    """
    spans: list[str] = []
    consumed: list[tuple[int, int]] = []

    def take(pattern: re.Pattern[str], text: str) -> None:
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(s <= start and end <= e for s, e in consumed):
                continue
            consumed.append((start, end))
            spans.append(match.group(0))

    take(_PROPER, draft)
    lowered = draft.casefold()
    take(_DATE, lowered)
    take(_TIME, lowered)
    take(_NUMBER, lowered)
    return spans


def check_provenance(draft: str, facts: VerifiedFacts) -> list[Finding]:
    """Stage 1. Every fact-shaped token resolves to `verified_facts`."""
    allowed = facts.tokens()
    findings: list[Finding] = []
    for span in _fact_tokens(draft):
        token = normalise(span)
        if token in allowed or token in _STOPWORDS:
            continue
        findings.append(Finding(Reason.FABRICATED_FACT, span, "not present in verified_facts"))
    return findings


def check_forbidden(draft: str) -> list[Finding]:
    """Stage 2. Medical advice, dosage, diagnosis (rule 5)."""
    text = normalise(draft)
    return [
        Finding(Reason.MEDICAL_ADVICE, match.group(0), label)
        for pattern, label in _COMPILED
        if (match := pattern.search(text))
    ]


def check_omission(draft: str, facts: VerifiedFacts, kind: MessageKind) -> list[Finding]:
    """Stage 3. The facts this message type must carry are actually in it."""
    findings: list[Finding] = []
    present = {normalise(span) for span in _fact_tokens(draft)}

    for name in REQUIRED[kind]:
        value = getattr(facts, name, None)
        if not value:
            # The fact itself is missing upstream, which is a tool failure
            # rather than a drafting failure. Still a rejection: the message
            # cannot be complete without it.
            findings.append(Finding(Reason.MISSING_REQUIRED_FACT, name, "not in verified_facts"))
            continue
        if not _mentions(name, facts, present):
            findings.append(Finding(Reason.MISSING_REQUIRED_FACT, name, "absent from draft"))
    return findings


def _mentions(name: str, facts: VerifiedFacts, present: Iterable[str]) -> bool:
    """Does the draft state this required fact in any rendered form?"""
    seen = set(present)
    if name == "appointment_starts_at" and facts.appointment_starts_at:
        slot = facts.all_times()[0]
        # Any rendered form of the moment counts as stating it. A draft saying
        # "a las 10:30" has told the patient when to come as surely as
        # "10:30 a.m." did, and demanding one spelling here would reject
        # correct drafts -- the behaviour stage 1 already has to tolerate.
        local = slot.starts_at.astimezone(BOGOTA)
        forms = {
            normalise(slot.time_text),
            normalise(slot.date_text),
            normalise(f"{local.hour % 12 or 12}:{local.minute:02d}"),
            normalise(f"{local.hour:02d}:{local.minute:02d}"),
        }
        return bool(seen & forms)
    if name == "offered_slots":
        return any(
            normalise(s.time_text) in seen or normalise(s.date_text) in seen
            for s in facts.offered_slots
        )
    value = getattr(facts, name, None)
    if not isinstance(value, str):
        return False
    whole = normalise(value)
    return whole in seen or any(whole in s or s in whole for s in seen)


def validate(draft: str, facts: VerifiedFacts, kind: MessageKind) -> Verdict:
    """Run stages 1-3. First rejecting stage wins (ADR-22).

    Returning on the first failing stage is deliberate rather than an
    optimisation: a draft that fabricates a date and also gives dosage advice
    needs one clear reason to retry against, and a retry handed three codes at
    once tends to fix none of them.
    """
    # Written out rather than looped so each stage runs only if the one before
    # it passed. The stages are cheap and pure, so evaluating all three would
    # be harmless -- but then the first-stage-wins rule would live in the
    # return statement rather than in the control flow, and the next person to
    # add a stage with a cost would not notice.
    if findings := check_provenance(draft, facts):
        return Verdict(approved=False, findings=tuple(findings))
    if findings := check_forbidden(draft):
        return Verdict(approved=False, findings=tuple(findings))
    if findings := check_omission(draft, facts, kind):
        return Verdict(approved=False, findings=tuple(findings))
    return Verdict(approved=True)
