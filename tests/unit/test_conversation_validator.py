"""M3's red-team suite: the deterministic gate must reject every bad draft.

Rev3's exit criterion asks for 95% of fabricated-fact cases rejected on every
run. This asserts **100%**, because the validator is deterministic code: it
either implements the rule or it has a bug, and a 5% allowance would mean one
fabricated fact in twenty reaching a patient. Exceeding the contract is the
safe direction. `test_every_fabrication_case_is_rejected` is the criterion.

Fixtures are Spanish for Colombia (§6), including the phrasings a clinic
actually writes: missing accents, "a. m." with a space, abbreviated months.
"""

from __future__ import annotations

import datetime as dt

import pytest

from src.conversation.facts import VerifiedFacts, spell_date, spell_time
from src.conversation.validator import (
    MessageKind,
    Reason,
    check_forbidden,
    check_provenance,
    validate,
)
from src.core.timezones import BOGOTA

APPOINTMENT = dt.datetime(2026, 10, 23, 10, 30, tzinfo=BOGOTA)

FACTS = VerifiedFacts(
    patient_given_name="Sara",
    doctor_name="Sara Betancur de la Cruz",
    specialty="Medicina general",
    location_name="Sede Norte",
    clinic_name="Clínica San Rafael",
    clinic_phone="+57 601 555 0101",
    appointment_starts_at=APPOINTMENT,
    appointment_duration_minutes=30,
    arrive_minutes_early=30,
)

#: What a correct confirmation looks like, rendered from the facts above.
GOOD_CONFIRMATION = (
    f"Su cita ha sido confirmada para el {spell_date(APPOINTMENT.date())} "
    f"a las {spell_time(APPOINTMENT)} con Dra. Sara Betancur de la Cruz. "
    "Por favor llegue 30 minutos antes."
)


# ------------------------------------------------------- stage 1: fabrication
#
# Every case changes exactly one fact to a value no tool put in the object. A
# human reading the draft could not tell; the whitelist match can.

FABRICATIONS: tuple[tuple[str, str], ...] = (
    (
        "a different day",
        f"Su cita es el viernes 24 de octubre a las {spell_time(APPOINTMENT)}.",
    ),
    (
        "a different time",
        f"Su cita es el {spell_date(APPOINTMENT.date())} a las 11:30 a.m.",
    ),
    (
        "a plausible nearby time",
        f"Su cita es el {spell_date(APPOINTMENT.date())} a las 10:45 a.m.",
    ),
    (
        "a doctor who does not exist",
        f"Su cita es el {spell_date(APPOINTMENT.date())} con Dr. Carlos Peña.",
    ),
    (
        "an invented room number",
        f"Su cita es a las {spell_time(APPOINTMENT)} en el consultorio 412.",
    ),
    (
        "an invented arrival time",
        f"Su cita es a las {spell_time(APPOINTMENT)}. Llegue 45 minutos antes.",
    ),
    (
        "an invented phone number",
        f"Su cita es a las {spell_time(APPOINTMENT)}. Llame al 601 555 9999.",
    ),
    (
        "a date in ISO form that is not the appointment",
        "Su cita es el 2026-11-05.",
    ),
    (
        "a date in slash form that is not the appointment",
        "Su cita es el 24/10/2026.",
    ),
    (
        "an invented duration",
        f"Su cita es a las {spell_time(APPOINTMENT)} y dura 60 minutos.",
    ),
    (
        "a second invented appointment alongside the real one",
        f"Su cita es el {spell_date(APPOINTMENT.date())} a las {spell_time(APPOINTMENT)}. "
        "Su control es el 30 de noviembre.",
    ),
    (
        "an invented month",
        "Su cita es el 23 de diciembre.",
    ),
)


@pytest.mark.parametrize(("label", "draft"), FABRICATIONS, ids=[c[0] for c in FABRICATIONS])
def test_every_fabrication_case_is_rejected(label: str, draft: str) -> None:
    """M3's exit criterion, at 100% rather than Rev3's 95%."""
    verdict = validate(draft, FACTS, MessageKind.CONFIRMATION)
    assert not verdict.approved, f"{label}: fabricated draft was approved"
    assert Reason.FABRICATED_FACT.value in verdict.reason_codes


def test_the_fabrication_rate_is_total() -> None:
    """Stated as a rate, so the number in the report is measured not claimed."""
    rejected = sum(
        1
        for _, draft in FABRICATIONS
        if not validate(draft, FACTS, MessageKind.CONFIRMATION).approved
    )
    assert rejected == len(FABRICATIONS)


def test_a_correct_confirmation_is_approved() -> None:
    """The gate has to pass good drafts, or it is a gate nobody can ship."""
    verdict = validate(GOOD_CONFIRMATION, FACTS, MessageKind.CONFIRMATION)
    assert verdict.approved, verdict.findings


@pytest.mark.parametrize(
    "draft",
    [
        # Accentless, as a Colombian receptionist often types.
        "Su cita es el viernes 23 de octubre a las 10:30 a.m. con Dra. Sara Betancur de la Cruz.",
        # "a. m." with a space.
        f"Su cita es el {spell_date(APPOINTMENT.date())} a las 10:30 a. m. con Dra. Sara Betancur de la Cruz.",
        # 24-hour clock.
        f"Su cita es el {spell_date(APPOINTMENT.date())} a las 10:30 con Dra. Sara Betancur de la Cruz.",
    ],
)
def test_spanish_variants_of_the_same_fact_are_approved(draft: str) -> None:
    """Normalisation is shared between storing and checking, so the same fact
    written three ways is one fact. Without this the gate rejects correct
    drafts and the retry loop burns its budget on nothing."""
    assert validate(draft, FACTS, MessageKind.CONFIRMATION).approved


def test_prose_without_facts_is_not_treated_as_fabrication() -> None:
    """A sentence stating no value has nothing to resolve."""
    assert check_provenance("Su cita ha sido confirmada. Gracias.", FACTS) == []


# --------------------------------------------------- stage 2: medical advice
#
# Rule 5. Each case is something a helpful-sounding model might write.

ADVICE: tuple[tuple[str, str], ...] = (
    ("a dosage", "Tome dos tabletas de acetaminofén cada 8 horas."),
    ("a dosage in milligrams", "Tome 500 mg antes de la cita."),
    ("a dose change", "Puede aumentar la dosis del medicamento si sigue el dolor."),
    ("stopping treatment", "Deje de tomar el tratamiento hasta su cita."),
    ("a diagnosis", "Usted tiene una infección urinaria."),
    ("symptom-to-diagnosis", "Esos son síntomas de apendicitis."),
    ("a hedged diagnosis", "Probablemente sea una gastritis."),
    ("a treatment recommendation", "Le recomiendo tomar antibióticos antes de venir."),
    ("false reassurance", "No se preocupe, no es grave."),
    ("telling a patient not to come", "No necesita consulta para eso."),
    ("an injection instruction", "Aplique 10 ml de la solución esta noche."),
)


@pytest.mark.parametrize(("label", "draft"), ADVICE, ids=[c[0] for c in ADVICE])
def test_medical_advice_is_rejected(label: str, draft: str) -> None:
    findings = check_forbidden(draft)
    assert findings, f"{label}: advice passed the lexicon"
    assert findings[0].reason is Reason.MEDICAL_ADVICE


@pytest.mark.parametrize(
    "draft",
    [
        # Naming a specialty is not diagnosing.
        "Su cita es de medicina general.",
        # Asking the patient to bring results is logistics, not advice.
        "Si tiene exámenes o imágenes, por favor tráigalos.",
        # Arrival advice is clinic policy, not clinical advice.
        "Por favor llegue 30 minutos antes.",
        # A cancellation naming a phone number.
        "Su cita fue cancelada. Para agendar otra llame al 601 555 0101.",
    ],
)
def test_ordinary_clinic_language_is_not_called_medical_advice(draft: str) -> None:
    """A gate that rejects "traiga sus exámenes" is a gate the clinic turns off.
    False rejection is bounded, not ignored (§6)."""
    assert check_forbidden(draft) == []


# -------------------------------------------------------- stage 3: omission
#
# 42% of patient-facing model errors in the 2025 analysis were omissions, ahead
# of fabrication, and a model judge measures completeness at near chance.


def test_a_confirmation_without_a_time_is_rejected() -> None:
    """The worst failure mode: the patient believes they are booked and does
    not know when."""
    verdict = validate(
        "Su cita ha sido confirmada con Dra. Sara Betancur de la Cruz.", FACTS, MessageKind.CONFIRMATION
    )
    assert not verdict.approved
    assert Reason.MISSING_REQUIRED_FACT.value in verdict.reason_codes


def test_a_confirmation_without_the_doctor_is_rejected() -> None:
    verdict = validate(
        f"Su cita ha sido confirmada para las {spell_time(APPOINTMENT)}.",
        FACTS,
        MessageKind.CONFIRMATION,
    )
    assert not verdict.approved
    assert Reason.MISSING_REQUIRED_FACT.value in verdict.reason_codes


def test_a_slot_offer_must_actually_offer_a_slot() -> None:
    offers = VerifiedFacts(
        offered_slots=(),
    )
    verdict = validate("Tenemos horarios disponibles.", offers, MessageKind.SLOT_OFFER)
    assert not verdict.approved


def test_a_cancellation_needs_no_appointment_facts() -> None:
    """Required facts are per message type: a cancellation that stated a time
    would be stating a fact about an appointment that no longer exists."""
    verdict = validate(
        "Su cita fue cancelada. Para agendar otra llame al 601 555 0101.",
        FACTS,
        MessageKind.CANCELLATION,
    )
    assert verdict.approved, verdict.findings


def test_the_fallback_template_carries_no_facts_and_passes() -> None:
    """ADR-23's exhaustion path. If this could not pass the gate, the retry
    loop would have nowhere safe to land."""
    verdict = validate(
        "Un miembro del personal se pondrá en contacto con usted.",
        VerifiedFacts(),
        MessageKind.FALLBACK,
    )
    assert verdict.approved, verdict.findings


# ------------------------------------------------------------------ ordering
#
# ADR-22: the first rejecting stage wins, and the order is not arbitrary.


def test_provenance_is_reported_before_medical_advice() -> None:
    """A draft that both fabricates and advises reports one reason, the first
    stage's. A retry handed three codes at once tends to fix none."""
    verdict = validate(
        "Su cita es el 24 de diciembre. Tome 500 mg antes de venir.",
        FACTS,
        MessageKind.CONFIRMATION,
    )
    assert verdict.reason_codes == (Reason.FABRICATED_FACT.value,)


def test_findings_quote_the_offending_span() -> None:
    """ADR-23 feeds retries reason codes and quoted spans. A retry told only
    "fabricated fact" has nothing to correct."""
    verdict = validate("Su cita es el 24/10/2026.", FACTS, MessageKind.CONFIRMATION)
    assert verdict.findings[0].span == "24/10/2026"


def test_an_empty_fact_set_rejects_any_stated_fact() -> None:
    """With nothing verified, nothing may be stated. This is the guard that
    makes a tool failure fail closed rather than let a model improvise."""
    verdict = validate(
        "Su cita es el 23 de octubre a las 10:30 a.m.", VerifiedFacts(), MessageKind.CONFIRMATION
    )
    assert not verdict.approved


# ------------------------------------------------------- a phone is one fact


@pytest.mark.parametrize(
    "written",
    [
        "+57 601 555 0101",
        "601 555 0101",
        "601-555-0101",
        "6015550101",
    ],
)
def test_a_clinic_phone_resolves_however_it_is_written(written: str) -> None:
    """The country code was reported as a fabrication until a phone was
    matched as one token: "+57 601 555 0101" tokenised to four numbers and the
    "57" resolved against nothing, so a correct cancellation failed the gate.
    """
    facts = VerifiedFacts(clinic_phone="+57 601 555 0101")
    draft = f"Su cita fue cancelada. Llame al {written}."
    verdict = validate(draft, facts, MessageKind.CANCELLATION)
    assert verdict.approved, verdict.findings


def test_a_phone_that_is_not_the_clinics_is_still_a_fabrication() -> None:
    """The fix must not have widened the gate into accepting any number."""
    facts = VerifiedFacts(clinic_phone="+57 601 555 0101")
    verdict = validate(
        "Su cita fue cancelada. Llame al +57 601 555 9999.", facts, MessageKind.CANCELLATION
    )
    assert not verdict.approved
    assert Reason.FABRICATED_FACT.value in verdict.reason_codes
