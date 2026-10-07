"""Proposing which canonical field each column holds.

The matcher's job is to be right without a model wherever it can, and to say
"I don't know" rather than guess wherever it cannot. A wrong-but-confident
proposal is the worst outcome: it is pre-ticked on the confirmation screen, so a
reviewer skimming a long list approves it, and a column of birth dates lands in
the appointment date field.
"""

from __future__ import annotations

import pytest

from src.onboarding.canonical import Entity
from src.onboarding.matcher import (
    Confidence,
    guess_entity,
    match_column,
    match_sheet,
    normalize_header,
)


# ------------------------------------------------------------ header cleanup
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Número Documento", "numero documento"),
        ("TELÉFONO", "telefono"),
        # Underscores separate words in exported headers; `\w` keeps them, so
        # without explicit handling "id_type" never matches "id type".
        ("id_type", "id type"),
        ("doctor_id", "doctor id"),
        ("No. Documento", "no documento"),
        ("  Fecha   Cita  ", "fecha cita"),
    ],
)
def test_headers_are_reduced_to_comparable_words(raw: str, expected: str) -> None:
    assert normalize_header(raw) == expected


# -------------------------------------------------------------- exact match
@pytest.mark.parametrize(
    ("header", "entity", "field"),
    [
        ("Tipo Documento", Entity.PATIENT, "document_type"),
        ("Número Documento", Entity.PATIENT, "document_number"),
        ("Celular", Entity.PATIENT, "phone_e164"),
        ("Correo Electrónico", Entity.PATIENT, "email"),
        ("EPS", Entity.PATIENT, "eps"),
        ("Apellidos", Entity.PATIENT, "family_names"),
        # The client's own file is in English with underscores.
        ("id_type", Entity.PATIENT, "document_type"),
        ("telegram_chat_id", Entity.PATIENT, "telegram_chat_id"),
        ("doctor_name", Entity.DOCTOR, "full_name"),
        ("office_number", Entity.DOCTOR, "office_number"),
        ("day_of_week", Entity.AVAILABILITY, "weekday"),
        ("slot_id", Entity.APPOINTMENT, "slot_ref"),
    ],
)
def test_known_headings_map_exactly(header: str, entity: Entity, field: str) -> None:
    proposal = match_column(header, entity)
    assert proposal.field is not None
    assert proposal.field.name == field
    assert proposal.auto


def test_spanish_and_english_reach_the_same_field() -> None:
    """Clinic files arrive in both, sometimes in the same workbook."""
    for header in ("Celular", "cel", "WhatsApp", "teléfono celular"):
        assert match_column(header, Entity.PATIENT).field is not None


# ------------------------------------------------------- the date guard
def test_a_birth_date_is_never_taken_for_an_appointment_date() -> None:
    """Both are "fecha", and confusing them corrupts a clinical record.

    A birth date imported as an appointment date creates an appointment in
    1985; an appointment date imported as a birth date makes every patient a
    newborn. The heading's own words decide it before any other stage looks.
    """
    birth = match_column("Fecha de Nacimiento", Entity.PATIENT)
    assert birth.field is not None
    assert birth.field.name == "birth_date"

    appointment = match_column("Fecha de la Cita", Entity.APPOINTMENT)
    assert appointment.field is not None
    assert appointment.field.name == "appointment_date"


def test_an_unqualified_date_heading_is_not_forced_into_either() -> None:
    """A bare "Fecha" on a patient sheet could be either; it is not assumed."""
    proposal = match_column("Fecha", Entity.PATIENT)
    assert proposal.field is None or not proposal.auto


# --------------------------------------------------------- misleading headers
def test_a_header_that_merely_contains_an_alias_is_not_matched_to_it() -> None:
    """ "rescheduled_from_appointment_id" contains "appointment id".

    Taking it for the appointment's own code would link every rescheduled
    appointment to the wrong row. The alias has to account for most of the
    heading, not appear somewhere inside it.
    """
    proposal = match_column("rescheduled_from_appointment_id", Entity.APPOINTMENT)
    assert proposal.field is None


def test_a_column_with_no_canonical_field_is_never_auto_applied() -> None:
    """Proposing nothing beats proposing something wrong.

    `consultation_result` is one word from `consultation_type` and means
    something else: the outcome of the visit, not its kind. The matcher may
    still offer it as a suggestion, but never pre-ticked, so a reviewer is asked
    rather than told.
    """
    for header in ("preferred_channel", "consultation_result", "lunch_break"):
        proposal = match_column(header, Entity.APPOINTMENT)
        assert not proposal.auto, f"{header} would be applied without review"


def test_an_unlabelled_column_is_not_matched() -> None:
    assert match_column("", Entity.PATIENT).field is None
    assert match_column("Unnamed: 0", Entity.PATIENT).field is None


# --------------------------------------------------------------- fuzzy match
@pytest.mark.parametrize(
    ("header", "field"),
    [
        ("identificacin", "document_number"),  # missing letter
        ("telefno celular", "phone_e164"),  # missing letter
        ("Núm. Documento", "document_number"),  # abbreviation and punctuation
    ],
)
def test_misspelled_headings_still_match(header: str, field: str) -> None:
    """Hand-made files contain typos, and a typo is not a reason to give up."""
    proposal = match_column(header, Entity.PATIENT)
    assert proposal.field is not None
    assert proposal.field.name == field


def test_nonsense_matches_nothing() -> None:
    assert match_column("zzzz qqqq", Entity.PATIENT).field is None


# -------------------------------------------------------------- whole sheets
def test_two_columns_are_never_proposed_for_one_field() -> None:
    """One would silently overwrite the other on import."""
    mapping = match_sheet(("Celular", "Teléfono", "Correo"), Entity.PATIENT)
    assigned = [p.field.name for p in mapping.proposals if p.field]
    assert len(assigned) == len(set(assigned))


def test_a_spanish_patient_sheet_maps_with_no_model_call() -> None:
    """The control case: ordinary Spanish headings resolve entirely offline.

    This is what keeps a typical import free of provider calls, and therefore
    free of any question about what left the machine.
    """
    headers = (
        "Tipo Documento",
        "Número Documento",
        "Nombres",
        "Apellidos",
        "Fecha Nacimiento",
        "Celular",
        "Correo Electrónico",
        "EPS",
    )
    mapping = match_sheet(headers, Entity.PATIENT)
    assert all(p.auto for p in mapping.proposals)
    assert mapping.missing_required == ()


def test_separate_name_columns_satisfy_the_name_requirement() -> None:
    """A file may give one name column or two; neither is required alone."""
    split = match_sheet(("Tipo Documento", "Documento", "Nombres", "Apellidos"), Entity.PATIENT)
    assert "full_name" not in split.missing_required

    single = match_sheet(("Tipo Documento", "Documento", "Nombre Completo"), Entity.PATIENT)
    assert "given_names" not in single.missing_required


def test_a_missing_required_field_is_reported() -> None:
    """The import cannot proceed until a human supplies it."""
    mapping = match_sheet(("Nombres", "Apellidos"), Entity.PATIENT)
    assert "document_number" in mapping.missing_required


# ------------------------------------------------------------- sheet purpose
@pytest.mark.parametrize(
    ("sheet", "entity"),
    [
        ("Pacientes", Entity.PATIENT),
        ("Patients", Entity.PATIENT),
        ("Medicos", Entity.DOCTOR),
        ("Doctors", Entity.DOCTOR),
        ("Citas", Entity.APPOINTMENT),
        ("Specialties", Entity.SPECIALTY),
        # Both of these contain a more general word as well, and the longest
        # match must win: "Doctor_Availability" is not a list of doctors, and
        # "Outpatient_Appointments" is not a list of patients.
        ("Doctor_Availability", Entity.AVAILABILITY),
        ("Outpatient_Appointments", Entity.APPOINTMENT),
    ],
)
def test_a_sheets_purpose_is_read_from_its_name(sheet: str, entity: Entity) -> None:
    guessed, reason = guess_entity(sheet, ("a", "b"))
    assert guessed is entity
    assert reason


def test_an_unhelpfully_named_sheet_is_judged_by_its_columns() -> None:
    """ "Hoja1" says nothing, so the headings have to decide."""
    guessed, reason = guess_entity(
        "Hoja1", ("Tipo Documento", "Número Documento", "Celular", "EPS")
    )
    assert guessed is Entity.PATIENT
    assert "%" in reason


# --------------------------------------------------- RIPS-shaped exports
# RIPS archivo US and Resolución 1036 de 2022 both split a name into four
# fields, and every IPS must emit them to bill. A clinic exporting in the shape
# the government asked for is therefore the normal case, not an exotic one.


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("tipoDocumentoIdentificacion", "document_type"),
        ("numDocumentoIdentificacion", "document_number"),
        ("primerNombre", "given_names"),
        ("segundoNombre", "given_names"),
        ("primerApellido", "family_names"),
        ("segundoApellido", "family_names"),
        ("fechaNacimiento", "birth_date"),
    ],
)
def test_rips_camel_case_headers_are_recognised(header: str, expected: str) -> None:
    """camelCase was collapsing to one word, so these matched nothing at all.

    `normalize_header` casefolds, which destroys the only boundary in
    `primerNombre`. Splitting before that is what makes a government-shaped
    export readable.
    """
    proposal = match_column(header, Entity.PATIENT)
    assert proposal.field is not None, f"{header} matched nothing"
    assert proposal.field.name == expected


def test_a_one_word_product_name_still_matches_after_the_split() -> None:
    """Splitting camelCase also splits "WhatsApp" into "whats app".

    Fixed by keeping every alias in both spellings rather than a list of brand
    names that would need maintaining forever. This test exists because the
    camelCase change broke it.
    """
    proposal = match_column("WhatsApp", Entity.PATIENT)
    assert proposal.field is not None
    assert proposal.field.name == "phone_e164"


def test_four_name_columns_all_reach_the_two_name_fields() -> None:
    """Four columns for two fields is not two columns competing for one.

    Every other field keeps the rule that a second column would silently
    overwrite the first; the name parts are joined instead, because that is what
    they are.
    """
    headers = (
        "primerNombre",
        "segundoNombre",
        "primerApellido",
        "segundoApellido",
    )
    mapping = match_sheet(headers, Entity.PATIENT)
    targets = {p.column: (p.field.name if p.field else None) for p in mapping.proposals}

    assert targets["primerNombre"] == "given_names"
    assert targets["segundoNombre"] == "given_names"
    assert targets["primerApellido"] == "family_names"
    assert targets["segundoApellido"] == "family_names"


def test_the_rips_document_number_is_confident_enough_to_pre_tick() -> None:
    """A weak match is not pre-ticked, and an unticked document number is fatal.

    `_default_mapping` only pre-ticks confident proposals, and `_apply_patients`
    skips any row with no document number. So a fuzzy-only match on
    `numDocumentoIdentificacion` means a RIPS export validates, commits, reports
    success and writes nobody. The alias is what makes it exact.
    """
    proposal = match_column("numDocumentoIdentificacion", Entity.PATIENT)
    assert proposal.field is not None
    assert proposal.field.name == "document_number"
    assert proposal.confidence is Confidence.EXACT


def test_an_entity_guess_with_no_evidence_says_so() -> None:
    """A file with positional headings matches nothing, and must admit it.

    `max` over an all-zero tie picks arbitrarily, and the message read "0% of
    the columns match specialty fields" while reading the sheet as specialties —
    a conclusion the evidence does not support, stated in the reviewer's own
    words. Patients are what a clinic imports, so that is the default, and the
    reason now says it is a guess.
    """
    entity, reason = guess_entity("export", ("column 1", "column 2", "column 3"))

    assert entity is Entity.PATIENT
    assert "guess" in reason.lower()
    assert "0%" not in reason
