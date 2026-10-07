"""Deterministic conversion of clinic values.

Two rules are being guarded, and they are opposites:

    ambiguous  -> REVIEW   a person decides, because the data cannot
    damaged    -> INVALID  the row is rejected, because the value is unrecoverable

Most of these tests assert a *refusal*. That is deliberate: every silent
conversion of an ambiguous value writes a plausible wrong record that nothing
downstream can detect, which is the failure the client was burned by before.
"""

from __future__ import annotations

import datetime as dt

import pytest

from src.onboarding import normalizers as n
from src.registry.models import DocumentType


# ------------------------------------------------------------- document type
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("CC", DocumentType.CC),
        ("C.C.", DocumentType.CC),
        ("cédula de ciudadanía", DocumentType.CC),
        ("ID Card", DocumentType.CC),  # the client's own file writes it in English
        ("Minor ID", DocumentType.TI),
        ("T.I.", DocumentType.TI),
        # PPT is what the card is called; PT is the RIPS code. Clinics write both.
        ("PPT", DocumentType.PT),
        ("PT", DocumentType.PT),
        ("PEP", DocumentType.PE),
    ],
)
def test_document_types_are_recognised_however_they_are_written(
    raw: str, expected: DocumentType
) -> None:
    outcome = n.document_type(raw)
    assert outcome.ok
    assert outcome.value is expected


def test_an_unknown_document_type_is_reviewed_never_defaulted() -> None:
    """Defaulting to CC pairs a real number with the wrong legal identity."""
    outcome = n.document_type("Carné")
    assert outcome.status is n.Status.REVIEW
    assert outcome.value is None


def test_de_is_not_matched_inside_another_label() -> None:
    """`DE` is a document code and also the preposition in every other label."""
    assert n.document_type("cedula de ciudadania").value is DocumentType.CC
    assert n.document_type("DE").value is DocumentType.DE


# ----------------------------------------------------------- document number
def test_a_document_number_keeps_its_digits() -> None:
    assert n.document_number("1045678901").value == "1045678901"
    # The Spanish thousands form loses nothing, so it is accepted.
    assert n.document_number("1.045.678.901").value == "1045678901"
    # Excel writing a whole number as a float loses nothing either.
    assert n.document_number("1045678901.0").value == "1045678901"


def test_scientific_notation_is_rejected_not_repaired() -> None:
    """`1.23457E+11` has lost its low-order digits; there is nothing to recover.

    Rounding it back produces a number that belongs to somebody else, so the row
    is rejected and the clinic is asked to re-export the column as text.
    """
    outcome = n.document_number("1.23457E+11")
    assert outcome.status is n.Status.INVALID
    assert "text" in outcome.message


def test_a_passport_keeps_its_letters() -> None:
    assert n.document_number("E0456789").value == "E0456789"


def test_a_number_too_short_to_be_a_document_is_rejected() -> None:
    """A cédula that lost its leading zeros is a different person's number."""
    assert n.document_number("123").status is n.Status.INVALID


# ------------------------------------------------------------------- phone
def test_colombian_mobiles_become_e164() -> None:
    assert n.phone("3001234567").value == "+573001234567"
    assert n.phone("+57 311 987 6543").value == "+573119876543"
    assert n.phone("(311) 987-6543").value == "+573119876543"


@pytest.mark.parametrize("raw", ["3067891234", "3098765432", "30987654"])
def test_an_unreachable_number_is_flagged_never_corrected(raw: str) -> None:
    """These three prefixes are unassigned, so the number reaches nobody.

    It is not repaired: the nearest valid number belongs to a stranger, and a
    reminder sent there discloses an appointment to the wrong person. The
    patient is still imported; only the number is flagged.
    """
    outcome = n.phone(raw)
    assert outcome.status is n.Status.REVIEW
    assert outcome.value is None


# ----------------------------------------------------------------- boolean
@pytest.mark.parametrize("raw", ["SI", "Sí", "sí", "X", "1", "TRUE", "yes"])
def test_affirmatives_are_read_as_true(raw: str) -> None:
    assert n.boolean(raw).value is True


@pytest.mark.parametrize("raw", ["NO", "no", "0", "FALSE"])
def test_negatives_are_read_as_false(raw: str) -> None:
    assert n.boolean(raw).value is False


@pytest.mark.parametrize("raw", ["", "N/A", "NA", "?"])
def test_an_unclear_boolean_is_reviewed_not_assumed_false(raw: str) -> None:
    """An empty attendance cell means "not recorded", not "did not attend".

    `NA` is worse: in Spanish clinic files it is either "no aplica" or
    "no asistió", which are opposite facts about the same patient.
    """
    assert n.boolean(raw).status is n.Status.REVIEW


# -------------------------------------------------------------------- names
@pytest.mark.parametrize(
    ("raw", "given", "family"),
    [
        ("Carlos Andrés Pérez Gómez", "Carlos Andrés", "Pérez Gómez"),
        ("María José Pérez Gómez", "María José", "Pérez Gómez"),
        ("Ana Gómez", "Ana", "Gómez"),
        # A particle belongs to the surname it introduces.
        ("Juan de la Cruz Pérez Gómez", "Juan de la Cruz", "Pérez Gómez"),
        # Three parts are decidable when the first two are a known pair.
        ("María José Pérez", "María José", "Pérez"),
    ],
)
def test_decidable_names_are_split(raw: str, given: str, family: str) -> None:
    outcome = n.split_full_name(raw)
    assert outcome.ok, outcome.message
    assert outcome.value == n.SplitName(given, family)


def test_a_three_part_name_is_refused_because_it_is_genuinely_ambiguous() -> None:
    """ "Carlos Pérez Gómez" and "Juan Carlos Pérez" have the same shape.

    One is a given name and two surnames; the other is two given names and one
    surname. Nothing in the text separates them, and Ley 2129 de 2021 lets
    parents choose the order of surnames, so no positional rule helps either.
    """
    outcome = n.split_full_name("Carlos Pérez Gómez")
    assert outcome.status is n.Status.REVIEW
    assert "confirm" in outcome.message.lower()


@pytest.mark.parametrize("raw", ["Ana", "Luis Carlos Vélez de Uribe Restrepo", ""])
def test_names_of_unsupported_shape_are_refused(raw: str) -> None:
    assert n.split_full_name(raw).status is n.Status.REVIEW


# -------------------------------------------------------------------- dates
def test_a_column_with_a_day_above_twelve_decides_the_whole_column() -> None:
    """The decision is made once per column, then applied strictly to every row."""
    assert n.detect_day_first(["03/04/1991", "15/10/2026"]) is n.DayFirst.DAY_FIRST
    assert n.detect_day_first(["03/04/1991", "10/15/2026"]) is n.DayFirst.MONTH_FIRST


def test_a_column_that_contradicts_itself_is_undecided() -> None:
    assert n.detect_day_first(["15/10/2026", "10/15/2026"]) is n.DayFirst.UNDECIDED


def test_a_fully_ambiguous_column_is_undecided_and_its_values_are_reviewed() -> None:
    """`03/04/1991` is 3 April or 4 March, and a birth date must not be guessed.

    Falling back to a Colombian locale would be a guess too: the file was
    written by an Excel whose locale we do not know.
    """
    assert n.detect_day_first(["03/04/1991", "05/06/1985"]) is n.DayFirst.UNDECIDED
    outcome = n.date("03/04/1991", order=n.DayFirst.UNDECIDED)
    assert outcome.status is n.Status.REVIEW


def test_dates_convert_under_a_decided_order() -> None:
    assert n.date("15/10/2026", order=n.DayFirst.DAY_FIRST).value == dt.date(2026, 10, 15)
    assert n.date("10/15/2026", order=n.DayFirst.MONTH_FIRST).value == dt.date(2026, 10, 15)
    assert n.date("2026-10-15", order=n.DayFirst.DAY_FIRST).value == dt.date(2026, 10, 15)
    # openpyxl hands back real date cells already converted, with a time part.
    assert n.date("1990-03-04 00:00:00", order=n.DayFirst.DAY_FIRST).value == dt.date(1990, 3, 4)


def test_an_impossible_date_is_rejected() -> None:
    assert n.date("31/02/2026", order=n.DayFirst.DAY_FIRST).status is n.Status.INVALID


def test_excel_serials_before_march_1900_are_rejected() -> None:
    """Excel's 1900 system contains a 29 February that never existed.

    Verified against openpyxl: serials 59 and 60 both convert to 1900-02-28, so
    two different dates collapse onto one. Anything in that range is refused
    rather than silently mapped to the wrong day.
    """
    assert n.date("60", order=n.DayFirst.DAY_FIRST).status is n.Status.INVALID
    assert n.date("59", order=n.DayFirst.DAY_FIRST).status is n.Status.INVALID
    # Past the broken range the 1900 system is reliable.
    assert n.date("61", order=n.DayFirst.DAY_FIRST).value == dt.date(1900, 3, 1)


def test_the_1904_epoch_is_supported() -> None:
    """Workbooks saved on a Mac count from 1904; mixing the two shifts by 1462 days."""
    assert n.date("0", order=n.DayFirst.DAY_FIRST, epoch_1904=True).value == dt.date(1904, 1, 1)


# -------------------------------------------------------------------- times
def test_twenty_four_hour_times_convert() -> None:
    assert n.time_of_day("07:00").value == dt.time(7, 0)
    assert n.time_of_day("16:45").value == dt.time(16, 45)


def test_the_spanish_twelve_hour_form_converts_including_odd_spaces() -> None:
    """es-CO Excel separates "a. m." with U+00A0 or U+202F, not a plain space.

    A parser that splits on an ordinary space fails on every time in the file,
    silently, because the value still looks like text.
    """
    assert n.time_of_day("8:30 a. m.").value == dt.time(8, 30)
    assert n.time_of_day("2:30 p. m.").value == dt.time(14, 30)
    assert n.time_of_day("8:30 a. m.").value == dt.time(8, 30)
    assert n.time_of_day("2:30 PM").value == dt.time(14, 30)


def test_midnight_and_noon_are_not_confused() -> None:
    assert n.time_of_day("12:00 a. m.").value == dt.time(0, 0)
    assert n.time_of_day("12:00 p. m.").value == dt.time(12, 0)


def test_a_time_stored_as_a_fraction_is_rounded_not_truncated() -> None:
    """0.354166666 of a day is 30599.99994 seconds.

    Truncating gives 08:29:59, a minute earlier than the clinic wrote, and the
    error is invisible because the result is still a valid time.
    """
    assert n.time_of_day("0.354166666").value == dt.time(8, 30)
    assert n.time_of_day("0.604166666").value == dt.time(14, 30)


def test_an_impossible_time_is_rejected() -> None:
    assert n.time_of_day("25:00").status is n.Status.INVALID


# ------------------------------------------------------------------- status
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Confirmada", "confirmed"),
        ("Atendida", "completed"),
        ("asistió", "completed"),
        ("no asistió", "no_show"),
        ("Inasistencia", "no_show"),
        ("Cancelada", "cancelled"),
        ("Reprogramada", "rescheduled"),
        ("Agendada", "scheduled"),
    ],
)
def test_spanish_statuses_map_to_the_canonical_set(raw: str, expected: str) -> None:
    outcome = n.appointment_status(raw)
    assert outcome.ok
    assert outcome.value == expected


@pytest.mark.parametrize("raw", ["Pendiente", "NA", "N/A"])
def test_a_status_with_two_meanings_is_reviewed(raw: str) -> None:
    """`pendiente` is both "not yet confirmed" and "on the waiting list".

    `NA` is either "no aplica" or "no asistió" — one says nothing about
    attendance, the other says the patient did not come.
    """
    assert n.appointment_status(raw).status is n.Status.REVIEW


# ------------------------------------------------------------------ general
def test_accents_are_stripped_for_matching_but_never_for_storage() -> None:
    """Unaccented spelling is the norm in real files, so matching must ignore it.

    Storage must not: "Muñoz" and "Munoz" are different names, and restoring an
    accent a file does not carry would be inventing data.
    """
    assert n.strip_accents("TELÉFONO") == "telefono"
    assert n.strip_accents("Muñoz") == "munoz"
    # The value itself survives untouched through a normalizer that stores it.
    assert n.split_full_name("Diana Muñoz").value.family_names == "Muñoz"


# ------------------------------------------------------- empty and unparseable
# Every normalizer is reached by a real clinic file containing blanks and
# typos. These paths decide whether a malformed cell stops the import or slips
# through as a wrong value, so each one is pinned even though none is exotic.
@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (n.document_type(""), n.Status.REVIEW),
        (n.document_number(""), n.Status.INVALID),
        (n.document_number("!!!"), n.Status.REVIEW),
        (n.phone(""), n.Status.REVIEW),
        (n.phone("no tiene"), n.Status.REVIEW),
        # A name ending in a connecting word is truncated, not a name.
        (n.split_full_name("Ana de"), n.Status.REVIEW),
        (n.date("", order=n.DayFirst.DAY_FIRST), n.Status.REVIEW),
        (n.date("pendiente", order=n.DayFirst.DAY_FIRST), n.Status.REVIEW),
        (n.date("2026-13-45", order=n.DayFirst.DAY_FIRST), n.Status.INVALID),
        (n.time_of_day(""), n.Status.REVIEW),
        # 13 has no meaning on a 12-hour clock.
        (n.time_of_day("13:00 a. m."), n.Status.INVALID),
        (n.time_of_day("por confirmar"), n.Status.REVIEW),
        # A day fraction must be below 1.0.
        (n.time_of_day("1.5"), n.Status.REVIEW),
        (n.appointment_status(""), n.Status.REVIEW),
        (n.appointment_status("zzz"), n.Status.REVIEW),
    ],
)
def test_empty_and_unparseable_values_never_convert(
    outcome: n.Outcome[object], expected: n.Status
) -> None:
    assert outcome.status is expected
    assert outcome.value is None
    # Every refusal names the rule that produced it, so the transform log can
    # tell a reviewer why the cell stopped (ADR-08a).
    assert outcome.rule
    assert outcome.message


def test_a_non_date_value_does_not_derail_the_column_decision() -> None:
    """Real date columns contain stray notes like "pendiente" or a blank.

    Those are skipped when deciding the column's order, so one junk cell cannot
    force an otherwise decidable column into manual review — nor can it decide
    a column on its own.
    """
    assert n.detect_day_first(["15/10/2026", "pendiente", "", "03/04/1991"]) is n.DayFirst.DAY_FIRST
    assert n.detect_day_first(["pendiente", "por confirmar"]) is n.DayFirst.UNDECIDED


# ------------------------------------------------------------------- weekday
# `weekday` is a required availability field that had no normalizer, so "lunes",
# "Wednesday" and "7" were all stored as text and all reported valid. The column
# is a SmallInteger with CHECK (weekday BETWEEN 0 AND 6).


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("lunes", 0),
        ("LUNES", 0),
        ("Miércoles", 2),
        ("miercoles", 2),  # the same day without its accent
        ("Wednesday", 2),
        ("mié", 2),
        ("domingo", 6),
        ("Sunday", 6),
    ],
)
def test_a_day_name_becomes_the_number_date_weekday_uses(raw: str, expected: int) -> None:
    """Monday is 0, matching `date.weekday()`, so no conversion is needed later."""
    outcome = n.weekday(raw)
    assert outcome.status is n.Status.VALID
    assert outcome.value == expected


@pytest.mark.parametrize("raw", ["1", "3", "7", "0"])
def test_a_numbered_day_is_asked_about_never_guessed(raw: str) -> None:
    """1 is Monday under ISO-8601 and Sunday to someone counting from Sunday.

    Nothing in a single cell says which, and choosing wrong moves a doctor's
    whole working week by a day. 7 is refused for the same reason plus a second
    one: `date.weekday()` has no 7, so a file with both 0 and 7 is using two
    conventions at once.
    """
    outcome = n.weekday(raw)
    assert outcome.status is n.Status.REVIEW
    assert "ambiguous" in outcome.message


def test_an_unrecognised_day_is_refused() -> None:
    assert n.weekday("xyz").status is n.Status.REVIEW


def test_an_empty_day_is_refused() -> None:
    assert n.weekday("   ").status is n.Status.REVIEW


def test_every_required_field_that_needs_conversion_has_a_normalizer() -> None:
    """A field with no normalizer is stored as whatever the file said.

    `weekday` was required, had no normalizer, and so "lunes", "Wednesday" and
    "7" were all reported valid for a SmallInteger column constrained to 0-6.
    Testing the normalizer in isolation does not catch that: the binding in
    `canonical.py` is what makes it run, so the binding is what is asserted.

    Fields legitimately stored as written are listed explicitly, so adding a new
    one is a deliberate decision rather than an omission.
    """
    from src.onboarding.canonical import ALL_FIELDS

    stored_as_written = {
        # Free text the clinic owns; there is nothing to convert or refuse.
        "given_names",
        "family_names",
        "specialty",
        "name",
        "office_number",
        "eps",
        "external_ref",
        "secondary_contact_name",
        "telegram_chat_id",
        "cancellation_reason",
        "consultation_type",
        "source",
        "slot_ref",
        # A reference resolved against other rows, not converted.
        "doctor_ref",
        "patient_document",
        # Stored encrypted and only ever masked for display; nothing in M1 sends
        # to it. When dispatch arrives (M4) this needs a normalizer, because an
        # address that cannot receive is a silently failed notification.
        "email",
        # A doctor's name is stored as written: the given-name/surname split
        # exists for patients, who are matched on it. Doctors are matched on the
        # clinic's own code.
        "full_name",
    }

    unbound = sorted(
        f.name for f in ALL_FIELDS if f.normalizer is None and f.name not in stored_as_written
    )
    assert unbound == [], f"these fields convert nothing: {unbound}"


# ------------------------------------- found by an adversarial review
# All three were silent or fatal on input a Colombian clinic really exports.


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("19900315", dt.date(1990, 3, 15)),
        ("20260101", dt.date(2026, 1, 1)),
    ],
)
def test_a_compact_yyyymmdd_date_is_read_as_a_date(raw: str, expected: dt.date) -> None:
    """RIPS and many Colombian systems export birth dates as yyyymmdd.

    Eight digits were treated as an Excel day count, so 19900315 was added to
    1899-12-30 and raised OverflowError -- which reached the person uploading as
    "Internal server error" and refused the whole file.
    """
    outcome = n.date(raw, order=n.DayFirst.UNDECIDED)
    assert outcome.status is n.Status.VALID
    assert outcome.value == expected


@pytest.mark.parametrize("raw", ["19901332", "99999999", "45678901"])
def test_eight_digits_that_are_not_a_date_are_refused_not_crashed(raw: str) -> None:
    """A refusal a receptionist can read, never an unhandled exception."""
    assert n.date(raw, order=n.DayFirst.UNDECIDED).status is n.Status.REVIEW


def test_a_serial_too_large_to_be_a_date_is_refused() -> None:
    """Excel's own last date is serial 2958465; beyond it there is no date.

    Unbounded, the addition raised OverflowError instead of returning something
    a reviewer could act on.
    """
    outcome = n.date("2958466", order=n.DayFirst.UNDECIDED)
    assert outcome.status is n.Status.REVIEW
    assert "too large" in outcome.message


def test_surnames_first_with_a_comma_is_not_swapped() -> None:
    """ "PEREZ GOMEZ, CARLOS ANDRES" is surnames first, and the comma says so.

    Read left to right it stored the surnames as given names and the given names
    as surnames, and marked the row valid -- a patient filed under the wrong
    identity with nothing to notice. This is the one name shape that IS
    decidable, because the comma is an explicit signal rather than a guess.
    """
    outcome = n.split_full_name("PEREZ GOMEZ, CARLOS ANDRES")
    assert outcome.status is n.Status.VALID
    assert outcome.value is not None
    assert outcome.value.given_names == "CARLOS ANDRES"
    assert outcome.value.family_names == "PEREZ GOMEZ"


def test_two_tokens_either_side_of_a_comma_still_split_on_the_comma() -> None:
    outcome = n.split_full_name("GOMEZ, ANA")
    assert outcome.status is n.Status.VALID
    assert outcome.value is not None
    assert outcome.value.given_names == "ANA"
    assert outcome.value.family_names == "GOMEZ"


@pytest.mark.parametrize("raw", ["Perez Gomez,", ", Ana", "A, B, C"])
def test_a_name_whose_comma_decides_nothing_is_refused(raw: str) -> None:
    """One side missing, or more than one comma, decides nothing."""
    assert n.split_full_name(raw).status is n.Status.REVIEW


def test_a_name_without_a_comma_keeps_its_previous_reading() -> None:
    """The comma rule must not change the shapes that already worked."""
    four = n.split_full_name("Carlos Andres Perez Gomez")
    assert four.status is n.Status.VALID
    assert four.value is not None
    assert four.value.given_names == "Carlos Andres"

    # Three tokens stay undecidable: no comma, no rule.
    assert n.split_full_name("Carlos Perez Gomez").status is n.Status.REVIEW


# ------------------------- a digit `int()` cannot parse (second review pass)
# `str.isdigit()` is true for superscripts, circled numerals, and fullwidth and
# Arabic-Indic digits. None of them belongs in a cédula or a date, and where
# `int()` followed the request died with a 500.


@pytest.mark.parametrize("raw", ["²", "³", "①", "１２３", "٣٣٣"])
def test_a_digit_int_cannot_parse_is_refused_not_crashed(raw: str) -> None:
    """A stray character in a date column answered the upload with a 500."""
    outcome = n.date(raw, order=n.DayFirst.DAY_FIRST)
    assert outcome.status in (n.Status.REVIEW, n.Status.INVALID)


@pytest.mark.parametrize("raw", ["²²²²", "１０２０３０４０５０", "٣٣٣٣٣"])
def test_a_cedula_of_non_ascii_digits_is_refused(raw: str) -> None:
    """Stored verbatim, these were a value nobody typed.

    A cédula is ASCII digits. `isdigit()` accepted characters that look like
    digits and reported them valid, which is the "never guess" rule broken by a
    standard-library surprise.
    """
    assert n.document_number(raw).status is n.Status.REVIEW


@pytest.mark.parametrize("raw", ["1020304050", "71234567", "1020304050.0"])
def test_a_real_cedula_still_passes(raw: str) -> None:
    """The narrower test must not refuse what clinics actually send."""
    assert n.document_number(raw).status is n.Status.VALID


# --------------------------------- a date bounded by what its field means
# `date()` accepted year 1 and year 9999. The bound cannot live in the
# normalizer: the same function serves a birth date and an appointment date,
# which disagree about whether next year is wrong. So the range belongs to the
# field, and an out-of-range date goes to review -- "1890" may be a typo for
# 1980 and a human can see which.

_TODAY = dt.date(2026, 10, 2)


@pytest.mark.parametrize(
    "value",
    [dt.date(1, 1, 1), dt.date(9999, 12, 31), dt.date(1890, 5, 5), dt.date(2030, 1, 1)],
)
def test_an_implausible_birth_date_is_questioned(value: dt.date) -> None:
    assert n.plausible_date(value, "birth_date", today=_TODAY) is not None


@pytest.mark.parametrize("value", [dt.date(1952, 10, 5), dt.date(2005, 6, 1), dt.date(1915, 1, 1)])
def test_a_real_birth_date_passes(value: dt.date) -> None:
    """130 years admits every living patient; the oldest verified are about 115."""
    assert n.plausible_date(value, "birth_date", today=_TODAY) is None


def test_an_appointment_next_year_is_not_questioned() -> None:
    """The same range on every date field would refuse ordinary scheduling."""
    assert n.plausible_date(dt.date(2027, 3, 1), "appointment_date", today=_TODAY) is None
    # ...while one from 1990 is as wrong as a birth date in 2030.
    assert n.plausible_date(dt.date(1990, 1, 1), "appointment_date", today=_TODAY) is not None


def test_a_field_with_no_range_is_never_questioned() -> None:
    assert n.plausible_date(dt.date(1, 1, 1), "some_other_field", today=_TODAY) is None


# ------------------------------------------- a name is the one free-text field
# Every other field rejects formula text by shape. A name does not, so
# `=HYPERLINK("http://evil.example","x")` was stored as a patient's given names:
# inert in the database, executable the moment anyone exports or opens the sheet.


@pytest.mark.parametrize(
    "value",
    [
        '=HYPERLINK("http://evil.example","x")',
        "+cmd|' /C calc'!A0",
        "@SUM(1)",
        "-2+3",
    ],
)
def test_a_name_that_a_spreadsheet_would_run_is_questioned(value: str) -> None:
    outcome = n.split_full_name(value)
    assert outcome.status is n.Status.REVIEW
    assert outcome.rule == "name.looks_like_a_formula"


@pytest.mark.parametrize(
    "value", ["Ana Perez", "Jean-Luc Picard", "O'Brien Smith", "PEREZ GOMEZ, CARLOS ANDRES"]
)
def test_a_real_name_is_not_mistaken_for_a_formula(value: str) -> None:
    """The guard is on the leading character only; a hyphen inside a name is fine."""
    assert n.split_full_name(value).status is n.Status.VALID


def test_formula_text_is_refused_in_fields_that_have_no_normalizer() -> None:
    """The guard belonged to more than one field.

    `split_full_name` refused a formula, but given_names, family_names, a
    doctor's full_name and a specialty's name have no normalizer and were stored
    as written, so the same payload reached the database through any of them.
    """
    from src.onboarding import canonical, service
    from src.onboarding.canonical import Entity

    payload = '=HYPERLINK("http://evil.example","Ana")'
    for entity, field_name in (
        (Entity.PATIENT, "given_names"),
        (Entity.PATIENT, "family_names"),
        (Entity.DOCTOR, "full_name"),
        (Entity.SPECIALTY, "name"),
    ):
        field = canonical.field_for(entity, field_name)
        assert field is not None, f"{entity.value}.{field_name} is gone"
        outcome = service._apply(field, payload, n.DayFirst.UNDECIDED)
        assert outcome.status is n.Status.REVIEW, f"{entity.value}.{field_name} accepted a formula"


def test_an_ordinary_value_in_those_fields_still_imports() -> None:
    from src.onboarding import canonical, service
    from src.onboarding.canonical import Entity

    for entity, field_name, value in (
        (Entity.PATIENT, "given_names", "Ana María"),
        (Entity.DOCTOR, "full_name", "Luis Pérez Gómez"),
        (Entity.SPECIALTY, "name", "Cardiología"),
    ):
        field = canonical.field_for(entity, field_name)
        assert field is not None
        assert service._apply(field, value, n.DayFirst.UNDECIDED).status is n.Status.VALID


def test_every_document_code_maps_to_itself_however_it_is_punctuated() -> None:
    """A table of hand-written aliases is one typo away from a wrong identity.

    "c.c." mapped to TI: a cédula de ciudadanía read as a tarjeta de identidad,
    which is the commonest document type in Colombia and the exact spelling the
    receptionist fixture uses. It pairs a real number with the wrong legal
    identity -- the failure CLAUDE.md section 11 names for defaulting an unknown
    type, arriving instead through a mis-keyed alias.

    Every code must therefore resolve to itself, plain and dotted, and this is
    the check that would have caught it.
    """
    from src.registry.models import DocumentType

    for code in DocumentType:
        plain = n.document_type(code.value)
        assert plain.status is n.Status.VALID, f"{code.value} is not recognised at all"
        assert plain.value is code, f"{code.value!r} maps to {plain.value}"

        dotted = n.document_type(".".join(code.value) + ".")
        if dotted.status is n.Status.VALID:
            assert dotted.value is code, (
                f"{'.'.join(code.value)}. maps to {dotted.value}, not {code}"
            )
