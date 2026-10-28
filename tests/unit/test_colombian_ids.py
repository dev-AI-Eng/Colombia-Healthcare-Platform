"""A document type has to fit the age beside it.

Colombia ties the identity document to age: Registro Civil to 6 years 11 months,
Tarjeta de Identidad from 7 to 17 years 364 days, Cédula de Ciudadanía from 18.
The client asked for one shared check run wherever a document type and a birth
date arrive together, so these tests cover the rule itself; the importer wiring
is covered at the bottom, through a real sheet.

The boundaries are the point. A rule that is right for a four-year-old and a
forty-year-old and wrong on the day someone turns seven would refuse correct
records for exactly one day a year each, which is the kind of defect nobody
reports and everybody works around.
"""

from __future__ import annotations

import datetime as dt

import pytest

from src.core.colombian_ids import (
    MAJORITY_AGE,
    TI_MIN_AGE,
    age_on,
    check_document_for_age,
)
from src.onboarding import service
from src.onboarding.canonical import Entity
from src.onboarding.reader import Sheet

TODAY = dt.date(2026, 10, 28)


def _born(years: int, months: int = 0, days: int = 0) -> dt.date:
    """A birth date that makes someone exactly this old on TODAY.

    `days` counts *after* the birthday, so `_born(7)` is the seventh birthday
    itself and `_born(7, days=-1)` is one day short of it.
    """
    month = TODAY.month - months
    year = TODAY.year - years
    while month < 1:
        month += 12
        year -= 1
    return dt.date(year, month, TODAY.day) - dt.timedelta(days=days)


# ------------------------------------------------------------------- the ladder


@pytest.mark.parametrize(
    ("document_type", "born", "fits"),
    [
        # Registro Civil: from birth to 6 years 11 months.
        ("RC", _born(0), True),
        ("RC", _born(4), True),
        ("RC", _born(6, months=11), True),
        ("RC", _born(7, days=-1), True),  # the day before the seventh birthday
        ("RC", _born(7), False),  # the seventh birthday itself
        ("RC", _born(9), False),
        # Tarjeta de Identidad: from 7 to 17 years 364 days.
        ("TI", _born(7, days=-1), False),
        ("TI", _born(7), True),
        ("TI", _born(12), True),
        ("TI", _born(18, days=-1), True),  # the day before the eighteenth birthday
        ("TI", _born(18), False),  # lapses automatically at 18
        ("TI", _born(30), False),
        # Cédula de Ciudadanía: from 18.
        ("CC", _born(17), False),
        ("CC", _born(18, days=-1), False),
        ("CC", _born(18), True),
        ("CC", _born(65), True),
    ],
)
def test_the_ladder_is_exact_on_both_sides_of_every_birthday(
    document_type: str, born: dt.date, fits: bool
) -> None:
    mismatch = check_document_for_age(document_type, born, TODAY)
    assert (mismatch is None) is fits, (
        f"{document_type} at age {age_on(born, TODAY)} (born {born}): "
        f"expected {'fits' if fits else 'mismatch'}, got {mismatch}"
    )


def test_the_constants_match_the_clients_ladder() -> None:
    """Pinned so a change to either age is a deliberate edit of a rule the
    client wrote down, not a drive-by."""
    assert TI_MIN_AGE == 7
    assert MAJORITY_AGE == 18


@pytest.mark.parametrize(
    "document_type", ["CE", "PA", "PT", "CD", "SC", "PE", "DE", "CN", "AS", "MS"]
)
@pytest.mark.parametrize("years", [0, 5, 12, 17, 40, 90])
def test_documents_outside_the_ladder_carry_no_age_rule(document_type: str, years: int) -> None:
    """Foreign residents hold CE, PA or PPT at any age, and the client described
    the ladder and nothing else. An age window invented for a type nobody
    described would refuse real patients -- a child on a passport is ordinary."""
    assert check_document_for_age(document_type, _born(years), TODAY) is None


def test_the_type_is_matched_however_the_clinic_cased_it() -> None:
    """Clinics write both cc and CC, and Excel leaves trailing spaces."""
    assert check_document_for_age(" cc ", _born(9), TODAY) is not None
    assert check_document_for_age("ti", _born(30), TODAY) is not None


# --------------------------------------------------------------- the mismatches


@pytest.mark.parametrize(
    ("document_type", "born", "rule"),
    [
        ("RC", _born(9), "document.age.rc_outgrown"),
        ("TI", _born(4), "document.age.ti_too_young"),
        ("TI", _born(25), "document.age.ti_expired"),
        ("CC", _born(12), "document.age.cc_under_majority"),
    ],
)
def test_each_mismatch_has_its_own_stable_rule_name(
    document_type: str, born: dt.date, rule: str
) -> None:
    mismatch = check_document_for_age(document_type, born, TODAY)
    assert mismatch is not None
    assert mismatch.rule == rule


def test_a_stale_record_and_a_mistyped_one_are_told_apart() -> None:
    """The two shapes call for different checks, and the message has to say
    which. An RC on a nine-year-old is a record that went out of date -- the
    document is probably now a TI. A CC on a twelve-year-old cannot be a stale
    record, because nobody holds a cédula at twelve: the type or the birth date
    was mistyped, and nothing in the row says which."""
    stale = check_document_for_age("RC", _born(9), TODAY)
    mistyped = check_document_for_age("CC", _born(12), TODAY)
    assert stale is not None and mistyped is not None
    assert "out of date" in stale.message
    assert "either the document type or the birth date is wrong" in mistyped.message.lower()


def test_the_message_names_the_age_it_found() -> None:
    mismatch = check_document_for_age("RC", _born(9), TODAY)
    assert mismatch is not None
    assert "is 9" in mismatch.message


# ------------------------------------------------------------------- the dates


def test_the_birthday_itself_counts_as_the_new_age() -> None:
    assert age_on(dt.date(2019, 10, 28), TODAY) == 7
    assert age_on(dt.date(2019, 10, 29), TODAY) == 6


def test_a_leap_day_birth_turns_a_year_older_on_the_first_of_march() -> None:
    """Born 29 February: in a common year the birthday falls on 1 March, which
    the month-and-day comparison gives without special handling. Pinned because
    the failure -- an eighteenth birthday arriving a day early -- would put a
    real minor on a cédula."""
    born = dt.date(2008, 2, 29)
    assert age_on(born, dt.date(2026, 2, 28)) == 17
    assert age_on(born, dt.date(2026, 3, 1)) == 18


def test_a_birth_date_in_the_future_is_not_this_modules_to_judge() -> None:
    """The date normalizer's `DATE_RANGES` owns that. Reporting it here too
    would blame the document type for a date that is simply wrong."""
    assert check_document_for_age("CC", TODAY + dt.timedelta(days=30), TODAY) is None


def test_the_check_never_changes_what_it_was_given() -> None:
    """CLAUDE.md §7: never repair a document. The function returns a finding
    and has no way to return a corrected type -- asserted so that adding one
    later is a visible decision."""
    mismatch = check_document_for_age("RC", _born(9), TODAY)
    assert mismatch is not None
    assert set(mismatch.__slots__) == {"rule", "message"}  # type: ignore[attr-defined]


# ---------------------------------------------------------- through the importer


def _sheet(*rows: tuple[str, ...]) -> Sheet:
    return Sheet(
        name="Pacientes",
        headers=("TIPO DOC", "IDENTIFICACION", "NOMBRES", "FECHA NACIMIENTO"),
        rows=tuple(rows),
        header_row=1,
    )


_MAPPING = {
    "TIPO DOC": "document_type",
    "IDENTIFICACION": "document_number",
    "NOMBRES": "given_names",
    "FECHA NACIMIENTO": "birth_date",
}


def _validate(sheet: Sheet) -> list[service.RowResult]:
    rows, _ = service.validate(sheet, Entity.PATIENT, _MAPPING, decisions={}, today=TODAY)
    return rows


def test_a_stale_registro_civil_goes_to_review_through_a_real_sheet() -> None:
    """End to end: the row converts, every cell is valid, and the row as a whole
    still asks a person -- because the pair cannot both be right."""
    rows = _validate(_sheet(("RC", "1020304050", "Sofía", "2017-03-15")))  # 9 years old
    row = rows[0]

    assert row.status.value == "review"
    assert row.errors == []
    assert len(row.reviews) == 1
    # Named by the clinic's own heading, as every other refusal is.
    assert row.reviews[0].startswith("TIPO DOC:")
    assert "Registro Civil" in row.reviews[0]


def test_the_review_does_not_alter_the_values_that_will_be_imported() -> None:
    """The type stays RC and the number stays the number. A reviewer decides;
    the importer does not promote it to a TI on their behalf."""
    row = _validate(_sheet(("RC", "1020304050", "Sofía", "2017-03-15")))[0]
    assert str(row.values["document_type"]) == "RC"
    assert row.values["document_number"] == "1020304050"


def test_a_matching_pair_is_left_alone() -> None:
    rows = _validate(
        _sheet(
            ("RC", "1020304050", "Sofía", "2023-01-10"),
            ("TI", "1020304051", "Mateo", "2014-06-02"),
            ("CC", "1020304052", "Laura", "1980-09-20"),
        )
    )
    assert [r.status.value for r in rows] == ["valid", "valid", "valid"]


def test_a_foreign_document_on_a_child_is_left_alone() -> None:
    row = _validate(_sheet(("PA", "AB1234567", "Lucas", "2018-05-05")))[0]
    assert row.status.value == "valid"


def test_a_missing_birth_date_is_nothing_to_check_against() -> None:
    """The field is optional. A patient with no birth date is not a mismatch."""
    row = _validate(_sheet(("CC", "1020304050", "Ana", "")))[0]
    assert row.reviews == []


def test_an_unrecognised_document_type_is_left_to_its_own_rule() -> None:
    """It already goes to review for being unrecognised, and there is no value
    to compare. Reporting it again as an age problem would be noise."""
    row = _validate(_sheet(("XYZ", "1020304050", "Ana", "2017-03-15")))[0]
    assert all("Registro Civil" not in note for note in row.reviews)
    assert all("Cédula" not in note for note in row.reviews)


def test_only_patients_are_checked() -> None:
    row = service.RowResult(
        row_number=2,
        entity=Entity.DOCTOR,
        raw={},
        values={"document_type": "RC", "birth_date": _born(40)},
    )
    assert service.document_age_review(row, TODAY) is None
