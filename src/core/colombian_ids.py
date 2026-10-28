"""Which identity document a person of a given age can hold, in Colombia.

The document type is tied to age, not chosen by the clinic:

    Registro Civil (RC)            from birth to 6 years 11 months
    Tarjeta de Identidad (TI)      from 7 to 17 years 364 days; lapses at 18
    Cédula de Ciudadanía (CC)      from 18

Foreign residents hold CE, PA or PPT instead, outside this ladder, so those
types carry no age rule at all.

WHAT THIS DOES, AND WHAT IT DELIBERATELY DOES NOT
-------------------------------------------------
It reports that a document type and a birth date cannot both be right. It never
says which one is wrong, and nothing here changes either value.

That restraint is CLAUDE.md §7, not caution: "Never repair a document number or
a phone: the repaired value belongs to somebody else". An RC on a nine-year-old
is the ordinary case -- the clinic registered the child at four and never
updated the record -- and the right answer is a person looking at the document,
not a function promoting the type to TI. A CC on a twelve-year-old is the other
shape: either the type or the birth date was mistyped, and nothing in the row
says which.

It takes plain two-letter codes rather than the `DocumentType` enum on purpose.
`core` imports nothing from the contexts, and this has two callers that live in
different ones -- the spreadsheet importer, and the conversational agent once it
registers a patient -- so the shared rule has to sit below both.

THE BOUNDARY AGES
-----------------
Whole years completed, so "6 years 11 months" is still six and a TI starts on
the seventh birthday. The birthday itself counts: someone turning seven today
is seven. A leap-day birth turns a year older on 1 March in a common year, which
is what the subtraction below does without special handling.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Final

#: The seventh birthday: an RC stops being the right document and a TI starts.
TI_MIN_AGE: Final = 7

#: The eighteenth birthday: a TI lapses and a CC is issued.
MAJORITY_AGE: Final = 18


def age_on(birth_date: dt.date, today: dt.date) -> int:
    """Whole years completed on `today`.

    Negative for a birth date in the future, which this module does not judge:
    that is the date normalizer's concern (`DATE_RANGES`), and reporting it here
    as well would blame the document type for a date that is simply wrong.
    """
    before_birthday = (today.month, today.day) < (birth_date.month, birth_date.day)
    return today.year - birth_date.year - before_birthday


@dataclass(frozen=True, slots=True)
class AgeMismatch:
    """A document type that does not fit the age beside it.

    `rule` is a stable name, in the same `area.name` shape the importer's
    normalizers use, so a caller can key wording and tests on it without
    parsing `message`.
    """

    rule: str
    message: str


def check_document_for_age(
    document_type: str, birth_date: dt.date, today: dt.date
) -> AgeMismatch | None:
    """`None` when the type fits the age or has no age rule, else the mismatch.

    Only RC, TI and CC are bound to age. Every other type -- foreign documents,
    and the special RIPS codes for newborns and unidentified patients -- returns
    `None`, because the client's rule covers the ladder and stops there, and
    inventing an age window for a type nobody described would refuse real
    patients.
    """
    age = age_on(birth_date, today)
    if age < 0:
        return None

    code = document_type.strip().upper()

    if code == "RC" and age >= TI_MIN_AGE:
        return AgeMismatch(
            "document.age.rc_outgrown",
            f"A Registro Civil (RC) identifies a child only up to 6 years 11 months, "
            f"and this patient is {age}. From the seventh birthday the document is a "
            f"Tarjeta de Identidad (TI), so the record is probably out of date. "
            f"Check the patient's current document.",
        )
    if code == "TI" and age < TI_MIN_AGE:
        return AgeMismatch(
            "document.age.ti_too_young",
            f"A Tarjeta de Identidad (TI) is issued from age 7, and this patient is "
            f"{age}. Either the document type or the birth date is wrong; check "
            f"which against the document.",
        )
    if code == "TI" and age >= MAJORITY_AGE:
        return AgeMismatch(
            "document.age.ti_expired",
            f"A Tarjeta de Identidad (TI) lapses at 18, and this patient is {age}. "
            f"They should now hold a Cédula de Ciudadanía (CC), so the record is "
            f"probably out of date. Check the patient's current document.",
        )
    if code == "CC" and age < MAJORITY_AGE:
        return AgeMismatch(
            "document.age.cc_under_majority",
            f"A Cédula de Ciudadanía (CC) is issued from age 18, and this patient is "
            f"{age}. Either the document type or the birth date is wrong; check "
            f"which against the document.",
        )
    return None
