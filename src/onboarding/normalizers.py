"""Deterministic conversion of clinic values into canonical ones.

Every function here is a pure, typed rule. No model writes a transform and no
rule is inferred from the values in the file: that is the failure ADR-08a exists
to prevent, where a transform learned from sampled rows is applied to rows the
sample never represented and produces plausible wrong values.

Each returns an `Outcome`, never a bare value, because three answers are
possible and only one of them is "converted":

    VALID    the value converted, and the rule that did it is recorded
    REVIEW   the value is genuinely ambiguous, so a human decides
    INVALID  the value is damaged beyond recovery, so the row is rejected

The distinction is the whole point. `03/04/1991` is *ambiguous* — it is either 3
April or 4 March and nothing in the cell says which, so guessing corrupts a birth
date silently. A document number that arrived as `1.23457E+11` is *damaged* — the
digits are gone, and a wrong cédula attaches records to the wrong patient. The
first must ask; the second must refuse.
"""

from __future__ import annotations

import datetime as dt
import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import phonenumbers

from src.core.timezones import BOGOTA
from src.registry.models import DocumentType


class Status(StrEnum):
    VALID = "valid"
    REVIEW = "review"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class Outcome[T]:
    """What a normalizer made of one cell, and which rule it applied.

    `rule` and `message` are written to the transform log for every row, so a
    reviewer can see why a value became what it became (ADR-08a).
    """

    status: Status
    value: T | None
    rule: str
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status is Status.VALID


def _valid[T](value: T, rule: str) -> Outcome[T]:
    return Outcome(Status.VALID, value, rule)


def _review[T](rule: str, message: str) -> Outcome[T]:
    return Outcome(Status.REVIEW, None, rule, message)


def _invalid[T](rule: str, message: str) -> Outcome[T]:
    return Outcome(Status.INVALID, None, rule, message)


#: How far back and forward each kind of date may plausibly fall, as years from
#: today. `date()` alone cannot judge this: it serves a birth date and an
#: appointment date, and next year is wrong for one and ordinary for the other.
#:
#: Generous on purpose. The oldest verified Colombians are around 115, so 130
#: admits every living patient and still catches a year typed as 1890 or 0001.
#: An appointment 5 years out is a long-range control; 2 years back covers a
#: historical import, and a clinic migrating older history will see a question
#: rather than a silent acceptance.
DATE_RANGES: Final = {
    "birth_date": (-130, 0),
    "consent_granted_at": (-30, 0),
    "appointment_date": (-2, 5),
    "valid_from": (-30, 5),
    "valid_to": (-30, 30),
}


def _years_from(day: dt.date, years: int) -> dt.date:
    """`day` shifted by whole years, landing on 28 February from a leap day.

    `date.replace(year=...)` raises on 29 February whenever the target year is
    not a leap year, and `day` here is today's date from the clock. Without this
    every import carrying a date column would answer HTTP 500 on 29 February and
    on no other day, which no test can reach because the date is not an input.
    """
    try:
        return day.replace(year=day.year + years)
    except ValueError:
        return day.replace(month=2, day=28, year=day.year + years)


def plausible_date(value: dt.date, field_name: str, *, today: dt.date | None = None) -> str | None:
    """Why this date is implausible for this field, or None if it is fine.

    Returns the message rather than an Outcome so the caller keeps the rule name
    it already has; a field with no range is never questioned.
    """
    window = DATE_RANGES.get(field_name)
    if window is None:
        return None
    today = today or dt.datetime.now(tz=BOGOTA).date()
    earliest = _years_from(today, window[0])
    latest = _years_from(today, window[1])
    if value < earliest:
        return (
            f"{value.isoformat()} is further back than a {field_name.replace('_', ' ')} "
            f"should reach. Confirm the year."
        )
    if value > latest:
        return (
            f"{value.isoformat()} is further ahead than a "
            f"{field_name.replace('_', ' ')} should reach. Confirm the year."
        )
    return None


def _ascii_digits(text: str) -> bool:
    """Whether every character is 0-9.

    `str.isdigit()` is not this test: it is true for superscripts, circled
    numerals and fullwidth and Arabic-Indic digits, none of which `int()` can
    parse and none of which belongs in a cédula or a date. Accepting them stored
    a value nobody typed and, where `int()` followed, crashed the request.
    """
    return bool(text) and text.isascii() and text.isdecimal()


def strip_accents(value: str) -> str:
    """Casefold and remove accents, for matching only.

    Used to compare headers and labels. It is never applied to a value being
    stored: "Muñoz" and "Munoz" are different names, and restoring an accent
    that a file does not carry would be inventing data.
    """
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold().strip()


# --------------------------------------------------------------- document type
# RIPS codes, with the spellings clinic files actually use. Resolución 948 de
# 2026 moved this list into a technical document MinSalud can revise without a
# new resolution, so it is configuration rather than a database constraint.
_DOCUMENT_TYPE_ALIASES: Final[dict[str, DocumentType]] = {
    "cc": DocumentType.CC,
    "c.c.": DocumentType.CC,
    "cedula": DocumentType.CC,
    "cedula de ciudadania": DocumentType.CC,
    "cedula ciudadania": DocumentType.CC,
    "ciudadania": DocumentType.CC,
    "id card": DocumentType.CC,
    "ced": DocumentType.CC,
    "ti": DocumentType.TI,
    "t.i.": DocumentType.TI,
    "tarjeta de identidad": DocumentType.TI,
    "tarjeta identidad": DocumentType.TI,
    "minor id": DocumentType.TI,
    "rc": DocumentType.RC,
    "r.c.": DocumentType.RC,
    "registro civil": DocumentType.RC,
    "registro civil de nacimiento": DocumentType.RC,
    "ce": DocumentType.CE,
    "c.e.": DocumentType.CE,
    "cedula de extranjeria": DocumentType.CE,
    "cedula extranjeria": DocumentType.CE,
    "foreign id": DocumentType.CE,
    "pa": DocumentType.PA,
    "pas": DocumentType.PA,
    "pasaporte": DocumentType.PA,
    "passport": DocumentType.PA,
    # PPT is the everyday name of the card; PT is the RIPS code. A frequent mismatch.
    "pt": DocumentType.PT,
    "ppt": DocumentType.PT,
    "permiso por proteccion temporal": DocumentType.PT,
    "pe": DocumentType.PE,
    "pep": DocumentType.PE,
    "permiso especial de permanencia": DocumentType.PE,
    "cd": DocumentType.CD,
    "carne diplomatico": DocumentType.CD,
    "sc": DocumentType.SC,
    "salvoconducto": DocumentType.SC,
    "salvoconducto de permanencia": DocumentType.SC,
    "de": DocumentType.DE,
    "documento extranjero": DocumentType.DE,
    "cn": DocumentType.CN,
    "certificado de nacido vivo": DocumentType.CN,
    "as": DocumentType.AS,
    "adulto sin identificar": DocumentType.AS,
    "ms": DocumentType.MS,
    "menor sin identificar": DocumentType.MS,
}


def document_type(raw: str) -> Outcome[DocumentType]:
    """Map a written document type onto its RIPS code.

    An unrecognised value goes to review and is never defaulted to `CC`: the
    document type is half of a patient's legal identity, and the wrong one pairs
    the right number with the wrong person.
    """
    text = strip_accents(raw)
    if not text:
        return _review("document_type.empty", "No document type given.")
    # Whole-token match only. "DE" is also the preposition inside almost every
    # other label ("cedula DE ciudadania"), so a substring match would see it
    # everywhere.
    found = _DOCUMENT_TYPE_ALIASES.get(text) or _DOCUMENT_TYPE_ALIASES.get(text.replace(".", ""))
    if found is None:
        return _review("document_type.unknown", f"Unrecognised document type {raw!r}.")
    return _valid(found, "document_type.alias")


# ------------------------------------------------------------- document number
_SCIENTIFIC = re.compile(r"^\d(?:\.\d+)?[eE][+-]?\d+$")


def document_number(raw: str) -> Outcome[str]:
    """Return the digits of a document number, or refuse the row.

    Excel damages these two ways, both unrecoverable, so both are rejected
    rather than repaired: a long number stored as a float becomes scientific
    notation and loses its low-order digits, and a number stored numerically
    loses any leading zero. A repaired cédula is a different person's.
    """
    text = raw.strip()
    if not text:
        return _invalid("document_number.empty", "Document number is empty.")

    if _SCIENTIFIC.match(text):
        return _invalid(
            "document_number.scientific_notation",
            f"{raw!r} was stored as a number and its digits are lost. "
            "Re-export the column formatted as text.",
        )

    # "1.045.678.901" is the Spanish thousands form; "1045678901.0" is a float
    # that happens to be whole. Neither loses information, so both are accepted.
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", text):
        text = text.replace(".", "")
    elif text.endswith(".0") and _ascii_digits(text[:-2]):
        text = text[:-2]

    if not _ascii_digits(text):
        # Passports and foreign documents legitimately contain letters.
        if re.fullmatch(r"[A-Za-z0-9-]{4,20}", text):
            return _valid(text.upper(), "document_number.alphanumeric")
        return _review("document_number.unexpected", f"{raw!r} is not a recognisable number.")

    if not 4 <= len(text) <= 20:
        return _invalid(
            "document_number.length",
            f"{raw!r} has {len(text)} digits; RIPS allows 4 to 20.",
        )
    return _valid(text, "document_number.digits")


# ----------------------------------------------------------------------- phone
def phone(raw: str, *, region: str = "CO") -> Outcome[str]:
    """Return an E.164 number, or say why it cannot be reached.

    A number that does not exist is never "corrected": the nearest valid number
    belongs to somebody else, and a reminder sent there is a disclosure to a
    stranger.

    The outcome is `review`, which holds the **whole row** back until a person
    answers it, because a row is only as importable as its least certain cell.
    Whether an otherwise-good patient should instead import with the phone left
    empty is a product decision, not this function's to make.
    """
    text = raw.strip()
    if not text:
        return _review("phone.empty", "No phone number given.")

    digits = re.sub(r"[^\d+]", "", text)
    try:
        parsed = phonenumbers.parse(digits, region)
    except phonenumbers.NumberParseException as error:
        return _review("phone.unparseable", f"{raw!r} is not a phone number ({error}).")

    # Which mobile prefixes exist is libphonenumber's metadata, not ours: its CO
    # pattern currently admits 300-305 and 310-319 but not 306-309. If real
    # clinic files start producing a cluster of rejections in one prefix, that
    # is evidence the metadata is stale, and the fix is upgrading the
    # `phonenumbers` package, which is where the metadata ships. Never widen a
    # pattern by hand here: a prefix we invent sends reminders into the void.
    if not phonenumbers.is_valid_number(parsed):
        return _review(
            "phone.not_assigned",
            f"{raw!r} is not an assigned Colombian number, so it cannot be reached.",
        )
    return _valid(
        phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164),
        "phone.e164",
    )


# --------------------------------------------------------------------- weekday
#: Spanish and English day names, to `date.weekday()` numbering where Monday is
#: 0. Accents are stripped before lookup, so "miercoles" and "miércoles" both
#: resolve; a clinic file has both spellings.
_WEEKDAYS: Final = {
    "lunes": 0,
    "monday": 0,
    "lun": 0,
    "mon": 0,
    "l": 0,
    "martes": 1,
    "tuesday": 1,
    "mar": 1,
    "tue": 1,
    "miercoles": 2,
    "wednesday": 2,
    "mie": 2,
    "wed": 2,
    "jueves": 3,
    "thursday": 3,
    "jue": 3,
    "thu": 3,
    "j": 3,
    "viernes": 4,
    "friday": 4,
    "vie": 4,
    "fri": 4,
    "v": 4,
    "sabado": 5,
    "saturday": 5,
    "sab": 5,
    "sat": 5,
    "s": 5,
    "domingo": 6,
    "sunday": 6,
    "dom": 6,
    "sun": 6,
    "d": 6,
}


def weekday(raw: str) -> Outcome[int]:
    """Return a weekday as `date.weekday()` numbers it, or refuse.

    A bare number is ambiguous and is NOT guessed: "1" is Monday under ISO-8601
    and Sunday in a spreadsheet written by someone counting from Sunday, and
    nothing in a single cell says which. Getting it wrong moves a doctor's whole
    working week by a day, so a numeric column asks rather than picking.

    `7` is refused outright rather than read as Sunday: under ISO numbering it
    is Sunday, but `date.weekday()` has no 7, and a file containing both 0 and 7
    is using two conventions at once.
    """
    text = strip_accents(raw.strip())
    if not text:
        return _review("weekday.empty", "No day of the week given.")

    if (found := _WEEKDAYS.get(text)) is not None:
        return _valid(found, "weekday.name")

    if _ascii_digits(text):
        return _review(
            "weekday.numeric",
            f"{raw!r} is a number, and a numbered day is ambiguous: 1 is Monday in "
            f"one convention and Sunday in another. Export the day names instead, "
            f"or confirm which convention this file uses.",
        )

    return _review(
        "weekday.unknown",
        f"{raw!r} is not a day of the week this system recognises.",
    )


# --------------------------------------------------------------------- consent
#: How clinics write the purpose a patient agreed to. Anything unrecognised goes
#: to review rather than being mapped to the broadest purpose: consent to an
#: appointment reminder is not consent to a wellness check-in.
_CONSENT_PURPOSES: Final = {
    "citas": "appointment_messaging",
    "cita": "appointment_messaging",
    "recordatorios": "appointment_messaging",
    "recordatorio de citas": "appointment_messaging",
    "mensajes de citas": "appointment_messaging",
    "appointment_messaging": "appointment_messaging",
    "appointments": "appointment_messaging",
    "bienestar": "wellness_checkins",
    "seguimiento": "wellness_checkins",
    "wellness_checkins": "wellness_checkins",
    "datos sensibles": "sensitive_data",
    "sensitive_data": "sensitive_data",
    "telemedicina": "telemedicine",
    "telemedicine": "telemedicine",
}

#: How the consent was obtained. `imported_declaration` is the honest value when
#: a clinic asserts consent exists without evidence we hold, and it is kept
#: distinct so a later audit can tell it from a signed form.
_EVIDENCE_KINDS: Final = {
    "escrito": "written",
    "firmado": "written",
    "formato fisico": "written",
    "written": "written",
    "verbal": "verbal_recorded",
    "telefonico": "verbal_recorded",
    "grabado": "verbal_recorded",
    "verbal_recorded": "verbal_recorded",
    "digital": "digital_form",
    "formulario": "digital_form",
    "web": "digital_form",
    "digital_form": "digital_form",
    "declaracion": "imported_declaration",
    "declarado": "imported_declaration",
    "imported_declaration": "imported_declaration",
}


def consent_purpose(raw: str) -> Outcome[str]:
    """What the patient agreed to receive.

    An unrecognised purpose is never widened to cover more than it says: that
    would manufacture consent the patient did not give.
    """
    text = strip_accents(raw.strip()).casefold()
    if not text:
        return _review("consent.purpose_empty", "No consent purpose given.")
    found = _CONSENT_PURPOSES.get(text)
    if found is None:
        return _review(
            "consent.purpose_unknown",
            f"{raw!r} is not a consent purpose this system recognises. Confirm which "
            f"of appointment messaging, wellness check-ins, sensitive data or "
            f"telemedicine it means.",
        )
    return _valid(found, "consent.purpose")


def consent_evidence(raw: str) -> Outcome[str]:
    """How the consent was obtained."""
    text = strip_accents(raw.strip()).casefold()
    if not text:
        return _review("consent.evidence_empty", "No consent evidence given.")
    found = _EVIDENCE_KINDS.get(text)
    if found is None:
        return _review(
            "consent.evidence_unknown",
            f"{raw!r} is not a kind of consent evidence this system recognises. "
            f"Confirm whether it was written, verbal and recorded, a digital form, "
            f"or the clinic declaring it on import.",
        )
    return _valid(found, "consent.evidence")


# ------------------------------------------------------------------ boolean-ish
_TRUE: Final = frozenset({"si", "s", "yes", "y", "true", "1", "x", "verdadero"})
_FALSE: Final = frozenset({"no", "n", "false", "0", "falso"})


def boolean(raw: str) -> Outcome[bool]:
    """Read SI/NO/X/1/0 and their variants.

    Blank is review rather than False: an empty attendance cell usually means
    "not recorded yet", and reading it as "did not attend" invents a fact.
    """
    text = strip_accents(raw).replace(".", "")
    if not text:
        return _review("boolean.empty", "Empty; it is not clear whether this means no.")
    if text in _TRUE:
        return _valid(True, "boolean.true")
    if text in _FALSE:
        return _valid(False, "boolean.false")
    # "N/A" is ambiguous in Spanish clinic files between "no aplica" and
    # "no asistió", which are opposites.
    return _review("boolean.unknown", f"{raw!r} is not a clear yes or no.")


# ------------------------------------------------------------------------ name
# Particles that belong to the surname that follows them.
_PARTICLES: Final = frozenset(
    {"de", "del", "la", "las", "los", "san", "santa", "van", "von", "da", "do", "di", "y", "e"}
)

# Given names that are two words. Without this list "María José Pérez Gómez"
# looks like four separate tokens and splits in the wrong place.
_COMPOUND_GIVEN: Final = frozenset(
    {
        "maria jose",
        "maria fernanda",
        "maria camila",
        "maria paula",
        "maria alejandra",
        "maria isabel",
        "maria clara",
        "maria del",
        "ana maria",
        "ana sofia",
        "ana lucia",
        "juan carlos",
        "juan pablo",
        "juan david",
        "juan jose",
        "juan manuel",
        "juan sebastian",
        "jose luis",
        "jose maria",
        "jose antonio",
        "jose david",
        "luis carlos",
        "luis fernando",
        "luis miguel",
        "carlos andres",
        "carlos alberto",
        "jorge luis",
        "diana carolina",
        "sandra milena",
        "luz marina",
        "luz dary",
        "leidy johana",
        "jhon jairo",
        "andres felipe",
        "luisa fernanda",
        "claudia patricia",
        "sandra patricia",
        "martha lucia",
    }
)


@dataclass(frozen=True, slots=True)
class SplitName:
    given_names: str
    family_names: str


#: Leading characters that make a cell a formula in Excel, LibreOffice and
#: Google Sheets. The whitespace leads are listed for completeness only: the
#: collapse above strips them, so a cell beginning with one is already compared
#: on the character behind it.
FORMULA_LEADS: Final = frozenset("""=+-@\t\r\n""")


def split_full_name(raw: str) -> Outcome[SplitName]:
    """Split one name column into given names and surnames, or refuse.

    Only two shapes are decidable. Two tokens are one given name and one
    surname. Four are two and two, once a compound given name is glued.

    Three tokens are refused because they are genuinely undecidable:
    "Carlos Pérez Gómez" is one given name and two surnames, while
    "Juan Carlos Pérez" is two given names and one surname, and nothing in the
    text distinguishes them. Ley 2129 de 2021 also lets parents choose the order
    of surnames, so no "the paternal surname comes first" rule can help.

    Five or more are refused for the same reason, compounded by particles.
    A refusal costs a human one decision on the confirmation screen; a wrong
    split attaches a record to the wrong person.
    """
    text = " ".join(raw.split())
    if not text:
        return _review("name.empty", "No name given.")

    # A name is the one free-text field here, and a cell beginning with one of
    # these is a formula to Excel, LibreOffice and Sheets. Stored as a name it
    # is inert; exported or opened later it executes, which is the CSV injection
    # OWASP describes. Every other field rejects it by shape already. It goes to
    # review rather than invalid because the cell may be a real name the export
    # mangled, and only a person can say.
    # Every token, not just the first. The name is split into two fields and each
    # becomes a cell of its own, so "Perez, =HYPERLINK(...)" and "Carlos @SUM(1)"
    # both put a formula in an output cell while the raw text starts with a
    # letter. Checking the whole string once here covers every split branch below.
    # Fields with no normalizer are guarded in `service._apply`'s as-written
    # branch; this one is stricter because the value is split before storage.
    for token in text.replace(",", " ").split():
        if token[0] in FORMULA_LEADS:
            return _review(
                "name.looks_like_a_formula",
                f"{raw!r} contains {token[0]!r} where a name should be, so a "
                "spreadsheet would read it as a formula. Confirm the real name.",
            )

    # "PEREZ GOMEZ, CARLOS ANDRES" is surnames first. The comma says so, which
    # makes this the one name shape that IS decidable -- and reading it
    # left-to-right stored the surnames as given names and the given names as
    # surnames, marked valid, which is a patient filed under the wrong identity.
    if text.count(",") == 1:
        family_part, _, given_part = text.partition(",")
        family_part, given_part = family_part.strip(), given_part.strip()
        if family_part and given_part:
            return _valid(SplitName(given_part, family_part), "name.surnames_first_comma")
        return _review(
            "name.comma_incomplete",
            f"{raw!r} has a comma but only one side of it. A name written "
            f"'APELLIDOS, NOMBRES' needs both.",
        )
    if text.count(",") > 1:
        return _review(
            "name.multiple_commas",
            f"{raw!r} has more than one comma, so where the surnames end cannot be read from it.",
        )

    tokens = text.split()

    # Glue particles onto the token they modify: "de la Cruz" is one surname.
    glued: list[str] = []
    buffer: list[str] = []
    for token in tokens:
        if strip_accents(token) in _PARTICLES:
            buffer.append(token)
            continue
        glued.append(" ".join([*buffer, token]) if buffer else token)
        buffer = []
    if buffer:  # trailing particle: the name is malformed
        return _review("name.trailing_particle", f"{raw!r} ends with a connecting word.")

    # Four parts are two given names and two surnames. The compound list is not
    # consulted here: "Carlos Andrés Pérez Gómez" splits the same way whether or
    # not "Carlos Andrés" is a recognised pair, and gluing it first would leave
    # three parts and send a perfectly clear name to review.
    if len(glued) == 4:
        return _valid(
            SplitName(f"{glued[0]} {glued[1]}", f"{glued[2]} {glued[3]}"), "name.four_tokens"
        )
    if len(glued) == 2:
        return _valid(SplitName(glued[0], glued[1]), "name.two_tokens")

    if len(glued) == 3:
        # Three parts are undecidable in general, but a recognised compound
        # given name settles it: "María José Pérez" is one person's two given
        # names and one surname, not one given name and two surnames.
        pair = " ".join(strip_accents(token) for token in glued[:2])
        if pair in _COMPOUND_GIVEN:
            return _valid(SplitName(f"{glued[0]} {glued[1]}", glued[2]), "name.compound_given")
        return _review(
            "name.three_tokens",
            f"{raw!r} could be one given name and two surnames, or two given names and one. "
            "Confirm the split, or export the name as separate columns.",
        )
    return _review(
        "name.unsupported_shape",
        f"{raw!r} has {len(glued)} parts; confirm which are given names and which are surnames.",
    )


# ------------------------------------------------------------------------ date
_ISO = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_SLASHED = re.compile(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})$")

# Excel's 1900 system is wrong before serial 61: it includes a 29 February 1900
# that never existed, and openpyxl maps serials 59 and 60 onto the same date.
_MIN_SAFE_1900_SERIAL: Final = 61
_EPOCH_1900: Final = dt.date(1899, 12, 30)
_EPOCH_1904: Final = dt.date(1904, 1, 1)


class DayFirst(StrEnum):
    """Which way a whole column reads. Decided once per column, never per row."""

    DAY_FIRST = "day_first"
    MONTH_FIRST = "month_first"
    UNDECIDED = "undecided"


def detect_day_first(values: list[str]) -> DayFirst:
    """Decide a column's date order from the whole column.

    A value whose first component exceeds 12 can only be a day, which settles
    the column. If no value settles it, the column stays UNDECIDED and every
    value in it goes to review: `03/04/1991` is 3 April or 4 March, and falling
    back to a locale guesses, because the file was written by an Excel whose
    locale we do not know.
    """
    saw_day_first = False
    saw_month_first = False
    for value in values:
        match = _SLASHED.match(value.strip())
        if not match:
            continue
        first, second = int(match.group(1)), int(match.group(2))
        if first > 12 and second <= 12:
            saw_day_first = True
        elif second > 12 and first <= 12:
            saw_month_first = True

    if saw_day_first and saw_month_first:
        return DayFirst.UNDECIDED  # the column contradicts itself
    if saw_day_first:
        return DayFirst.DAY_FIRST
    if saw_month_first:
        return DayFirst.MONTH_FIRST
    return DayFirst.UNDECIDED


def date(raw: str, *, order: DayFirst, epoch_1904: bool = False) -> Outcome[dt.date]:
    """Convert one cell to a date under a decision already made for the column."""
    text = raw.strip()
    if not text:
        return _review("date.empty", "No date given.")

    # openpyxl hands back real date cells already converted, as ISO text.
    if " " in text and _ISO.match(text.split(" ")[0]):
        text = text.split(" ")[0]
    if _ISO.match(text):
        try:
            return _valid(dt.date.fromisoformat(text), "date.iso")
        except ValueError as error:
            return _invalid("date.impossible", f"{raw!r} is not a real date ({error}).")

    if match := _SLASHED.match(text):
        first, second, year = (int(g) for g in match.groups())
        if order is DayFirst.UNDECIDED:
            return _review(
                "date.ambiguous_column",
                f"{raw!r} could be day/month or month/day. Confirm the format for this column.",
            )
        day, month = (first, second) if order is DayFirst.DAY_FIRST else (second, first)
        try:
            return _valid(dt.date(year, month, day), f"date.{order.value}")
        except ValueError as error:
            return _invalid("date.impossible", f"{raw!r} is not a real date ({error}).")

    if _ascii_digits(text):
        # `19900315` is a compact date, not a serial: eight digits beginning
        # with a plausible year is the form RIPS and many Colombian systems
        # export, and reading it as a day count is how it became an
        # OverflowError rather than a birth date.
        if len(text) == 8:
            try:
                return _valid(dt.date.fromisoformat(text), "date.compact_iso")
            except ValueError:
                return _review(
                    "date.compact_invalid",
                    f"{raw!r} looks like a yyyymmdd date but is not a real one.",
                )
        return _excel_serial(int(text), epoch_1904=epoch_1904)

    return _review("date.unrecognised", f"{raw!r} is not a date we recognise.")


#: The largest serial Excel itself accepts, which is 31 December 9999. Beyond it
#: there is no date to convert to, and adding the number to the epoch raises
#: OverflowError rather than returning anything a reviewer could act on.
_MAX_SERIAL: Final = 2_958_465


def _excel_serial(serial: int, *, epoch_1904: bool) -> Outcome[dt.date]:
    """Convert a raw Excel serial, refusing the range Excel itself gets wrong."""
    if serial > _MAX_SERIAL or serial < 0:
        return _review(
            "date.serial_out_of_range",
            f"{serial} is too large to be a date. If this is a number rather than "
            f"a date, the column is mapped to the wrong field.",
        )
    if epoch_1904:
        return _valid(_EPOCH_1904 + dt.timedelta(days=serial), "date.serial_1904")
    if serial < _MIN_SAFE_1900_SERIAL:
        # Serial 60 is Excel's imaginary 29 February 1900, and everything below
        # it is off by one depending on who is counting.
        return _invalid(
            "date.serial_pre_1900_bug",
            f"Serial {serial} falls in the range Excel dates incorrectly (before 1 March 1900).",
        )
    return _valid(_EPOCH_1900 + dt.timedelta(days=serial), "date.serial_1900")


# ------------------------------------------------------------------------ time
# Spanish clinic files write the meridiem several ways, and Excel in a Spanish
# locale separates the letters with a non-breaking or narrow no-break space, so
# splitting on an ordinary space misses them.
_TIME_12H = re.compile(
    r"^(\d{1,2})[:.](\d{2})(?:[:.](\d{2}))?\s*"
    r"([ap])\s*\.?\s*m\s*\.?$",
    re.IGNORECASE,
)
_TIME_24H = re.compile(r"^(\d{1,2})[:.](\d{2})(?:[:.](\d{2}))?$")
_SPACES: Final = str.maketrans({" ": " ", " ": " ", " ": " "})


def time_of_day(raw: str) -> Outcome[dt.time]:
    """Convert a clock time, in 24-hour or Spanish 12-hour form."""
    text = raw.strip().translate(_SPACES)
    if not text:
        return _review("time.empty", "No time given.")

    if match := _TIME_12H.match(text):
        hour, minute = int(match.group(1)), int(match.group(2))
        second = int(match.group(3) or 0)
        meridiem = match.group(4).lower()
        if not 1 <= hour <= 12:
            return _invalid("time.impossible", f"{raw!r} has no valid 12-hour clock hour.")
        if meridiem == "p" and hour != 12:
            hour += 12
        elif meridiem == "a" and hour == 12:
            hour = 0  # 12 a.m. is midnight
        return _valid(dt.time(hour, minute, second), "time.twelve_hour")

    if match := _TIME_24H.match(text):
        hour, minute = int(match.group(1)), int(match.group(2))
        second = int(match.group(3) or 0)
        if hour > 23 or minute > 59 or second > 59:
            return _invalid("time.impossible", f"{raw!r} is not a real time.")
        return _valid(dt.time(hour, minute, second), "time.twenty_four_hour")

    # Excel stores a time as a fraction of a day.
    try:
        fraction = float(text)
    except ValueError:
        return _review("time.unrecognised", f"{raw!r} is not a time we recognise.")
    if not 0.0 <= fraction < 1.0:
        return _review("time.out_of_range", f"{raw!r} is not a fraction of a day.")
    # Rounded, not truncated: 0.354166666 is 08:29:59.99, which truncates to
    # 08:29:59 and reads as a minute earlier than the clinic wrote.
    total = round(fraction * 86_400)
    return _valid(dt.time(total // 3600 % 24, total % 3600 // 60, total % 60), "time.day_fraction")


# ---------------------------------------------------------------------- status
_STATUS_ALIASES: Final[dict[str, str]] = {
    "agendada": "scheduled",
    "asignada": "scheduled",
    "programada": "scheduled",
    "reservada": "scheduled",
    "pending": "scheduled",
    "scheduled": "scheduled",
    "confirmada": "confirmed",
    "confirmado": "confirmed",
    "confirmed": "confirmed",
    "atendida": "completed",
    "cumplida": "completed",
    "asistio": "completed",
    "realizada": "completed",
    "completed": "completed",
    "cancelada": "cancelled",
    "anulada": "cancelled",
    "cancelled": "cancelled",
    "no asistio": "no_show",
    "inasistencia": "no_show",
    "no cumplio": "no_show",
    "no show": "no_show",
    "no_show": "no_show",
    "reprogramada": "rescheduled",
    "reagendada": "rescheduled",
    "rescheduled": "rescheduled",
}


def appointment_status(raw: str) -> Outcome[str]:
    """Map a written appointment state onto the canonical set.

    `pendiente` is deliberately absent: in Colombian files it means both "not
    yet confirmed" and "waiting list", which are different states. Bare `NA` is
    absent too, being either "no aplica" or "no asistió" — opposites.
    """
    text = strip_accents(raw)
    if not text:
        return _review("status.empty", "No status given.")
    if found := _STATUS_ALIASES.get(text):
        return _valid(found, "status.alias")
    if text in {"pendiente", "na", "n/a"}:
        return _review("status.ambiguous", f"{raw!r} has more than one meaning; confirm it.")
    return _review("status.unknown", f"Unrecognised status {raw!r}.")
