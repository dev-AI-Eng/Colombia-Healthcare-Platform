"""A doctors sheet can carry the doctor's phone, second phone and email.

The client asked for those three on the doctors screen. Adding columns nobody
can fill would have been half a feature, so the importer's doctor entity gained
the fields too, and these tests cover the part that is easy to get wrong: which
of a clinic's own headings reach which field.
"""

from __future__ import annotations

from src.onboarding import service
from src.onboarding.canonical import Entity
from src.onboarding.matcher import match_sheet
from src.onboarding.reader import Sheet

HEADERS = ("Código", "Nombre", "Especialidad", "Celular", "Teléfono 2", "Correo electrónico")


def _mapping() -> dict[str, str | None]:
    proposed = match_sheet(HEADERS, Entity.DOCTOR)
    return {p.column: (p.field.name if p.field else None) for p in proposed.proposals}


def test_a_doctors_own_headings_reach_the_contact_fields() -> None:
    mapping = _mapping()
    assert mapping["Celular"] == "phone"
    assert mapping["Teléfono 2"] == "secondary_phone"
    assert mapping["Correo electrónico"] == "email"


def test_the_contact_columns_convert_through_the_same_rules_as_everyone_elses() -> None:
    """The phone is the phone normalizer's, so a number that does not exist is
    flagged rather than corrected -- the nearest valid number belongs to
    somebody else, and a doctor is no exception."""
    sheet = Sheet(
        name="Medicos",
        headers=HEADERS,
        rows=(
            ("M01", "Ana Restrepo", "Pediatría", "300 111 2233", "6012345678", "ana@clinica.co"),
        ),
        header_row=1,
    )
    rows, _ = service.validate(sheet, Entity.DOCTOR, _mapping(), decisions={})

    values = rows[0].values
    assert values["phone"] == "+573001112233"
    assert values["email"] == "ana@clinica.co"
    assert rows[0].status.value == "valid"


def test_a_doctors_phone_that_does_not_exist_is_not_corrected() -> None:
    sheet = Sheet(
        name="Medicos",
        headers=HEADERS,
        rows=(("M01", "Ana Restrepo", "Pediatría", "3954336730", "", ""),),
        header_row=1,
    )
    rows, _ = service.validate(sheet, Entity.DOCTOR, _mapping(), decisions={})

    assert rows[0].status.value != "valid"
    assert "phone" not in rows[0].values
