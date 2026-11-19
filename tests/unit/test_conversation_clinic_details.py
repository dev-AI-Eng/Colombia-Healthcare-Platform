"""What a clinic configures, and how it reaches a patient.

The address and the arrival advice come from the clinic's own records, not from
code, so each clinic's confirmation says its own thing. They are verified facts
like any other: the provenance stage resolves every number and name in the
message against them, so a clinic's address cannot be a way to smuggle in a
value nothing established.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import replace

from src.conversation import templates as T
from src.conversation.facts import VerifiedFacts
from src.conversation.validator import MessageKind, validate
from src.core.timezones import BOGOTA

START = dt.datetime(2026, 12, 1, 10, 30, tzinfo=BOGOTA)

FACTS = VerifiedFacts(
    doctor_name="Sara Betancur de la Cruz",
    clinic_name="Clínica San Rafael",
    clinic_phone="+57 601 555 0101",
    appointment_starts_at=START,
    appointment_duration_minutes=30,
    arrive_minutes_early=30,
)


def _with(**changes: object) -> VerifiedFacts:
    return replace(FACTS, **changes)  # type: ignore[arg-type]


def test_a_confirmation_states_the_clinics_address_and_note() -> None:
    facts = _with(
        location_address="Calle 100 # 15-20, Bogotá",
        clinic_note="Traiga su documento de identidad.",
    )

    text = T.render(T.CONFIRMED, facts)

    assert "Dirección: Calle 100 # 15-20, Bogotá." in text
    assert text.endswith("Traiga su documento de identidad.")
    assert validate(text, facts, MessageKind.CONFIRMATION).approved


def test_a_confirmation_with_no_address_and_no_note_says_neither() -> None:
    text = T.render(T.CONFIRMED, FACTS)

    assert "Dirección" not in text
    assert text.endswith("minutos antes.")
    assert validate(text, FACTS, MessageKind.CONFIRMATION).approved


def test_an_address_with_numbers_and_names_is_a_verified_fact_not_a_fabrication() -> None:
    """The provenance stage reads "100", "15-20" and "Chapinero" as fact-shaped.
    They resolve because they are in the address the clinic recorded."""
    facts = _with(location_address="Carrera 7 # 32-16, Edificio Torre Sur, Chapinero")

    text = T.render(T.CONFIRMED, facts)

    assert validate(text, facts, MessageKind.CONFIRMATION).approved


def test_the_address_is_not_a_licence_for_other_numbers() -> None:
    """Only what the address contains resolves. A number the address does not
    hold is still refused."""
    facts = _with(location_address="Calle 100 # 15-20")

    verdict = validate(
        T.render(T.CONFIRMED, facts) + " Llame al 300 123 4567.", facts, MessageKind.CONFIRMATION
    )

    assert not verdict.approved


def test_the_note_is_not_a_licence_for_other_numbers() -> None:
    facts = _with(clinic_note="Llegue temprano.")

    verdict = validate(
        T.render(T.CONFIRMED, facts) + " Su turno es el 77.", facts, MessageKind.CONFIRMATION
    )

    assert not verdict.approved
