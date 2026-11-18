"""Turning down an offer is read as that, and nothing else is.

`declines_offer` is only consulted while an offer is open, and what it decides
is the difference between a second round of dates and a cancelled
appointment: a bare "no" to "which of these dates?" must never cancel.
"""

from __future__ import annotations

import datetime as dt

import pytest

from src.conversation.classify import chosen_slot, declines_offer
from src.conversation.facts import spell_date


@pytest.mark.parametrize(
    "reply",
    [
        "ninguna",
        "Ninguno de esos",
        "NINGUNA",
        "no",
        "No",
        "no me sirve ninguna",
        "otras fechas",
        # Found in review: punctuation, and the shortened "ningún" that comes
        # before a masculine noun, both read as nothing.
        "No.",
        "No!",
        "Ningún horario me sirve",
        "ningún día",
    ],
)
def test_these_replies_turn_every_date_down(reply: str) -> None:
    assert declines_offer(reply)


@pytest.mark.parametrize(
    "reply",
    ["2", "el 3", "Si, confirmo", "cancelar", "No puedo ir, cancelar", "no gracias", "gracias"],
)
def test_a_pick_a_confirmation_or_a_cancellation_is_not_a_refusal(reply: str) -> None:
    assert not declines_offer(reply)


@pytest.mark.parametrize(
    ("day", "spelled"),
    [
        (dt.date(2026, 11, 25), "miércoles 25 de noviembre"),
        (dt.date(2026, 11, 28), "sábado 28 de noviembre"),
    ],
)
def test_offered_dates_are_spelled_with_their_accents(day: dt.date, spelled: str) -> None:
    """Patient-facing Spanish (rule 10). Found in the live run of the offer
    rounds, which listed "miercoles"."""
    assert spell_date(day) == spelled


@pytest.mark.parametrize(
    ("reply", "index"),
    [
        ("2", 2),
        ("el 2", 2),
        ("La opción 3", 3),
        ("opcion 5", 5),
        ("2 gracias", 2),
        ("quiero el 4, por favor", 4),
        ("Si, el 1", 1),
        ("2.", 2),
        ("el 3!", 3),
    ],
)
def test_one_number_is_a_pick(reply: str, index: int) -> None:
    assert chosen_slot(reply, 5) == index


@pytest.mark.parametrize(
    "reply",
    [
        # Found in the delivery campaign: each of these read as a pick of the
        # first number in it, which books a date the patient refused.
        "el 2 no",
        "no el 2",
        "no puedo el 2",
        "ni el 2 ni el 3",
        "tampoco el 3",
        "el 4 no puedo, el 5 sí",
        "no, el 3",
        "-1",
        # two options named: which one is not known
        "2 y 3",
        "el 2 o el 3",
        # a number that is a quantity or a time, not an option
        "1 hora antes",
        "5 minutos",
        "a las 10",
        "el 2 de diciembre",
        "10:30",
        # out of range or absent
        "6",
        "0",
        "el segundo",
        "",
    ],
)
def test_a_reply_that_is_not_one_clear_choice_is_not_a_pick(reply: str) -> None:
    assert chosen_slot(reply, 5) is None
