"""Patient text to one intent from a closed set (rule 3).

This is the only place a patient's words are read, and the only thing it may
produce is an `Intent`. The text selects no patient, no appointment and no
tool — `identify` resolved those from the sender's blind index before this runs.

**Deterministic, not a model.** M3's intents are six keyword-shaped decisions
on short replies: "SI", "confirmo", "cancelar", "otra fecha", "2". A model adds
a provider call, a cost and a failure mode to a decision that a lexicon answers
more reliably, and the one case a model would handle better — genuinely
ambiguous prose — is the case that must go to a human anyway.

**`unclear` is a real answer, not a fallback.** Anything this cannot place
becomes `UNCLEAR`, which routes to a templated reply offering the clinic's
number. Guessing between "cancel" and "reschedule" on an unclear message is how
a patient loses an appointment they meant to move.

**Clinical content outranks everything.** A message that mentions symptoms is
`CLINICAL` even if it also says "confirmo", because rule 5 routes anything
clinical to a human and a confirmation is not worth missing that.
"""

from __future__ import annotations

import re
from typing import Final

from src.conversation.facts import normalise
from src.conversation.state import Intent

#: Checked first and wins outright: rule 5. Symptom and medication words, not
#: an attempt at diagnosis — the question is only "is this clinical", and a
#: human answers it from here.
_CLINICAL: Final[tuple[str, ...]] = (
    r"\bdolor(es)?\b",
    r"\bme duele\b",
    r"\bfiebre\b",
    r"\bsangr(a|ado|e)\b",
    r"\bmareo?s?\b",
    r"\bvomit(o|os|ando)\b",
    r"\bdiarrea\b",
    r"\bsintomas?\b",
    r"\bgrave\b",
    r"\bemergencia\b",
    r"\burgencia\b",
    r"\bmedicamento\b",
    r"\bpastilla\b",
    r"\breceta\b",
    r"\bdosis\b",
    r"\bresultado?s?\b",
    r"\bexamen(es)?\b.{0,20}\b(sali(o|eron)|dice|significa)\b",
    r"\bque tengo\b",
    r"\bes normal\b",
    r"\bestoy (mal|peor|preocupad)",
    r"\bno puedo (respirar|caminar|moverme)\b",
    r"\bembarazada\b",
    r"\balergi(a|co)\b",
)

#: Asking for a person. Checked before the action words, because "quiero
#: hablar con alguien para cancelar" wants a person, not a cancellation.
_HUMAN: Final[tuple[str, ...]] = (
    r"\bhablar con (alguien|una persona|un humano|la clinica|recepcion)\b",
    r"\bcomunicar(me|se)\b",
    r"\batencion (personal|humana)\b",
    r"\bllamenme\b",
    r"\bque me llamen\b",
    r"\bno entiendo\b.{0,30}\b(ayuda|persona)\b",
    r"\bagente\b",
)

#: Opting out of the channel. Must be honoured whatever else the message says.
_OPT_OUT: Final[tuple[str, ...]] = (
    r"\bno (quiero|deseo) (mas )?(mensajes|recibir)\b",
    r"\bdar de baja\b",
    r"\bbaja\b",
    r"\bdeten(er|ga)\b.{0,15}\bmensajes\b",
    r"^stop$",
    r"\bno me escriban\b",
    r"\bno escriban mas\b",
)

_CANCEL: Final[tuple[str, ...]] = (
    r"\bcancel(ar|o|en|e|ada|ado)\b",
    r"\banul(ar|o|e)\b",
    r"\bno (voy|podre|puedo) (a )?(ir|asistir)\b",
    r"\bno asistir(e)?\b",
    r"\bya no\b.{0,15}\bcita\b",
    r"^no$",
)

_RESCHEDULE: Final[tuple[str, ...]] = (
    r"\breprogram(ar|e|en)\b",
    r"\bcambiar\b.{0,20}\b(cita|fecha|hora|dia)\b",
    r"\botra (fecha|hora|cita|dia)\b",
    r"\botro (dia|horario)\b",
    r"\bmover\b.{0,15}\bcita\b",
    r"\bmas tarde\b",
    r"\bmas temprano\b",
    r"\bno me (sirve|queda bien|funciona)\b",
    r"\bpuede ser (otro|otra)\b",
    r"\bdisponib(le|ilidad)\b",
)

_CONFIRM: Final[tuple[str, ...]] = (
    r"\bconfirm(ar|o|ado|ada|e)\b",
    r"\bsi,? (asisto|voy|confirmo)\b",
    r"\bahi (estare|voy)\b",
    r"\bde acuerdo\b",
    r"\bperfecto\b",
    r"\bok(ay)?\b",
    r"\bentendido\b",
    r"\bvoy a (ir|asistir)\b",
    r"^si$",
    r"^s$",
    r"^1$",
)


def _matches(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def classify_intent(inbound: str) -> Intent:
    """One intent from a closed set.

    Order is the design. Clinical first (rule 5), then a request for a person,
    then opting out, then the three scheduling actions. Cancel is checked
    before confirm so "no voy a confirmar" is a cancellation rather than a
    confirmation, and reschedule before confirm so "ok pero otra fecha" moves
    the appointment rather than confirming the one the patient just rejected.
    """
    text = normalise(inbound)
    if not text:
        return Intent.UNCLEAR

    if _matches(text, _CLINICAL):
        return Intent.CLINICAL
    if _matches(text, _HUMAN):
        return Intent.HUMAN
    if _matches(text, _OPT_OUT):
        return Intent.OPT_OUT
    if _matches(text, _RESCHEDULE):
        return Intent.RESCHEDULE
    if _matches(text, _CANCEL):
        return Intent.CANCEL
    if _matches(text, _CONFIRM):
        return Intent.CONFIRM
    return Intent.UNCLEAR


#: Turning down every offered date. Only consulted while an offer is open,
#: which is what makes a bare "no" safe to read this way: in answer to "which
#: of these dates?", "no" means none of them, not "cancel my appointment".
_DECLINES_OFFER: Final[tuple[str, ...]] = (
    r"^no[.!]*$",
    # "ningún" before a masculine noun ("ningún horario"), as well as
    # "ninguno" and "ninguna".
    r"\bningun[oa]?s?\b",
    r"\botras? (fechas?|opciones|horas|dias)\b",
    r"\bno me (sirve|sirven|queda|quedan)\b",
)


def declines_offer(inbound: str) -> bool:
    """Whether a reply to an offer turns down every date in it."""
    return _matches(normalise(inbound), _DECLINES_OFFER)


def chosen_slot(inbound: str, offered: int) -> int | None:
    """Which numbered slot the patient picked, or None.

    An offer is numbered (`templates.render_offer`) precisely so this is a
    resolvable choice rather than a sentence to interpret: "2" resolves against
    the offer we sent, while "el martes" would not when two Tuesdays were
    offered.

    **A pick is one clear number, or nothing.** Booking is the one action here
    that moves a patient's appointment, so a reply that could mean two things
    is not read as either. Reading the first number in "el 2 no" or "ni el 2
    ni el 3" booked the very date the patient had just refused, and "1 hora
    antes" booked option 1. Anything with a negation, with more than one
    number, or with a number that is a quantity or a date is not a pick; the
    patient is asked again and answers with the number alone.
    """
    if offered <= 0:
        return None
    text = normalise(inbound)
    if _NOT_A_PICK.search(text):
        return None
    numbers = {int(n) for n in re.findall(r"(?<![\w-])(\d{1,2})(?!\w)", text)}
    if len(numbers) != 1:
        return None
    index = numbers.pop()
    return index if 1 <= index <= offered else None


#: What makes a reply containing a number something other than a choice: a
#: negation (the patient may be refusing that very date), a minus sign, a
#: clock time, a quantity of time, or a date.
_NOT_A_PICK: Final = re.compile(
    r"\b(no|ni|tampoco|nunca|jamas|nada)\b"
    r"|-\s*\d"
    r"|\d:\d"
    r"|\d+\s*(horas?|hrs?|minutos?|mins?|dias?|semanas?|meses|mes|am|pm|a\.m\.?|p\.m\.?)\b"
    r"|\d+\s+de\s+(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\b"
)
