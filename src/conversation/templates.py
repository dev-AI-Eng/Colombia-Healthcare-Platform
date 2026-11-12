"""Pre-approved Spanish text, and the facts each piece is allowed to carry.

Two message classes exist (ADR-13). Generated text goes generator → evaluator.
Templated text is wording a person approved once, with positional variables
filled by code, and it is gated by **provenance validation** instead: there is
no model in its path, so there is nothing for a judge to judge, and routing it
through one would introduce an LLM into a path that has none.

Everything here is templated. Each entry is a format string whose variables are
filled from `verified_facts` by `render`, so a template cannot state a fact
that no tool established — the same guarantee the generated path gets from the
provenance stage, obtained here by construction.

Wording is Spanish for Colombia (rule 10), and deliberately plain: these are
read by a patient on a cheap phone, often an elderly one. 12-hour times with
`a.m.`/`p.m.` per §7.

**These strings are not final.** Every one that WhatsApp will send has to be
submitted to Meta for template approval before M4 can use it (§9a), which takes
hours to days and can come back rejected. They are written here so M3's graph
is complete and testable; the wording is the client's to confirm.
"""

from __future__ import annotations

from typing import Final

from src.conversation.facts import VerifiedFacts
from src.conversation.validator import MessageKind

#: A fact a template needs but `verified_facts` does not hold renders as this
#: rather than as "None". It never reaches a patient: `render` reports the
#: missing field and the graph escalates instead of sending.
MISSING: Final = "�"


# --------------------------------------------------------------------- bodies
#
# Named by what they tell the patient, not by the intent that produced them: a
# cancellation confirmed by the patient and one forced by the clinic read
# differently and are different templates.

CONFIRMED: Final = (
    "Su cita quedó confirmada para el {date} a las {time} con {doctor}. "
    "Por favor llegue {arrive} minutos antes. "
    "Si tiene exámenes o imágenes previas, tráigalos."
)

CANCELLED: Final = (
    "Su cita fue cancelada. "
    "Si desea agendar una nueva, por favor llame a la clínica al {clinic_phone}."
)

#: Offered times are appended by `render_offer`, because the count varies.
OFFER_INTRO: Final = "Estos son los horarios disponibles con {doctor}:"

OFFER_FOOTER: Final = (
    "Responda con el número del horario que prefiera, o escriba CANCELAR si desea cancelar la cita."
)

REMINDER: Final = (
    "Le recordamos su cita del {date} a las {time} con {doctor}. "
    "Responda CONFIRMAR para confirmar, o CANCELAR si no puede asistir."
)

#: ADR-23's exhaustion path. Carries no appointment facts at all, so it passes
#: the provenance stage whatever the state of `verified_facts` -- which is the
#: property that makes it a safe landing place when everything else failed.
STAFF_WILL_CALL: Final = (
    "Gracias por su mensaje. Un miembro del personal de la clínica se pondrá en contacto con usted."
)

#: Rule 5. A clinical question is never answered, in any form.
CLINICAL_TO_HUMAN: Final = (
    "Por su seguridad no podemos dar información médica por este medio. "
    "Un miembro del personal de la clínica se pondrá en contacto con usted."
)

#: ADR-18: the sender is on no binding. Says nothing about any patient, because
#: the clinic does not know who is writing -- naming a patient here would
#: disclose their data to a stranger.
UNKNOWN_SENDER: Final = (
    "No pudimos identificar este número en nuestros registros. "
    "Si necesita agendar o modificar una cita, por favor llame a la clínica."
)

#: ADR-18: several patients share this handset. Asks rather than guesses, and
#: names nobody for the same reason as above.
SHARED_HANDSET: Final = (
    "Este número está registrado para más de una persona. "
    "Para continuar, por favor llame a la clínica e indique a nombre de quién "
    "es la cita."
)

#: ADR-19: no valid consent for this purpose. Does not re-ask for consent over
#: the channel the consent would authorise -- that circularity is why consent
#: is captured at reception (CLAUDE.md §9a).
NO_CONSENT: Final = (
    "Para poder escribirle por este medio necesitamos su autorización. "
    "Por favor acérquese a la clínica o llámenos para registrarla."
)

OPTED_OUT: Final = (
    "Hemos registrado que no desea recibir más mensajes por este medio. "
    "Si cambia de opinión, por favor llame a la clínica."
)

NO_SLOTS: Final = (
    "No encontramos horarios disponibles con los criterios indicados. "
    "Un miembro del personal de la clínica se pondrá en contacto con usted."
)

#: Said when a patient confirms or cancels something that is not there. Not an
#: error message: a patient tapping a stale button has done nothing wrong.
NO_APPOINTMENT: Final = (
    "No encontramos una cita activa asociada a este número. "
    "Si cree que es un error, por favor llame a la clínica."
)


#: Which templates carry appointment facts, and therefore what the provenance
#: stage will require of them. A template absent from here carries none, and
#: `MessageKind.FALLBACK` requires nothing — that is what lets the exhaustion
#: path pass the gate from any state.
KINDS: Final[dict[str, MessageKind]] = {
    CONFIRMED: MessageKind.CONFIRMATION,
    CANCELLED: MessageKind.CANCELLATION,
    REMINDER: MessageKind.REMINDER,
    OFFER_INTRO: MessageKind.SLOT_OFFER,
}


class MissingFact(Exception):
    """A template needed a fact that `verified_facts` does not hold.

    Raised rather than rendered, because a message reading "su cita el �"
    is worse than no message: the patient cannot act on it and may believe the
    appointment is confirmed.
    """


def render(template: str, facts: VerifiedFacts) -> str:
    """Fill a template from verified facts only.

    The variable names here are the whole vocabulary a template may use. There
    is no path from a template to a value a tool did not establish, which is
    how templated text gets the provenance guarantee without a model in its
    way.
    """
    slot = facts.all_times()[0] if facts.all_times() else None
    values = {
        "date": slot.date_text if slot else MISSING,
        "time": slot.time_text if slot else MISSING,
        "doctor": facts.doctor_name or MISSING,
        "specialty": facts.specialty or MISSING,
        "location": facts.location_name or MISSING,
        "clinic": facts.clinic_name or MISSING,
        "clinic_phone": facts.clinic_phone or MISSING,
        "name": facts.patient_given_name or MISSING,
        "arrive": (
            str(facts.arrive_minutes_early) if facts.arrive_minutes_early is not None else MISSING
        ),
    }
    text = template.format(**values)
    if MISSING in text:
        missing = [
            key for key, value in values.items() if value == MISSING and f"{{{key}}}" in template
        ]
        raise MissingFact(f"template needs facts that are not verified: {', '.join(missing)}")
    return text


def render_offer(facts: VerifiedFacts) -> str:
    """The slot offer, numbered so a reply of "2" is unambiguous.

    Numbered rather than relying on the patient retyping a date: a reply of
    "2" is a choice this code can resolve against the offer it sent, while
    "el martes" is text that would have to be interpreted.
    """
    if not facts.offered_slots:
        raise MissingFact("an offer with no slots is not an offer")
    lines = [render(OFFER_INTRO, facts)]
    lines.extend(
        f"{index}. {slot.date_text} a las {slot.time_text}"
        for index, slot in enumerate(facts.offered_slots, start=1)
    )
    lines.append(OFFER_FOOTER)
    return "\n".join(lines)
