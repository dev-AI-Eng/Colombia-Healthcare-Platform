"""The generator: writes the wording around facts that code renders (ADR-02).

ADR-02, as Rev3 states it: "Appointment facts (date, time, doctor, site) are
never written by the LLM ... code renders the facts from the database, and a
deterministic validator rejects any draft whose free text contains a date,
time, number or doctor name."

So the model is given **no facts at all** -- not the date, not the doctor, not
the patient's name. It is told only what kind of message this is, and writes
one or two sentences of courtesy for it. Code then appends the facts block,
rendered by the same pre-approved templates the templated path uses
(`templates.py`), so every date, time, name and phone number a patient reads
came from the database through code. The gate checks the model's wording on
its own first (`validator.check_authored`): any digit, date, weekday, month,
relative day, spelled-out number, name or contact detail in it is refused,
even one that happens to be right, because the model had nothing to copy it
from.

An earlier version handed the model the verified facts as pre-rendered strings
and relied on the validator to catch a mis-copied one. That is a weaker
boundary than the one Rev3 agreed, and it also sent the patient's name and
appointment to the provider on every turn. Neither is true now.

Two constraints carried over unchanged:

**The patient's own words are never in the prompt.** Rule 3 makes patient text
data; a generator that read it could be instructed by it -- prompt injection
against medical LLMs succeeded in 94.4% of evaluations at the decision turn
(JAMA Network Open, 19 Dec 2025). The *intent* reaches the generator, as an
enum deterministic code chose.

**A failure is a refusal, never a sentence.** No key, a timeout, an empty
response: all return a draft that cannot be approved, and the gate then does
what it always does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final, Protocol

from src.conversation.facts import VerifiedFacts
from src.conversation.state import ConversationState, Draft, Intent
from src.conversation.templates import (
    CANCELLED,
    CONFIRMED,
    REMINDER,
    STAFF_WILL_CALL,
    MissingFact,
    render,
    render_offer,
)
from src.conversation.validator import MessageKind
from src.core.config import Settings

logger = logging.getLogger(__name__)

#: Which message type each intent produces. The same mapping the templated
#: path uses, because the gate's omission stage keys on it and the two paths
#: must be judged by the same standard.
KIND_FOR: Final[dict[Intent, MessageKind]] = {
    Intent.CONFIRM: MessageKind.CONFIRMATION,
    Intent.CANCEL: MessageKind.CANCELLATION,
    Intent.RESCHEDULE: MessageKind.SLOT_OFFER,
    Intent.CHOOSE: MessageKind.CONFIRMATION,
    Intent.UNCLEAR: MessageKind.FALLBACK,
}

#: Which pre-approved template renders the facts for each kind. The offer is
#: rendered by `render_offer`, because the number of slots varies.
FACTS_TEMPLATE: Final[dict[MessageKind, str]] = {
    MessageKind.CONFIRMATION: CONFIRMED,
    MessageKind.CANCELLATION: CANCELLED,
    MessageKind.REMINDER: REMINDER,
    MessageKind.FALLBACK: STAFF_WILL_CALL,
}

#: What each message is about, told to the model in words that carry no fact.
PURPOSE: Final[dict[MessageKind, str]] = {
    MessageKind.CONFIRMATION: "La cita del paciente quedó confirmada.",
    MessageKind.CANCELLATION: "La cita del paciente fue cancelada.",
    MessageKind.SLOT_OFFER: "Se le ofrecen al paciente horarios para reprogramar su cita.",
    MessageKind.REMINDER: "Se le recuerda al paciente su próxima cita.",
    MessageKind.FALLBACK: "Una persona de la clínica se pondrá en contacto con el paciente.",
}

SYSTEM_PROMPT: Final = """\
Escribes la frase inicial de un mensaje de WhatsApp de una clínica colombiana \
a uno de sus pacientes. El sistema de la clínica añade después, por su cuenta, \
todos los datos de la cita.

REGLAS ABSOLUTAS:
1. No escribas NINGÚN dato: ni fechas, ni días, ni meses, ni horas, ni \
números (tampoco en letras), ni nombres de personas, médicos o lugares, ni \
direcciones, teléfonos o correos. No los conoces; el sistema los añade.
2. Nunca des información médica: ni diagnóstico, ni dosis, ni tratamiento, ni \
interpretación de síntomas, ni tranquilización sobre la gravedad de algo.
3. Escribe en español de Colombia, con usted. Tono claro y respetuoso. Una o \
dos frases cortas. No saludes por el nombre y no firmes.

Responde solo con esas frases. Sin comillas, sin explicaciones."""


class GeneratorClient(Protocol):
    """What `generate` needs from a provider.

    A Protocol so tests pass a deterministic double and never reach a network
    (§6). The real implementation is `OpenAIGenerator`.
    """

    def complete(self, system: str, user: str) -> str: ...


@dataclass(frozen=True, slots=True)
class OpenAIGenerator:
    """The real generator: OpenAI, which is a different family from the judge.

    Different by configuration, not by accident: the judge defaults to Groq's
    open-weights model, and `conversation_llm_available` requires both keys for
    exactly this reason (CLAUDE.md §11 — a model grading its own prose is not a
    second opinion).
    """

    settings: Settings

    def complete(self, system: str, user: str) -> str:
        from openai import OpenAI

        client = OpenAI(
            api_key=self.settings.openai_api_key.get_secret_value(),
            timeout=self.settings.conversation_timeout_seconds,
            max_retries=1,
        )
        response = client.chat.completions.create(
            model=self.settings.generator_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
        )
        return (response.choices[0].message.content or "").strip()


def build_user_message(kind: MessageKind) -> str:
    """The exact payload sent to a provider. Nothing else is transmitted.

    The message type and what it is about, and nothing more: no patient, no
    appointment, no clinic. The provider learns that some patient of some
    clinic is having an appointment confirmed, which is not patient data.
    """
    return f"Tipo de mensaje: {kind.value}\nSituación: {PURPOSE[kind]}"


def facts_block(facts: VerifiedFacts, kind: MessageKind) -> str:
    """The facts a patient reads, rendered by code from verified facts only.

    The same pre-approved wording the templated path sends, so a generated
    message and a templated one state their facts identically. Raises
    `MissingFact` when a fact the block needs was not verified.
    """
    if kind is MessageKind.SLOT_OFFER:
        return render_offer(facts)
    return render(FACTS_TEMPLATE[kind], facts)


def _refusal(kind: MessageKind, detail: str) -> Draft:
    """A draft the gate cannot approve, carrying why.

    Returned rather than raised so the retry bound applies and the escalation
    carries the reason. The text is a marker, not Spanish: it must never be
    mistaken for something sendable, and the gate refuses it either way because
    it states no required fact.
    """
    return Draft(text=f"[generator unavailable: {detail}]", kind=kind, templated=False)


def generate(
    state: ConversationState,
    *,
    client: GeneratorClient | None = None,
    settings: Settings | None = None,
) -> Draft:
    """Draft one message: model wording, then the facts block code rendered.

    Returns a non-templated draft carrying the model's words in `authored`, so
    the gate checks them on their own (they may state no fact) and the judge
    reads them (ADR-01). Never raises: every failure path is a draft the gate
    will refuse.
    """
    kind = KIND_FOR.get(state["intent"], MessageKind.FALLBACK)
    facts = state["verified_facts"]

    try:
        block = facts_block(facts, kind)
    except MissingFact as missing:
        # Nothing verified to state, so the model is not asked: there would
        # be no facts to attach its words to.
        return _refusal(kind, str(missing))

    if client is None:
        if settings is None or not settings.openai_api_key.get_secret_value():
            return _refusal(kind, "no generator key configured")
        client = OpenAIGenerator(settings)

    try:
        authored = client.complete(SYSTEM_PROMPT, build_user_message(kind)).strip()
    except Exception as error:
        # Deliberately broad: an SDK raises a dozen types and all of them mean
        # the same thing here.
        logger.warning("generator unavailable: %s", type(error).__name__)
        return _refusal(kind, type(error).__name__)

    if not authored:
        return _refusal(kind, "empty response")

    return Draft(text=f"{authored}\n\n{block}", kind=kind, templated=False, authored=authored)
