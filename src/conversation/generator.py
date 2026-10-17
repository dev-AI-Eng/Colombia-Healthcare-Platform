"""The generator: drafts one Spanish message from verified facts (ADR-02).

Rule 2 shapes this file more than anything else. **The generator never gets a
database session.** It receives a `verified_facts` object that deterministic
code built from tool calls, and nothing else that looks like a fact. There is
no query it can run, no record it can choose, and no patient it can name that a
tool did not already name.

Three further constraints, each with a reason:

**The facts are rendered by code before they reach the model.** ADR-02 says
"appointment facts rendered by code", and `facts.spell_date` / `spell_time` are
that: the model is handed "viernes 23 de octubre" and "10:30 a.m." as strings to
place in a sentence, never a timestamp to format. A model formatting 10:30 as
11:30 is the failure the whole pipeline exists to prevent, and the only way to
remove it is to not ask.

**The patient's own words are never in the prompt.** Not as context, not as
"the patient said". Rule 3 makes patient text data, and a generator that read it
would be a generator that could be instructed by it — prompt injection against
medical LLMs succeeded in 94.4% of evaluations at the decision turn (JAMA
Network Open, 19 Dec 2025). The *intent* reaches the generator, as an enum
deterministic code chose.

**A failure is a refusal, never a sentence.** No key, a timeout, an empty
response: all return a draft that cannot be approved, and the gate then does
what it always does. A generator that invented an apology on error would be
putting unchecked text one step from a patient.

What the model actually adds over `templates.py` is phrasing: a reply that
reads as written for this patient rather than filled from a form. It adds no
facts, and the gate refuses it if it tries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final, Protocol

from src.conversation.facts import VerifiedFacts
from src.conversation.state import ConversationState, Draft, Intent
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
    Intent.UNCLEAR: MessageKind.FALLBACK,
}

#: What the message must contain, per type, stated to the model as a
#: requirement rather than left to its judgement. The gate checks this
#: independently (stage 3) -- saying it here only reduces how often a draft is
#: refused for something avoidable.
MUST_STATE: Final[dict[MessageKind, str]] = {
    MessageKind.CONFIRMATION: "la fecha, la hora y el nombre del médico",
    MessageKind.CANCELLATION: "que la cita fue cancelada, y el teléfono de la clínica",
    MessageKind.SLOT_OFFER: "cada horario ofrecido, numerado",
    MessageKind.REMINDER: "la fecha y la hora",
    MessageKind.FALLBACK: "que una persona de la clínica se pondrá en contacto",
}

SYSTEM_PROMPT: Final = """\
Escribes mensajes de WhatsApp para los pacientes de una clínica colombiana.

REGLAS ABSOLUTAS:
1. Usa ÚNICAMENTE los datos que se te entregan. No inventes ni deduzcas \
fechas, horas, nombres, direcciones, números de teléfono ni duraciones. Si un \
dato no está en la lista, no existe para ti.
2. Copia las fechas y horas EXACTAMENTE como aparecen. Ya están escritas en el \
formato correcto; no las reformatees ni las recalcules.
3. Nunca des información médica: ni diagnóstico, ni dosis, ni tratamiento, ni \
interpretación de síntomas, ni tranquilización sobre la gravedad de algo.
4. Escribe en español de Colombia, con usted. Tono claro y respetuoso, sin \
tecnicismos. Dos o tres frases cortas.
5. No saludes con el nombre del paciente más de una vez, y no firmes.

Responde solo con el texto del mensaje. Sin comillas, sin explicaciones."""


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


def build_user_message(facts: VerifiedFacts, kind: MessageKind) -> str:
    """The exact payload sent to a provider. Nothing else is transmitted.

    Every value here came from a tool. Dates and times arrive pre-rendered by
    `facts.spell_date` and `spell_time`, so the model places strings in a
    sentence rather than formatting an instant -- ADR-02's "facts rendered by
    code", which is what removes the class of error where a model writes 11:30
    for a 10:30 appointment.

    The patient's own message is deliberately absent. Rule 3.
    """
    lines: list[str] = [f"Tipo de mensaje: {kind.value}", "", "Datos verificados:"]

    simple = (
        ("Nombre del paciente", facts.patient_given_name),
        ("Médico", facts.doctor_name),
        ("Especialidad", facts.specialty),
        ("Sede", facts.location_name),
        ("Dirección", facts.location_address),
        ("Clínica", facts.clinic_name),
        ("Teléfono de la clínica", facts.clinic_phone),
    )
    for label, value in simple:
        if value:
            lines.append(f"- {label}: {value}")

    if facts.arrive_minutes_early is not None:
        lines.append(f"- Llegar antes (minutos): {facts.arrive_minutes_early}")

    if facts.appointment_starts_at is not None:
        slot = facts.all_times()[0]
        lines.append(f"- Fecha de la cita: {slot.date_text}")
        lines.append(f"- Hora de la cita: {slot.time_text}")

    for index, slot in enumerate(facts.offered_slots, start=1):
        lines.append(f"- Horario {index}: {slot.date_text} a las {slot.time_text}")

    lines.extend(
        [
            "",
            f"El mensaje debe indicar {MUST_STATE[kind]}.",
        ]
    )
    return "\n".join(lines)


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
    """Draft one message from the verified facts in this state.

    Returns a non-templated draft, so the full gate applies including the model
    judge (ADR-13 exempts only pre-approved template wording). Never raises:
    every failure path is a draft the gate will refuse.
    """
    kind = KIND_FOR.get(state["intent"], MessageKind.FALLBACK)
    facts = state["verified_facts"]

    if client is None:
        if settings is None or not settings.openai_api_key.get_secret_value():
            return _refusal(kind, "no generator key configured")
        client = OpenAIGenerator(settings)

    if facts.is_empty() and kind is not MessageKind.FALLBACK:
        # Nothing verified to say. A model asked to write a confirmation with
        # no appointment would invent one, so it is not asked.
        return _refusal(kind, "no verified facts to state")

    try:
        text = client.complete(SYSTEM_PROMPT, build_user_message(facts, kind))
    except Exception as error:
        # Deliberately broad: an SDK raises a dozen types and all of them mean
        # the same thing here. The facts are not logged -- they are patient
        # data, and ADR-24 records the disclosure separately.
        logger.warning("generator unavailable: %s", type(error).__name__)
        return _refusal(kind, type(error).__name__)

    if not text.strip():
        return _refusal(kind, "empty response")

    return Draft(text=text.strip(), kind=kind, templated=False)
