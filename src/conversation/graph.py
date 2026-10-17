"""The conversation graph (ADR-03, §9).

```
[identify] ── unknown ──► templated UNKNOWN_SENDER ──► END
     │        ambiguous ─► templated SHARED_HANDSET ──► END
     ▼
[consent] ─── none ──────► templated NO_CONSENT + escalate ──► END
     ▼
[classify] ── clinical ──► templated CLINICAL_TO_HUMAN + escalate ──► END
     │        human ─────► templated STAFF_WILL_CALL + escalate ──► END
     │        opt_out ───► templated OPTED_OUT ──► END
     │
     ├ confirm ────► [act: confirm_appointment] ─┐
     ├ cancel ─────► [act: cancel_appointment] ──┤
     ├ reschedule ─► [act: find_slots]           │
     └ unclear ────────────────────────────────► │
                                                 ▼
                                            [compose]
                                                 ▼
                                             [gate] ◄── retry, max 2 (ADR-23)
                                                 ▼
                                            [dispatch] ──► END
```

**The graph is fixed.** Adding a branch means editing this file and its test,
never prompting a model to decide a route (§9). Every edge above is a
`add_conditional_edges` on a value deterministic code computed.

**Side effects sit after the interrupt, not before.** §9: node resumption
re-executes a node from its start, so a write placed before an interrupt runs
twice. `act` is the only node that writes, it is its own node, and every write
it makes is idempotent on a key derived from the thread.

**Patient text never routes.** `classify` reads `inbound_text` to produce an
`Intent` from a closed set, and that enum is what the edges switch on. The text
selects no patient, no appointment and no tool (rule 3); `identify` resolved
those from the sender's blind index before `classify` ran.

M3 is the prototype (ADR-05), so this module composes the pipeline and proves
it end to end. Dispatch hands the approved text back to the caller rather than
to a provider: the channel layer is M4, and inventing it here would mean
building against WhatsApp before the legal question about it is answered.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from functools import partial
from typing import Final, Protocol

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.conversation import templates as T
from src.conversation import tools
from src.conversation.classify import classify_intent
from src.conversation.facts import FactsBuilder, VerifiedFacts
from src.conversation.state import (
    ConversationState,
    Draft,
    Intent,
    PatientContext,
    Stop,
)
from src.conversation.validator import Verdict, validate
from src.core.config import Settings
from src.identity.models import Consent, ConsentPurpose, PhoneBinding
from src.registry.models import Clinic, Patient
from src.scheduling import offers
from src.scheduling.models import ACTIVE_STATUSES, Appointment, EscalationReason


class Node(Protocol):
    """A graph node: reads the state, returns a partial update.

    A Protocol rather than a `Callable` alias because every node takes `deps`
    keyword-only, which `Callable[...]` cannot express.
    """

    def __call__(
        self, state: ConversationState, *, deps: Dependencies
    ) -> Awaitable[ConversationState]: ...


#: ADR-23. Two retries, then a pre-approved static template and a human. The
#: counter lives in graph state so a resumed thread cannot reset it.
MAX_ATTEMPTS: Final = 2


@dataclass(frozen=True, slots=True)
class Dependencies:
    """What the graph needs from outside itself.

    Passed in rather than imported, so a test can supply a deterministic
    generator and judge and never reach a provider (§6). `session` is here and
    nowhere near the generator: rule 2.
    """

    session: AsyncSession
    settings: Settings
    #: Drafts a message from verified facts. In M3's prototype this is the
    #: templated renderer; M4 swaps in the model generator behind the same
    #: signature, and the gate does not change either way.
    generate: Callable[[ConversationState], Draft]
    #: Stage 4. Returns a verdict and can only withhold approval.
    judge: Callable[[str], Verdict]


# ------------------------------------------------------------------- identify


async def identify(state: ConversationState, *, deps: Dependencies) -> ConversationState:
    """Resolve the sender to a patient, from the binding only (ADR-18).

    Looked up by blind index, so the number is never decrypted to find it. A
    number on no binding is a stranger; a number on several is a household
    handset and the graph asks rather than guessing — a wrong guess discloses
    one patient's appointment to another person in the same house.
    """
    rows = (
        await deps.session.execute(
            select(PhoneBinding.patient_id)
            .where(
                PhoneBinding.clinic_id == state["clinic_id"],
                PhoneBinding.phone_e164_bidx == state["sender_bidx"],
                PhoneBinding.revoked_at.is_(None),
            )
            .order_by(PhoneBinding.created_at)
        )
    ).scalars()
    patient_ids = tuple(dict.fromkeys(rows))

    if not patient_ids:
        return {"patient": PatientContext(), "stop": Stop.UNKNOWN_SENDER}
    if len(patient_ids) > 1:
        return {
            "patient": PatientContext(candidates=patient_ids),
            "stop": Stop.SHARED_HANDSET,
        }

    patient = (
        await deps.session.execute(select(Patient).where(Patient.id == patient_ids[0]))
    ).scalar_one_or_none()
    if patient is None or patient.deleted_at is not None:
        # A binding to a soft-deleted patient is a stranger as far as this
        # conversation goes (§7: deleted patients appear in no query).
        return {"patient": PatientContext(), "stop": Stop.UNKNOWN_SENDER}

    return {
        "patient": PatientContext(
            patient_id=patient.id,
            given_name=patient.given_names.split()[0] if patient.given_names else None,
        )
    }


# -------------------------------------------------------------------- consent


async def consent(state: ConversationState, *, deps: Dependencies) -> ConversationState:
    """Require a live consent for appointment messaging (ADR-19).

    Checked before anything is said, not before sending: a patient with no
    consent gets one templated reply telling them how to give it, and that
    reply is the only thing this channel may carry for them.
    """
    patient_id = state["patient"].patient_id
    if patient_id is None:  # pragma: no cover - identify stopped already
        return {"stop": Stop.UNKNOWN_SENDER}

    found = (
        await deps.session.execute(
            select(Consent.id).where(
                Consent.clinic_id == state["clinic_id"],
                Consent.patient_id == patient_id,
                Consent.purpose == ConsentPurpose.APPOINTMENT_MESSAGING,
                Consent.revoked_at.is_(None),
            )
        )
    ).first()

    if found is None:
        return {"stop": Stop.NO_CONSENT}
    return {}


# ------------------------------------------------------------------- classify


#: The three intents that end the turn with a templated reply, and the stop
#: reason each becomes. Set here rather than in the routing function: a
#: conditional edge decides *where* to go, and a node decides *what to say*.
_TERMINAL_INTENTS: Final[dict[Intent, Stop]] = {
    Intent.CLINICAL: Stop.CLINICAL_CONCERN,
    Intent.HUMAN: Stop.ASKED_FOR_HUMAN,
    Intent.OPT_OUT: Stop.ASKED_FOR_HUMAN,
}


async def classify(state: ConversationState, *, deps: Dependencies) -> ConversationState:
    """Read the patient's words and produce one enum from a closed set.

    This is the only node that looks at `inbound_text`, and all it may produce
    is an `Intent`. The text chooses no record and no tool (rule 3).

    An intent that ends the turn also sets its stop reason here, so the stop
    node always has one: a `stop` of None would fail the template lookup, and
    a patient would get silence.
    """
    intent = classify_intent(state["inbound_text"])
    update: ConversationState = {"intent": intent}
    if intent in _TERMINAL_INTENTS:
        update["stop"] = _TERMINAL_INTENTS[intent]
    return update


# ------------------------------------------------------------------------ act
#
# The only node that writes. Its own node, after any interrupt, with every
# write idempotent on a key derived from the thread (§9).


async def act(state: ConversationState, *, deps: Dependencies) -> ConversationState:
    """Call the one tool this intent implies, and collect the facts it returns."""
    patient_id = state["patient"].patient_id
    if patient_id is None:
        # `identify` stops the turn before this node for an unresolved sender,
        # so arriving here means a routing bug. Checked rather than asserted:
        # `assert` is stripped under `-O`, and a patient-facing path must not
        # have a guard that disappears with an interpreter flag. The answer is
        # the stranger's template, which names no patient.
        return {"stop": Stop.UNKNOWN_SENDER}

    facts = await _clinic_facts(state, deps)
    intent = state["intent"]
    appointment = await _active_appointment(state, deps, patient_id)

    if intent in {Intent.CONFIRM, Intent.CANCEL}:
        if appointment is None:
            return {"stop": Stop.NO_APPOINTMENT, "verified_facts": facts.build()}
        if intent is Intent.CONFIRM:
            if appointment.status not in tools.CONFIRMABLE:
                return {"stop": Stop.NO_APPOINTMENT, "verified_facts": facts.build()}
            result = await tools.confirm_appointment(
                deps.session,
                clinic_id=state["clinic_id"],
                appointment_id=appointment.id,
                now=state["now"],
            )
            if not result.ok:
                return {"stop": Stop.NO_APPOINTMENT, "verified_facts": facts.build()}
        else:
            result = await tools.cancel_appointment(
                deps.session,
                clinic_id=state["clinic_id"],
                appointment_id=appointment.id,
                reason_code="patient_request",
            )
        return {
            "appointment_id": appointment.id,
            "verified_facts": _merge(facts, result.facts),
        }

    if intent is Intent.RESCHEDULE:
        if appointment is None:
            return {"stop": Stop.NO_APPOINTMENT, "verified_facts": facts.build()}
        result = await tools.find_slots(
            deps.session,
            clinic_id=state["clinic_id"],
            doctor_id=appointment.doctor_id,
            appointment_type_id=appointment.appointment_type_id,
            now=state["now"],
        )
        if not result.ok:
            return {"stop": Stop.NO_SLOTS, "verified_facts": facts.build()}
        return {
            "appointment_id": appointment.id,
            "verified_facts": _merge(facts, result.facts),
        }

    # UNCLEAR reached `act` only to gather the clinic's own facts, so the
    # fallback template can name the clinic's phone number.
    return {"verified_facts": facts.build()}


#: Where a clinic's public contact number lives. `clinics.settings` is a JSONB
#: column that existed unused until now, which is the right home for a value
#: the schema has no dedicated field for: a migration adding `clinics.phone`
#: belongs to M4, where the channel layer needs it for real, rather than being
#: slipped into M3 as a side effect of writing a template.
CLINIC_PHONE_SETTING: Final = "public_phone_e164"

#: How early a patient is asked to arrive. A clinic policy rather than a
#: clinical instruction, which is why a template may state it at all — but it
#: is a number the clinic should own, so it reads from settings with this as
#: the fallback.
DEFAULT_ARRIVE_MINUTES: Final = 30


async def _clinic_facts(state: ConversationState, deps: Dependencies) -> FactsBuilder:
    """The clinic's own details, which several templates need.

    A clinic with no configured phone number yields no `clinic_phone` fact,
    and the templates that need one then raise `MissingFact` rather than
    sending "llame al �". That is the correct failure: a cancellation
    telling a patient to call a number that is not there is worse than an
    escalation telling staff to call the patient.
    """
    builder = FactsBuilder()
    clinic = (
        await deps.session.execute(select(Clinic).where(Clinic.id == state["clinic_id"]))
    ).scalar_one_or_none()
    if clinic is not None:
        config = clinic.settings or {}
        phone = config.get(CLINIC_PHONE_SETTING)
        builder.set(
            clinic_name=clinic.name,
            clinic_phone=str(phone) if phone else None,
            arrive_minutes_early=int(config.get("arrive_minutes_early", DEFAULT_ARRIVE_MINUTES)),
        )
    builder.set(patient_given_name=state["patient"].given_name)
    return builder


async def _active_appointment(
    state: ConversationState, deps: Dependencies, patient_id: uuid.UUID
) -> Appointment | None:
    """The patient's next active appointment, resolved by code not by text."""

    return (
        await deps.session.execute(
            select(Appointment)
            .where(
                Appointment.clinic_id == state["clinic_id"],
                Appointment.patient_id == patient_id,
                Appointment.status.in_([s.value for s in ACTIVE_STATUSES]),
            )
            .order_by(Appointment.during)
            .limit(1)
        )
    ).scalar_one_or_none()


def _merge(base: FactsBuilder, extra: FactsBuilder) -> VerifiedFacts:
    """Combine clinic facts with what a tool established."""
    built = extra.build()
    base.set(
        appointment_starts_at=built.appointment_starts_at,
        appointment_duration_minutes=built.appointment_duration_minutes,
        doctor_name=built.doctor_name,
    )
    merged = base.build()
    return replace(merged, offered_slots=built.offered_slots)


# -------------------------------------------------------------------- compose


async def compose(state: ConversationState, *, deps: Dependencies) -> ConversationState:
    """Produce a candidate message.

    In M3's prototype this renders a pre-approved template (ADR-13), so the
    whole pipeline is exercised without a provider. The seam is
    `deps.generate`: M4 passes a model generator with the same signature, and
    the gate below does not change.
    """
    draft = deps.generate(state)
    return {"drafts": (*state["drafts"], draft)}


# ----------------------------------------------------------------------- gate


async def gate(state: ConversationState, *, deps: Dependencies) -> ConversationState:
    """Stages 1-3, then stage 4 for generated text only (ADR-22).

    A templated draft skips the model judge: its wording was approved by a
    person, there is no generated text to judge, and sending it to a judge
    would introduce a model into a path that has none (ADR-13).
    """
    draft = state["drafts"][-1]
    verdict = validate(draft.text, state["verified_facts"], draft.kind)

    if verdict.approved and not draft.templated:
        verdict = deps.judge(draft.text)

    checked = replace(draft, findings=verdict.findings, approved=verdict.approved)
    updated = (*state["drafts"][:-1], checked)

    if verdict.approved:
        return {"drafts": updated, "reason_codes": ()}

    return {
        "drafts": updated,
        "attempt": state["attempt"] + 1,
        "reason_codes": verdict.reason_codes,
    }


# ------------------------------------------------------------------- dispatch


async def dispatch(state: ConversationState, *, deps: Dependencies) -> ConversationState:
    """Hand the approved text to the caller. Nothing unapproved gets here.

    The last line of defence rather than the first: `gate` already approved
    this draft, and this re-reads the flag instead of trusting the routing. A
    future edge added to the graph by mistake then produces no message rather
    than an unchecked one.
    """
    draft = state["drafts"][-1] if state["drafts"] else None
    if draft is None or not draft.approved:
        return {"outbound_text": None}
    return {"outbound_text": draft.text}


# ------------------------------------------------------------------- stopping


async def stop_with_template(state: ConversationState, *, deps: Dependencies) -> ConversationState:
    """Send the one pre-approved reply this stop reason has, and escalate if
    a person needs to act.

    Every `Stop` has a template, so the graph never goes quiet on a patient.
    The templates that name no patient facts — unknown sender, shared handset —
    are the ones that must not, because the clinic does not know who is
    writing.
    """
    # Every edge into this node sets a stop reason first. Reaching it without
    # one would mean a routing bug, and the right answer to that is the
    # fact-free template plus an escalation -- not a KeyError, and certainly
    # not silence on a patient who is waiting.
    reason = state.get("stop") or Stop.GATE_EXHAUSTED
    text, escalate_as = _STOP_REPLIES[reason]

    escalation_id: uuid.UUID | None = None
    if escalate_as is not None:
        row = await offers.escalate(
            deps.session,
            clinic_id=state["clinic_id"],
            reason=escalate_as,
            detail_es=_ESCALATION_DETAIL[reason],
            patient_id=state["patient"].patient_id,
            appointment_id=state.get("appointment_id"),
            context={"thread_id": state["thread_id"], "stop": reason.value},
        )
        escalation_id = row.id

    # Rendered where it needs facts, and the fact-free templates are returned
    # as they are. A stop reply that cannot render is itself a failure, so it
    # falls back to the one template that needs nothing.
    try:
        rendered = T.render(text, state["verified_facts"]) if "{" in text else text
    except T.MissingFact:
        rendered = T.STAFF_WILL_CALL

    return {"outbound_text": rendered, "escalation_id": escalation_id}


#: Stop reason -> (template, escalation reason or None).
#: A stop that no human needs to see has None: an unrecognised number is
#: usually a wrong number, and an escalation per wrong number is a queue
#: nobody triages.
_STOP_REPLIES: Final[dict[Stop, tuple[str, EscalationReason | None]]] = {
    Stop.UNKNOWN_SENDER: (T.UNKNOWN_SENDER, None),
    Stop.SHARED_HANDSET: (T.SHARED_HANDSET, None),
    Stop.NO_CONSENT: (T.NO_CONSENT, EscalationReason.MISSING_CONSENT),
    Stop.NO_APPOINTMENT: (T.NO_APPOINTMENT, None),
    Stop.NO_SLOTS: (T.NO_SLOTS, EscalationReason.NO_ACCEPTABLE_SLOT),
    Stop.CLINICAL_CONCERN: (T.CLINICAL_TO_HUMAN, EscalationReason.CLINICAL_CONCERN),
    Stop.ASKED_FOR_HUMAN: (T.STAFF_WILL_CALL, EscalationReason.PATIENT_ASKED_FOR_A_HUMAN),
    Stop.GATE_EXHAUSTED: (T.STAFF_WILL_CALL, EscalationReason.EVALUATOR_EXHAUSTED),
}

_ESCALATION_DETAIL: Final[dict[Stop, str]] = {
    Stop.NO_CONSENT: "El paciente escribió sin autorización vigente para este canal.",
    Stop.NO_SLOTS: "No hay horarios disponibles que el paciente pueda tomar.",
    Stop.CLINICAL_CONCERN: "El paciente hizo una consulta clínica por este canal.",
    Stop.ASKED_FOR_HUMAN: "El paciente pidió hablar con una persona.",
    Stop.GATE_EXHAUSTED: "El evaluador rechazó el mensaje dos veces; no se envió nada generado.",
    Stop.UNKNOWN_SENDER: "Número no reconocido.",
    Stop.SHARED_HANDSET: "Número compartido por varios pacientes.",
    Stop.NO_APPOINTMENT: "No hay cita activa.",
}


# -------------------------------------------------------------------- routing


def _after_identify(state: ConversationState) -> str:
    return "stop" if state.get("stop") else "consent"


def _after_consent(state: ConversationState) -> str:
    return "stop" if state.get("stop") else "classify"


def _after_classify(state: ConversationState) -> str:
    """Intent to node. A closed mapping: an intent with no entry cannot route,
    which is a loud failure rather than a silent send."""
    intent = state["intent"]
    if intent is Intent.CLINICAL:
        return "clinical"
    if intent is Intent.HUMAN:
        return "human"
    if intent is Intent.OPT_OUT:
        return "opt_out"
    return "act"


def _after_act(state: ConversationState) -> str:
    return "stop" if state.get("stop") else "compose"


def _after_gate(state: ConversationState) -> str:
    """Approved, retry, or exhausted (ADR-23).

    Retries are bounded at two and the counter is in state, so a resumed
    thread cannot reset it. Exhaustion sends the static template and raises an
    escalation carrying the draft history — never silence, never an unchecked
    send.
    """
    if state["drafts"][-1].approved:
        return "dispatch"
    if state["attempt"] >= MAX_ATTEMPTS:
        return "exhausted"
    return "compose"


async def exhausted(state: ConversationState, *, deps: Dependencies) -> ConversationState:
    """ADR-23's landing place: set the stop reason, then let `stop` reply.

    Its own node rather than a branch inside `gate`, because `gate` judges a
    draft and this records that judging is over. The static template it leads
    to carries no appointment facts, so it passes the provenance stage from any
    state -- which is what makes it safe to land on when everything else failed.
    """
    return {"stop": Stop.GATE_EXHAUSTED}


def build_graph(
    deps: Dependencies,
) -> CompiledStateGraph[ConversationState, None, ConversationState, ConversationState]:
    """Compile the graph. Fixed shape; `deps` supplies the seams.

    Not compiled with a checkpointer here: `run_turn` supplies one, so a caller
    that does not want persistence — a test, the demo screen — can drive the
    same graph without a connection for checkpoints.
    """

    def bind(node: Node):  # type: ignore[no-untyped-def]  # see below
        """Partially apply `deps` so a node becomes a plain state function.

        Two details here are load-bearing against LangGraph's typing, and both
        cost a detour to find:

        `functools.partial` rather than a closure, because it preserves the
        callable's signature and `add_node`'s overloads resolve against that.

        No return annotation, because writing one erases what those overloads
        need to see: an explicit `Callable[[State], Awaitable[State]]` is
        rejected while the inferred `partial[...]` is accepted.
        """
        return partial(node, deps=deps)

    # Parameterised with the real state type so `add_node` can infer each
    # node's input. With `Any` the overloads do not resolve at all.
    builder: StateGraph[ConversationState, None, ConversationState, ConversationState] = StateGraph(
        ConversationState
    )
    builder.add_node("identify", bind(identify))
    builder.add_node("consent", bind(consent))
    builder.add_node("classify", bind(classify))
    builder.add_node("act", bind(act))
    builder.add_node("compose", bind(compose))
    builder.add_node("gate", bind(gate))
    builder.add_node("dispatch", bind(dispatch))
    builder.add_node("stop", bind(stop_with_template))
    builder.add_node("exhausted", bind(exhausted))

    builder.add_edge(START, "identify")
    builder.add_conditional_edges(
        "identify", _after_identify, {"stop": "stop", "consent": "consent"}
    )
    builder.add_conditional_edges(
        "consent", _after_consent, {"stop": "stop", "classify": "classify"}
    )
    builder.add_conditional_edges(
        "classify",
        _after_classify,
        {"act": "act", "clinical": "stop", "human": "stop", "opt_out": "stop"},
    )
    builder.add_conditional_edges("act", _after_act, {"stop": "stop", "compose": "compose"})
    builder.add_edge("compose", "gate")
    builder.add_conditional_edges(
        "gate",
        _after_gate,
        {"dispatch": "dispatch", "compose": "compose", "exhausted": "exhausted"},
    )
    builder.add_edge("exhausted", "stop")
    builder.add_edge("dispatch", END)
    builder.add_edge("stop", END)
    return builder.compile()


__all__ = ["MAX_ATTEMPTS", "Dependencies", "build_graph"]
