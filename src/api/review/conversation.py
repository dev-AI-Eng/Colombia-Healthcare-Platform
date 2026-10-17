"""One HTTP route over the conversation graph, for the M3 prototype.

Rev3's M3 deliverable is "full conversation flow demo on the prototyping
platform", validated on "a lightweight chatbot testing platform". This is that
surface: a staff screen posts what a patient would write and gets back the
reply, the intent, and — the part that matters for review — **what the gate
decided and why**.

Showing the verdict is the point. A demo that only showed the outgoing message
would prove the pipeline produces Spanish; it would not show that a fabricated
fact was caught, which draft attempt caught it, or that an unapproved draft
never reached the patient. Those are M3's exit criteria, and they are invisible
unless the route reports them.

It lives under `/review` for the same reason the M2 write routes do: that is
where the access gate already is. `require_synthetic_review` mounts it only in
synthetic-data mode, and `app.py` requires a loopback Host for the whole
surface.

No provider is called. `generate` renders a pre-approved template (ADR-13), so
the graph runs deterministically and the gate still inspects every draft. M4
swaps a model generator in behind the same seam, and nothing here changes.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Callable
from typing import Annotated, Final, cast

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from src.api.dependencies import ClinicScopeDep, SessionDep, SettingsDep
from src.conversation import templates as T
from src.conversation.generator import generate as generate_draft
from src.conversation.graph import Dependencies, build_graph
from src.conversation.judge import judge as model_judge
from src.conversation.state import ConversationState, Draft, initial
from src.conversation.validator import MessageKind, Verdict
from src.core.crypto import blind_index

router = APIRouter()

#: Which template the prototype's generator renders for each intent. In M4 a
#: model drafts these; the mapping is what makes M3 runnable without one, and
#: it is deliberately explicit rather than inferred, so a reviewer can see
#: which wording a given reply produced.
_TEMPLATE_FOR: Final[dict[str, tuple[MessageKind, str]]] = {
    "confirm": (MessageKind.CONFIRMATION, T.CONFIRMED),
    "cancel": (MessageKind.CANCELLATION, T.CANCELLED),
    "reschedule": (MessageKind.SLOT_OFFER, T.OFFER_INTRO),
    "unclear": (MessageKind.FALLBACK, T.STAFF_WILL_CALL),
}


class TurnIn(BaseModel):
    """What a patient would send.

    `sender_phone` is a phone number, not a patient id, and that is the point:
    the graph resolves the patient from the number's blind index exactly as a
    real channel would (ADR-18). Letting the caller name a patient directly
    would skip the step the demo is meant to show.
    """

    sender_phone: Annotated[str, Field(min_length=5, max_length=20)]
    text: Annotated[str, Field(min_length=0, max_length=2000)]
    #: Overridable so a reviewer can drive a hold to its expiry without
    #: waiting fifteen minutes. The engine takes the clock as a parameter
    #: everywhere for the same reason.
    now: dt.datetime | None = None


class FindingOut(BaseModel):
    reason: str
    span: str
    detail: str


class DraftOut(BaseModel):
    """One attempt, and what the gate said about it.

    Every attempt is returned, not just the last. ADR-23 escalates "carrying
    the full draft history", and a reviewer checking the retry bound needs to
    count them.
    """

    text: str
    kind: str
    approved: bool
    templated: bool
    findings: list[FindingOut]


class TurnOut(BaseModel):
    intent: str
    #: Set when the turn ended early, naming which of the eight exits it took.
    stop: str | None
    #: What would be sent to the patient. None means nothing is sent, which is
    #: only ever the result of a routing bug -- every designed path replies.
    outbound_text: str | None
    drafts: list[DraftOut]
    attempts: int
    reason_codes: list[str]
    patient_resolved: bool
    escalation_id: uuid.UUID | None
    #: Facts the message was allowed to state, so a reviewer can see that the
    #: reply contains nothing else (rule 4).
    verified_facts: dict[str, str]


def _prototype_generator() -> Callable[[ConversationState], Draft]:
    """Render the pre-approved template this intent implies.

    Returns an unapprovable draft when a fact is missing rather than raising,
    so the gate reports it and the retry bound applies -- a render failure is a
    draft that cannot be sent, which is exactly what the gate is for.
    """

    def generate(state: ConversationState) -> Draft:
        kind, template = _TEMPLATE_FOR.get(
            state["intent"].value, (MessageKind.FALLBACK, T.STAFF_WILL_CALL)
        )
        try:
            text = (
                T.render_offer(state["verified_facts"])
                if kind is MessageKind.SLOT_OFFER
                else T.render(template, state["verified_facts"])
            )
        except T.MissingFact as error:
            return Draft(text=f"[no renderizable: {error}]", kind=kind, templated=True)
        return Draft(text=text, kind=kind, templated=True)

    return generate


def _generator_for(settings: SettingsDep) -> Callable[[ConversationState], Draft]:
    """The model generator when a key is set, the templated one otherwise.

    Both are drafted from the same `verified_facts` and both face the same
    gate; the difference is phrasing, and whether a provider is reachable.
    Falling back rather than failing means the demo works on a machine with no
    keys -- which is every machine running the test suite (§6) -- while a
    configured install exercises the real generator/validator/evaluator
    pipeline M3's deliverable names.

    A generated draft is marked non-templated, so it also faces the model
    judge; a templated one does not, because ADR-13 exempts wording a person
    already approved.
    """
    if settings.openai_api_key.get_secret_value():

        def model_backed(state: ConversationState) -> Draft:
            return generate_draft(state, settings=settings)

        return model_backed
    return _prototype_generator()


def _facts_for_display(state: ConversationState) -> dict[str, str]:
    """The verified facts, flattened for a screen.

    Rendered through the same helpers the templates use, so what a reviewer
    reads is what a message could have said -- not a different formatting of
    the same instant.
    """
    facts = state["verified_facts"]
    out: dict[str, str] = {}
    for name in ("doctor_name", "specialty", "clinic_name", "clinic_phone", "patient_given_name"):
        value = getattr(facts, name, None)
        if value:
            out[name] = str(value)
    if facts.arrive_minutes_early is not None:
        out["arrive_minutes_early"] = str(facts.arrive_minutes_early)
    for index, slot in enumerate(facts.all_times()):
        label = "appointment" if index == 0 and facts.appointment_starts_at else f"slot_{index}"
        out[label] = f"{slot.date_text} {slot.time_text}"
    return out


@router.post(
    "/conversation/turn",
    response_model=TurnOut,
    status_code=status.HTTP_200_OK,
    summary="Run one conversation turn through the graph",
)
async def run_turn(
    body: TurnIn,
    session: SessionDep,
    scope: ClinicScopeDep,
    settings: SettingsDep,
) -> TurnOut:
    """Drive the real graph: identify, consent, classify, act, gate, dispatch.

    Always 200, even when the turn ends early or nothing is sent. An early exit
    is a designed outcome with its own template, not an error, and returning
    4xx for "this number is not registered" would make the demo look broken
    where it is working.
    """
    now = body.now or dt.datetime.now(tz=dt.UTC)

    def judge(text: str) -> Verdict:
        # The real judge when both provider keys are set; otherwise it reports
        # unavailable and the graph escalates. Templated drafts never reach it
        # (ADR-13), so the prototype runs either way.
        return model_judge(text, settings=settings)

    graph = build_graph(
        Dependencies(
            session=session,
            settings=settings,
            generate=_generator_for(settings),
            judge=judge,
        )
    )
    state = initial(
        thread_id=f"{scope.clinic_id}:{blind_index(body.sender_phone).hex()[:16]}",
        clinic_id=scope.clinic_id,
        channel="prototype",
        sender_bidx=blind_index(body.sender_phone),
        inbound_text=body.text,
        now=now,
    )
    # `ainvoke` is typed as returning a plain dict, so the narrowing
    # happens once here rather than at every read below.
    final = cast(ConversationState, await graph.ainvoke(state))

    return TurnOut(
        intent=final["intent"].value,
        stop=stopped.value if (stopped := final.get("stop")) else None,
        outbound_text=final.get("outbound_text"),
        drafts=[
            DraftOut(
                text=draft.text,
                kind=draft.kind.value,
                approved=draft.approved,
                templated=draft.templated,
                findings=[
                    FindingOut(reason=f.reason.value, span=f.span, detail=f.detail)
                    for f in draft.findings
                ],
            )
            for draft in final.get("drafts", ())
        ],
        attempts=final.get("attempt", 0),
        reason_codes=list(final.get("reason_codes", ())),
        patient_resolved=final["patient"].is_resolved,
        escalation_id=final.get("escalation_id"),
        verified_facts=_facts_for_display(final),
    )
