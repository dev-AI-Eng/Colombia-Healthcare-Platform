"""The graph's state, and the one rule that governs what may go in it.

**Patient data must never be held in a channel whose value is a bare string,
number or boolean** (§9). LangGraph inlines scalar channel values into the
`checkpoints` row directly, where the encrypting serializer in
`checkpointer.py` never sees them — so a `patient_name: str` channel would
write a name to disk in the clear while every other path encrypted it.

The shape here follows from that. Patient-derived values live inside
`PatientContext` and `Draft`, which are structured objects the serializer does
encrypt. The bare channels are all non-patient: a thread id, a clinic id, an
attempt counter, enum codes. `inbound_text` is the one that looks like an
exception and is not — see its comment.

`ConversationState` is a TypedDict because that is what LangGraph reduces over;
the nested values are frozen dataclasses because they are facts about a moment
and nothing should edit one in place.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import TypedDict

from src.conversation.facts import VerifiedFacts
from src.conversation.validator import Finding, MessageKind


class Intent(StrEnum):
    """What the patient appears to want.

    A closed set. Adding a branch means editing the graph and its test, not
    prompting a model to invent a route (§9).
    """

    CONFIRM = "confirm"
    CANCEL = "cancel"
    RESCHEDULE = "reschedule"
    OPT_OUT = "opt_out"
    HUMAN = "human"
    #: Understood as a question the bot must not answer: anything clinical.
    CLINICAL = "clinical"
    UNCLEAR = "unclear"


class Stop(StrEnum):
    """Why the graph stopped, when it stopped early.

    Every value here ends the turn without a generated message, and each one
    has a templated reply or an escalation behind it. A graph that could stop
    for a reason not in this set would be a graph that goes quiet on a patient.
    """

    UNKNOWN_SENDER = "unknown_sender"
    SHARED_HANDSET = "shared_handset"
    NO_CONSENT = "no_consent"
    NO_APPOINTMENT = "no_appointment"
    NO_SLOTS = "no_slots"
    CLINICAL_CONCERN = "clinical_concern"
    ASKED_FOR_HUMAN = "asked_for_human"
    GATE_EXHAUSTED = "gate_exhausted"


@dataclass(frozen=True, slots=True)
class PatientContext:
    """Who the sender is, resolved deterministically from the binding.

    A structured value, not loose channels, for the §9 reason above: the given
    name is patient data and has to reach the encrypting serializer.

    Resolved by `identify` from the phone blind index — never from anything the
    patient wrote (rule 3). `candidates` holds every patient on a shared
    handset, because a household phone names several people and the graph must
    not guess which one is writing.
    """

    patient_id: uuid.UUID | None = None
    given_name: str | None = None
    candidates: tuple[uuid.UUID, ...] = ()

    @property
    def is_resolved(self) -> bool:
        return self.patient_id is not None

    @property
    def is_ambiguous(self) -> bool:
        return self.patient_id is None and len(self.candidates) > 1


@dataclass(frozen=True, slots=True)
class Draft:
    """A candidate outbound message and what the gate said about it.

    Structured because the text may quote patient data back, and because the
    retry loop needs the history: ADR-23 escalates "carrying the full draft
    history", which a single overwritten string could not provide.
    """

    text: str
    kind: MessageKind
    findings: tuple[Finding, ...] = ()
    approved: bool = False
    #: True when the text came from a pre-approved template rather than a
    #: model. Templated text still passes provenance (ADR-13), but never the
    #: model judge: there is no generated wording for it to judge.
    templated: bool = False


class ConversationState(TypedDict, total=False):
    """One turn of one conversation.

    `total=False` because LangGraph merges partial updates: a node returns only
    the keys it changed, and a node returning the whole state would overwrite
    a sibling's work.
    """

    # --- identity of the turn. Not patient data: ids and codes.
    thread_id: str
    clinic_id: uuid.UUID
    channel: str
    #: Blind index of the sending number, not the number. The number itself is
    #: patient data; its index is a lookup key that reveals nothing.
    sender_bidx: bytes
    now: dt.datetime

    # --- what arrived.
    #: The patient's own words. A bare string, which §9 forbids for patient
    #: data -- so this is deliberately **not** persisted: `prune_for_checkpoint`
    #: clears it before the state is written, and the audit log records that a
    #: message arrived without copying its text into a second store. Inside one
    #: turn it is held here because every node needs it as *data* (rule 3).
    inbound_text: str
    inbound_kind: str

    # --- what deterministic code worked out.
    patient: PatientContext
    intent: Intent
    appointment_id: uuid.UUID | None
    verified_facts: VerifiedFacts

    # --- the gate.
    drafts: tuple[Draft, ...]
    attempt: int
    stop: Stop | None
    reason_codes: tuple[str, ...]

    # --- what leaves.
    outbound_text: str | None
    escalation_id: uuid.UUID | None


def initial(
    *,
    thread_id: str,
    clinic_id: uuid.UUID,
    channel: str,
    sender_bidx: bytes,
    inbound_text: str,
    now: dt.datetime,
    inbound_kind: str = "text",
) -> ConversationState:
    """A turn's starting state.

    Every field a node reads is present, so no node has to guard against a
    missing key. `attempt` starts at 0 and is bounded at `MAX_ATTEMPTS`.
    """
    return ConversationState(
        thread_id=thread_id,
        clinic_id=clinic_id,
        channel=channel,
        sender_bidx=sender_bidx,
        now=now,
        inbound_text=inbound_text,
        inbound_kind=inbound_kind,
        patient=PatientContext(),
        intent=Intent.UNCLEAR,
        appointment_id=None,
        verified_facts=VerifiedFacts(),
        drafts=(),
        attempt=0,
        stop=None,
        reason_codes=(),
        outbound_text=None,
        escalation_id=None,
    )


def prune_for_checkpoint(state: ConversationState) -> ConversationState:
    """Drop what must not be written to the checkpoint row.

    `inbound_text` is the patient's own words in a bare string channel, which
    §9 forbids persisting: LangGraph would inline it into `checkpoints`
    unencrypted. It is needed within the turn and not after it, so it is
    cleared on the way out. The audit log already records that a message
    arrived; keeping the text would duplicate patient data into a second store
    for no further purpose.
    """
    pruned: ConversationState = state.copy()
    pruned["inbound_text"] = ""
    return pruned
