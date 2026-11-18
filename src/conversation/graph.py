"""The conversation graph (ADR-03, §9).

```
[identify] ── unknown ──► templated UNKNOWN_SENDER ──► END
     │        ambiguous ─► templated SHARED_HANDSET ──► END
     ▼
[consent] ─── none ──────► templated NO_CONSENT + escalate ──► END
     ▼        withdrawn ─► continues, marked on the patient context
[classify] ── clinical ──► templated CLINICAL_TO_HUMAN + escalate ──► END
     │        human ─────► templated STAFF_WILL_CALL + escalate ──► END
     │        withdrawn, any other intent but opt_out
     │          ─────────► templated OPTED_OUT ──► END
     │
     ├ opt_out ────► [act: withdraw_messaging_consent] ──► templated OPTED_OUT ──► END
     ├ confirm ────► [act: confirm_appointment] ─┐
     ├ cancel ─────► [act: cancel_appointment] ──┤
     ├ reschedule ─► [act: find_slots ──► hold_slot per date ──► record_offer_round]
     │   (5 dates, one per day; a second round never repeats the first;
     │    after two rounds turned down: templated NO_MORE_OFFERS + escalate ──► END)
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

import datetime as dt
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from functools import partial
from typing import Final, Protocol

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.audit.context import AccessAction
from src.audit.service import Access, record_access
from src.conversation import templates as T
from src.conversation import tools
from src.conversation.classify import chosen_slot, classify_intent, declines_offer
from src.conversation.facts import FactsBuilder, VerifiedFacts
from src.conversation.state import (
    ConversationState,
    Draft,
    Intent,
    PatientContext,
    Stop,
)
from src.conversation.validator import Verdict, check_authored, validate
from src.core.config import Settings
from src.core.timezones import BOGOTA
from src.identity.models import Consent, ConsentPurpose, PhoneBinding
from src.registry import repository as registry
from src.registry.models import Clinic, Doctor
from src.scheduling import booking, offers
from src.scheduling.models import (
    ACTIVE_STATUSES,
    Appointment,
    AppointmentStatus,
    EscalationReason,
    OfferRound,
)


class Node(Protocol):
    """A graph node: reads the state, returns a partial update.

    A Protocol rather than a `Callable` alias because every node takes `deps`
    keyword-only, which `Callable[...]` cannot express.
    """

    def __call__(
        self, state: ConversationState, *, deps: Dependencies
    ) -> Awaitable[ConversationState]: ...


#: ADR-23: "at most two retries". A retry is a *re*-draft, so the budget is one
#: initial draft plus two more — three compositions, then the static template
#: and a human. Named for retries rather than attempts because the first
#: version of this counted attempts against the same number and gave the
#: patient one fewer chance than the contract allows, which only showed up
#: when the graph was run end to end.
MAX_RETRIES: Final = 2

#: Total compositions allowed per turn, derived rather than written twice.
MAX_ATTEMPTS: Final = MAX_RETRIES + 1

#: The client's reschedule rule: five dates, then five different ones, then a
#: closing message. A third request in the same reschedule gets the message.
MAX_OFFER_ROUNDS: Final = 2

#: How long one reschedule lasts for counting rounds. A day, so a patient who
#: comes back next week to move the appointment starts from the first round
#: rather than being told nothing is available.
OFFER_EPISODE: Final = dt.timedelta(hours=24)


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

    # Through the repository, not a raw select: it records the access (rule 8)
    # and excludes soft-deleted patients (§7). Reading the row directly here
    # would disclose a patient's name to the conversation with nothing in the
    # audit log saying so, which is the exact failure §7 warns about when it
    # says repositories record accesses and routes never do.
    patient = await registry.get_patient(
        deps.session, clinic_id=state["clinic_id"], patient_id=patient_ids[0]
    )
    if patient is None:
        # A binding to a patient who is gone is a stranger as far as this
        # conversation goes.
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
    consent gets one templated reply, and that reply is the only thing this
    channel may carry for them.

    A patient who never gave consent is told how to give it. A patient who
    withdrew it -- usually by asking this channel to stop -- goes on to
    `classify` marked as withdrawn, rather than stopping here: rule 5 outranks
    the opt-out, so a clinical message from them must still reach a person, as
    must a request to speak to one. Everything else they send is answered
    with the opted-out notice and acted on not at all. Asking them to
    authorise the channel again would contact somebody who asked not to be.
    """
    patient_id = state["patient"].patient_id
    if patient_id is None:  # pragma: no cover - identify stopped already
        return {"stop": Stop.UNKNOWN_SENDER}

    revoked_at = (
        await deps.session.scalars(
            select(Consent.revoked_at).where(
                Consent.clinic_id == state["clinic_id"],
                Consent.patient_id == patient_id,
                Consent.purpose == ConsentPurpose.APPOINTMENT_MESSAGING,
            )
        )
    ).all()

    if any(at is None for at in revoked_at):
        return {}
    if revoked_at:
        return {"patient": replace(state["patient"], consent_withdrawn=True)}
    return {"stop": Stop.NO_CONSENT}


# ------------------------------------------------------------------- classify


#: The intents that end the turn here with a templated reply, and the stop
#: reason each becomes. Set here rather than in the routing function: a
#: conditional edge decides *where* to go, and a node decides *what to say*.
#: Opting out is not one of them: it has a write to make first, so it goes
#: to `act`, which sets `Stop.OPTED_OUT` once the consent is revoked.
_TERMINAL_INTENTS: Final[dict[Intent, Stop]] = {
    Intent.CLINICAL: Stop.CLINICAL_CONCERN,
    Intent.HUMAN: Stop.ASKED_FOR_HUMAN,
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
    elif state["patient"].consent_withdrawn and intent is not Intent.OPT_OUT:
        # Opted out earlier: nothing is confirmed, cancelled or offered for
        # them, and the reply is the notice that tells them how to come back.
        update["stop"] = Stop.OPTED_OUT
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

    intent = state["intent"]
    if intent is Intent.OPT_OUT:
        # Before anything is read: an opt-out needs no appointment and no
        # facts, and reading them would be a patient-data access with no use.
        await tools.withdraw_messaging_consent(
            deps.session, clinic_id=state["clinic_id"], patient_id=patient_id, now=state["now"]
        )
        return {"stop": Stop.OPTED_OUT}

    facts = await _clinic_facts(state, deps)
    appointment = await _active_appointment(state, deps, patient_id)
    if appointment is not None:
        facts.set(
            **await _doctor_facts(
                deps, clinic_id=state["clinic_id"], doctor_id=appointment.doctor_id
            )
        )

    offers = await _offers(state, deps, appointment) if appointment else []
    rounds = await _rounds(state, deps, appointment) if appointment else []
    if appointment is not None and (offers or rounds) and declines_offer(state["inbound_text"]):
        # An answer to an open offer that turns every date down: the next
        # round, never a cancellation of the appointment it would replace.
        # Checked before a number is read, so "el 3 no me sirve" is a refusal
        # rather than a pick of the third date.
        intent = Intent.RESCHEDULE
    live = [o for o in offers if o.expires_at is not None and o.expires_at > state["now"]]
    #: The dates of an offer the patient picked from too late, to offer again.
    again: tuple[dt.date, ...] = ()
    if (
        appointment is not None
        and (offers or rounds)
        and intent in {Intent.CONFIRM, Intent.UNCLEAR}
    ):
        # An earlier turn offered numbered times. A reply naming one of them
        # is a choice; "1" is a choice here, not "confirm", because the
        # patient is answering the offer.
        index = chosen_slot(state["inbound_text"], len(live)) if live else None
        if index is not None:
            moved = await tools.move_to_offered_slot(
                deps.session,
                clinic_id=state["clinic_id"],
                hold_id=live[index - 1].id,
                now=state["now"],
            )
            if moved.ok:
                return {
                    "intent": Intent.CHOOSE,
                    "appointment_id": live[index - 1].id,
                    "verified_facts": _merge(facts, moved.facts),
                }
        shown = len(offers) or (len(rounds[-1].offered_dates) if rounds else 0)
        if chosen_slot(state["inbound_text"], shown) is not None:
            # A pick of a time no longer held: expired, swept, or taken
            # between the read and the lock. ADR-17: offer again rather than
            # book a time that may have gone to somebody else. The same
            # dates, because a late answer is a choice, not a refusal: it
            # neither uses up a round nor rules out the date it named.
            intent = Intent.RESCHEDULE
            again = tuple(rounds[-1].offered_dates) if rounds else ()

    if appointment is not None and offers and intent in {Intent.CONFIRM, Intent.CANCEL}:
        # The patient answered about the appointment itself, so the times
        # held as alternatives go back now rather than at expiry.
        await booking.release_offers(
            deps.session, clinic_id=state["clinic_id"], appointment_id=appointment.id
        )

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
        # Every exit from here reports the intent it acted on: a decline such
        # as "ninguna" or "no" was read by the classifier as something else,
        # and the review screen should show what the turn did.
        acted: ConversationState = {"intent": Intent.RESCHEDULE}
        if appointment is None:
            return {**acted, "stop": Stop.NO_APPOINTMENT, "verified_facts": facts.build()}
        if not again and len(rounds) >= MAX_OFFER_ROUNDS:
            # Both rounds turned down: the closing message, and a case for
            # staff so a patient the channel could not serve is not lost.
            await booking.release_offers(
                deps.session, clinic_id=state["clinic_id"], appointment_id=appointment.id
            )
            return {
                **acted,
                "stop": Stop.OFFER_ROUNDS_EXHAUSTED,
                "appointment_id": appointment.id,
                "verified_facts": facts.build(),
            }
        # Never the appointment's own date: the patient is asking for another.
        starts = appointment.during.lower
        refused = set() if starts is None else {starts.astimezone(BOGOTA).date()}
        if not again:
            refused |= {day for r in rounds for day in r.offered_dates}
        result = await tools.find_slots(
            deps.session,
            clinic_id=state["clinic_id"],
            doctor_id=appointment.doctor_id,
            appointment_type_id=appointment.appointment_type_id,
            now=state["now"],
            exclude_dates=tuple(sorted(refused)),
            only_dates=again,
        )
        if not result.ok:
            return {**acted, "stop": Stop.NO_SLOTS, "verified_facts": facts.build()}
        # Each offered time is held (ADR-17), linked to the appointment it
        # would replace, so the patient's later "2" names a time nobody else
        # can take meanwhile. A fresh offer replaces any earlier one.
        await booking.release_offers(
            deps.session, clinic_id=state["clinic_id"], appointment_id=appointment.id
        )
        held = FactsBuilder()
        for slot in result.facts.build().offered_slots:
            end = slot.starts_at + dt.timedelta(minutes=slot.duration_minutes)
            taken = await tools.hold_slot(
                deps.session,
                clinic_id=state["clinic_id"],
                patient_id=patient_id,
                doctor_id=appointment.doctor_id,
                location_id=appointment.location_id,
                appointment_type_id=appointment.appointment_type_id,
                start=slot.starts_at,
                end=end,
                now=state["now"],
                idempotency_key=(
                    f"offer:{appointment.id}:{slot.starts_at.isoformat()}:"
                    f"{state['now'].isoformat()}"
                ),
                replaces_appointment_id=appointment.id,
            )
            if taken.ok:
                held.offer(slot.starts_at, slot.duration_minutes)
        offered = held.build().offered_slots
        if not offered:
            return {**acted, "stop": Stop.NO_SLOTS, "verified_facts": facts.build()}
        if not again:
            await tools.record_offer_round(
                deps.session,
                clinic_id=state["clinic_id"],
                appointment_id=appointment.id,
                patient_id=patient_id,
                offered_dates=tuple(slot.starts_at.astimezone(BOGOTA).date() for slot in offered),
                now=state["now"],
            )
        return {
            **acted,
            "appointment_id": appointment.id,
            "verified_facts": _merge(facts, held),
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


async def _offers(
    state: ConversationState, deps: Dependencies, appointment: Appointment
) -> list[Appointment]:
    """The times held as offers to replace `appointment`, live or expired but
    not yet swept, in the order they were offered (by start time, as
    `render_offer` numbered them). Each is patient data read, so each is
    recorded (rule 8)."""
    rows = list(
        (
            await deps.session.execute(
                select(Appointment)
                .where(
                    Appointment.clinic_id == state["clinic_id"],
                    Appointment.rescheduled_from_appointment_id == appointment.id,
                    Appointment.status == AppointmentStatus.HOLD,
                )
                .order_by(Appointment.during)
            )
        )
        .scalars()
        .all()
    )
    for row in rows:
        await record_access(
            deps.session,
            AccessAction.READ,
            Access(resource="appointments", resource_id=str(row.id), patient_id=row.patient_id),
        )
    return rows


async def _rounds(
    state: ConversationState, deps: Dependencies, appointment: Appointment
) -> list[OfferRound]:
    """The rounds of dates offered for this appointment in the current
    reschedule, oldest first. A reschedule lasts `OFFER_EPISODE`; a patient who
    asks again after that starts from the first round."""
    return await offers.recent_rounds(
        deps.session,
        clinic_id=state["clinic_id"],
        appointment_id=appointment.id,
        since=state["now"] - OFFER_EPISODE,
    )


async def _active_appointment(
    state: ConversationState, deps: Dependencies, patient_id: uuid.UUID
) -> Appointment | None:
    """The patient's next active appointment, resolved by code not by text.

    The ORM row rather than `scheduling.repository`'s projection, because the
    tools mutate it. That means the access is recorded here: the repository
    functions that would have done it return a read-only view, and an
    appointment is patient data whichever shape it arrives in (rule 8).
    """
    found = (
        await deps.session.execute(
            select(Appointment)
            .where(
                Appointment.clinic_id == state["clinic_id"],
                Appointment.patient_id == patient_id,
                Appointment.status.in_([s.value for s in ACTIVE_STATUSES]),
                # An offered slot is held for the patient but is not their
                # appointment: "sí" or "cancelar" after an offer act on the
                # appointment the offer would replace, never on the offer.
                or_(
                    Appointment.status != AppointmentStatus.HOLD,
                    Appointment.rescheduled_from_appointment_id.is_(None),
                ),
            )
            .order_by(Appointment.during)
            .limit(1)
        )
    ).scalar_one_or_none()
    if found is not None:
        await record_access(
            deps.session,
            AccessAction.READ,
            Access(
                resource="appointments",
                resource_id=str(found.id),
                patient_id=found.patient_id,
            ),
        )
    return found


async def _doctor_facts(
    deps: Dependencies, *, clinic_id: uuid.UUID, doctor_id: uuid.UUID
) -> dict[str, str | None]:
    """The doctor's name and specialty, for the templates that state them.

    Resolved here rather than returned by a tool: a tool reports what it *did*,
    and no tool changes a doctor. Without this the confirmation template cannot
    render — which is how it failed the first time the graph met a real
    database, exhausting the gate on a draft that was never going to pass.
    """
    doctor = (
        await deps.session.execute(
            select(Doctor).where(Doctor.clinic_id == clinic_id, Doctor.id == doctor_id)
        )
    ).scalar_one_or_none()
    if doctor is None:
        return {}
    return {"doctor_name": doctor.full_name, "specialty": doctor.specialty}


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


async def _record_disclosure(
    state: ConversationState,
    deps: Dependencies,
    *,
    processor: str,
    fields: tuple[str, ...],
) -> None:
    """Log that patient data left the machine for a third-party processor.

    One entry per provider call. `fields_disclosed` names what was sent rather
    than carrying it, because the audit log is not a copy of the disclosure --
    an auditor needs to know a name and an appointment time went to a model
    provider, not to read them again here.

    `patient_id` may be absent on a turn that never resolved a patient (a
    stranger's number), and the entry is still written: "we sent something to a
    provider about a sender we could not identify" is a fact worth having.
    """
    patient = state.get("patient")
    await record_access(
        deps.session,
        AccessAction.DISCLOSE,
        Access(
            resource="conversation_turns",
            resource_id=state.get("thread_id"),
            patient_id=patient.patient_id if patient else None,
            processor=processor,
            processor_model=_processor_model(deps.settings, processor),
            fields_disclosed=fields,
            zero_retention=False,
        ),
    )


def _processor_model(settings: Settings, processor: str) -> str:
    """Which model the disclosure went to, as configured."""
    return settings.evaluator_model if processor == "judge" else settings.generator_model


# -------------------------------------------------------------------- compose


async def compose(state: ConversationState, *, deps: Dependencies) -> ConversationState:
    """Produce a candidate message.

    In M3's prototype this renders a pre-approved template (ADR-01a), so the
    whole pipeline is exercised without a provider. The seam is
    `deps.generate`: M4 passes a model generator with the same signature, and
    the gate below does not change.
    """
    draft = deps.generate(state)

    # Rule 8: "including every disclosure to a model or speech provider."
    # Recorded here rather than inside `generate`, because the generator must
    # not hold a session (rule 2) and this node already does -- so the
    # disclosure is logged without handing the generator a database.
    #
    # Keyed on `draft.templated`, which is the only honest signal available:
    # a templated draft was rendered locally and reached no provider, while a
    # generated one did. A refusal is still a disclosure -- the provider saw
    # the facts before it failed -- and `generator._refusal` returns
    # `templated=False`, so those are recorded too.
    if not draft.templated:
        # Since ADR-02 was enforced the generator is sent the message type and
        # nothing else (`generator.build_user_message`), so that is what the
        # entry names. Still recorded: a call to a processor on this patient's
        # turn is an event an auditor asks about, even when it carried no data.
        await _record_disclosure(
            state,
            deps,
            processor="generator",
            fields=("message_kind",),
        )

    return {"drafts": (*state["drafts"], draft)}


# ----------------------------------------------------------------------- gate


async def gate(state: ConversationState, *, deps: Dependencies) -> ConversationState:
    """Stages 1-3, then stage 4 for generated text only (ADR-22).

    A templated draft skips the model judge: its wording was approved by a
    person, there is no generated text to judge, and sending it to a judge
    would introduce a model into a path that has none (ADR-01a).
    """
    draft = state["drafts"][-1]
    # Stage 0 (ADR-02): the words a model wrote may state no fact. Facts are
    # appended by code, so anything fact-shaped in this part is invented --
    # even a correct-looking one -- and the draft is refused before the
    # whole-message stages run.
    if draft.authored is not None and (authored := check_authored(draft.authored)):
        verdict = Verdict(approved=False, findings=tuple(authored))
    else:
        verdict = validate(draft.text, state["verified_facts"], draft.kind)

    if verdict.approved and not draft.templated:
        # ADR-01: the evaluator screens the free text. That is the model's
        # wording; the facts block is pre-approved template text. Judging only
        # the wording also means the judge is sent no patient data.
        await _record_disclosure(
            state,
            deps,
            processor="judge",
            fields=("authored_text",),
        )
        verdict = deps.judge(draft.authored if draft.authored is not None else draft.text)

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
    # No escalation: a staff case is a promise that someone will call, and the
    # patient has just asked not to be contacted.
    Stop.OPTED_OUT: (T.OPTED_OUT, None),
    # M2's no-acceptable-slot case, which M14's waitlist later builds on.
    Stop.OFFER_ROUNDS_EXHAUSTED: (T.NO_MORE_OFFERS, EscalationReason.NO_ACCEPTABLE_SLOT),
}

_ESCALATION_DETAIL: Final[dict[Stop, str]] = {
    Stop.NO_CONSENT: "El paciente escribió sin autorización vigente para este canal.",
    Stop.NO_SLOTS: "No hay horarios disponibles que el paciente pueda tomar.",
    Stop.CLINICAL_CONCERN: "El paciente hizo una consulta clínica por este canal.",
    Stop.ASKED_FOR_HUMAN: "El paciente pidió hablar con una persona.",
    Stop.GATE_EXHAUSTED: "El evaluador rechazó el mensaje en todos los intentos; no se envió nada generado.",
    Stop.UNKNOWN_SENDER: "Número no reconocido.",
    Stop.SHARED_HANDSET: "Número compartido por varios pacientes.",
    Stop.NO_APPOINTMENT: "No hay cita activa.",
    Stop.OPTED_OUT: "El paciente pidió no recibir más mensajes.",
    Stop.OFFER_ROUNDS_EXHAUSTED: (
        "El paciente rechazó las dos rondas de fechas ofrecidas para reprogramar su cita."
    ),
}


# -------------------------------------------------------------------- routing


def _after_identify(state: ConversationState) -> str:
    return "stop" if state.get("stop") else "consent"


def _after_consent(state: ConversationState) -> str:
    return "stop" if state.get("stop") else "classify"


def _after_classify(state: ConversationState) -> str:
    """`classify` sets a stop reason for every turn it ends -- a clinical
    message, a request for a person, anything from a patient who opted out
    other than opting out again -- and everything else goes on to `act`."""
    return "stop" if state.get("stop") else "act"


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
        {"act": "act", "stop": "stop"},
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


__all__ = ["MAX_ATTEMPTS", "MAX_RETRIES", "Dependencies", "build_graph"]
