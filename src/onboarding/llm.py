"""Ask a model which canonical field an ambiguous column holds.

This is a **fallback**, not the mapping engine. The dictionary, containment and
fuzzy stages in `matcher.py` resolve ordinary Spanish and English headings on
their own, so a typical import calls nothing here at all. A model is asked only
about a column those stages left ambiguous, and only when a key is configured;
with no key the column simply goes to the person confirming the import, which is
where it would have gone anyway.

**No patient value reaches a provider.** `ColumnQuestion` carries a column
*name*, our derived statistics, and examples we generated ourselves; it has no
field that can hold a cell, so a caller cannot pass one by mistake.

The type alone is not sufficient, and claiming it was hid a real leak: a
heading is only a heading once the file is known to HAVE headings. For a file
with no heading row the reader's "headers" are row 1 of the data, so a cédula,
a name and a phone number were sent as column names until `service._ask_model`
learned to wait for the structure question to be answered. Two sentinel tests
cover this now -- one on the payload shape, one that routes a real headerless
file through `analyse` and asserts nothing was asked -- and the second fails if
that wait is removed.

**The model chooses from a list; it never writes a transform.** The response
schema constrains `target_field` to an enum of canonical field names plus
`__NO_MATCH__`, so a hallucinated field is rejected by the schema rather than
detected afterwards. Which normalizer then converts the column is fixed in
`canonical.py`. This is the separation ADR-08a exists to enforce.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from pydantic import BaseModel, Field, ValidationError, field_validator

from src.core.config import Settings, get_settings
from src.core.logging import get_logger
from src.onboarding.canonical import FIELDS_BY_ENTITY, Entity

if TYPE_CHECKING:  # a runtime import would be circular: guidance imports this
    from src.onboarding.guidance import Explanation

log = get_logger(__name__)

#: Returned by the model when no canonical field fits. Explicit, so "none of
#: these" is a choice it can make rather than something it has to fake.
NO_MATCH: Final = "__NO_MATCH__"


class LLMUnavailable(Exception):
    """No provider could answer. The column goes to the human instead."""


@dataclass(frozen=True, slots=True)
class ColumnQuestion:
    """One ambiguous column, described without any of its contents.

    There is deliberately no field here for a cell value. The type is the
    guarantee: a caller cannot hand this object a patient's name, so no amount
    of carelessness downstream can transmit one.
    """

    #: The heading exactly as the file spells it. A heading is not patient data.
    header: str
    #: What the values look like, derived locally and never the values
    #: themselves: "10 digits", "date-like", "1 of 4 repeating values".
    shape: str
    #: How full the column is, as a percentage. A count, not content.
    filled_percent: int
    #: How many distinct values it holds. Again a count.
    distinct_count: int
    #: Examples **we generated** to show the shape, never sampled from the file.
    synthetic_examples: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Suggestion:
    column: str
    target_field: str | None
    confidence: float
    reason: str
    provider: str
    model: str


#: How much of the model's reason we keep. It is shown beside the column on
#: the confirmation screen, where a paragraph would push the next column off
#: the page, so it is bounded. The bound is also sent to the provider as
#: `maxLength`, because a limit the model is not told about is one it breaks.
REASON_LIMIT: Final = 200


class _Proposal(BaseModel):
    """The shape the model must answer in."""

    target_field: str = Field(description="A canonical field name, or __NO_MATCH__.")
    confidence: float = Field(ge=0.0, le=1.0)
    # Truncated rather than rejected. A model that writes 220 characters has
    # still chosen a field, and discarding a sound mapping over the length of
    # its prose sends the column to a human for no reason -- which is what
    # happened: every call failed validation and the feature did nothing.
    reason: str = Field(max_length=REASON_LIMIT)

    @field_validator("reason", mode="before")
    @classmethod
    def _shorten(cls, value: object) -> object:
        if isinstance(value, str) and len(value) > REASON_LIMIT:
            return value[: REASON_LIMIT - 1].rstrip() + "\u2026"
        return value


#: Model families that reject an explicit `temperature`. OpenAI's reasoning
#: models accept only the default of 1: sending 0 returns a 400 rather than
#: being clamped, so every call fails and the column falls back to the human.
#:
#: Matched on a prefix of the model name because the family is what decides
#: this, not the exact snapshot. A name we do not recognise keeps `temperature=0`,
#: which is the safer default: determinism is what makes an audited path
#: reproducible, and a provider that rejects it tells us so in one 400 rather
#: than drifting silently.
_NO_TEMPERATURE: Final = ("gpt-5", "o1", "o3", "o4")


def _sampling(model: str) -> dict[str, float]:
    """`{"temperature": 0}`, or nothing when the model refuses it."""
    name = model.rsplit("/", 1)[-1]
    return {} if name.startswith(_NO_TEMPERATURE) else {"temperature": 0}


@dataclass(slots=True)
class _Provider:
    name: str
    model: str
    api_key: str
    base_url: str | None = None


def _providers(settings: Settings) -> list[_Provider]:
    """Providers to try, in order.

    OpenAI first because that is the key the client pays for. Groq second: its
    quota is independent, so it is a real failover for a rate limit rather than
    a second attempt at the same exhausted budget.
    """
    found: list[_Provider] = []
    if key := settings.openai_api_key.get_secret_value():
        found.append(_Provider("openai", settings.mapping_model_openai, key))
    if key := settings.groq_api_key.get_secret_value():
        found.append(_Provider("groq", settings.mapping_model_groq, key, settings.groq_base_url))
    return found


def _candidate_fields(entity: Entity, candidates: tuple[str, ...]) -> list[dict[str, str]]:
    """The fields the model may choose between, described for a reader."""
    allowed = set(candidates)
    return [
        {
            "name": f.name,
            "means": f.description,
            # Examples from the canonical schema, written by us. The file's own
            # values are never used for this.
            "looks_like": ", ".join(f.examples) if f.examples else "",
        }
        for f in FIELDS_BY_ENTITY[entity]
        if not allowed or f.name in allowed
    ]


def build_messages(
    question: ColumnQuestion, entity: Entity, candidates: tuple[str, ...] = ()
) -> list[dict[str, str]]:
    """The exact payload sent to a provider. Nothing else is transmitted.

    The heading is repeated in the user message: with no real values to go on,
    emphasising the header is the one thing measured to help a zero-shot model
    (Magneto, VLDB 2025), and our privacy rule puts us firmly in that regime.
    """
    fields = _candidate_fields(entity, candidates)
    return [
        {
            "role": "system",
            "content": (
                "You map a column heading from a Colombian medical clinic's spreadsheet "
                "onto one canonical field. Headings may be Spanish or English, may lack "
                "accents, and may be abbreviated. Choose the single best field, or "
                f"{NO_MATCH} if none fits. Never invent a field name. You are given no "
                "patient data and must not ask for any."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "column_heading": question.header,
                    "heading_again": question.header,
                    "value_shape": question.shape,
                    "filled_percent": question.filled_percent,
                    "distinct_values": question.distinct_count,
                    "example_of_that_shape": list(question.synthetic_examples),
                    "candidate_fields": fields,
                },
                ensure_ascii=False,
            ),
        },
    ]


def _schema(entity: Entity, candidates: tuple[str, ...]) -> dict[str, Any]:
    """A JSON schema whose enum makes a hallucinated field impossible."""
    allowed = [f["name"] for f in _candidate_fields(entity, candidates)] + [NO_MATCH]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "column_mapping",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "target_field": {"type": "string", "enum": allowed},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string", "maxLength": REASON_LIMIT},
                },
                "required": ["target_field", "confidence", "reason"],
                "additionalProperties": False,
            },
        },
    }


def suggest(
    question: ColumnQuestion,
    entity: Entity,
    *,
    candidates: tuple[str, ...] = (),
    settings: Settings | None = None,
    client_factory: Any = None,
) -> Suggestion:
    """Ask the configured providers about one column, or raise.

    Raising is a normal outcome, not an error: the caller shows the column to a
    person instead, which is what would have happened without a provider at all.
    """
    settings = settings or get_settings()
    providers = _providers(settings)
    if not providers:
        raise LLMUnavailable("No model provider is configured.")

    messages = build_messages(question, entity, candidates)
    # A digest of what was sent, so a reviewer can confirm afterwards exactly
    # which payload left the machine without the payload itself being stored.
    digest = hashlib.sha256(
        json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    failures: list[str] = []
    for index, provider in enumerate(providers):
        started = time.monotonic()
        try:
            client = (client_factory or _openai_client)(provider, settings)
            response = client.chat.completions.create(
                model=provider.model,
                messages=messages,
                response_format=_schema(entity, candidates),
                **_sampling(provider.model),
            )
            content = response.choices[0].message.content or ""
            proposal = _Proposal.model_validate_json(content)
        except ValidationError as error:
            # Under a strict schema this should be impossible, so it means the
            # model has stopped honouring it. Failing over is right; retrying
            # the same provider would only repeat the same violation.
            failures.append(f"{provider.name}: response did not match the schema")
            _log_call(provider, question, digest, started, "schema_violation", str(error))
            continue
        except Exception as error:  # provider errors are not ours to classify
            failures.append(f"{provider.name}: {type(error).__name__}")
            _log_call(provider, question, digest, started, "error", str(error)[:200])
            continue

        _log_call(
            provider,
            question,
            digest,
            started,
            "ok",
            "",
            usage=getattr(response, "usage", None),
            attempt=index,
        )
        target = None if proposal.target_field == NO_MATCH else proposal.target_field
        return Suggestion(
            column=question.header,
            target_field=target,
            confidence=proposal.confidence,
            reason=proposal.reason,
            provider=provider.name,
            model=provider.model,
        )

    raise LLMUnavailable("; ".join(failures))


#: The wording call's response shape. Two sentences, nothing structural: the
#: model is supplying prose for a screen, not a decision.
_WORDING_SCHEMA: Final = {
    "type": "json_schema",
    "json_schema": {
        "name": "rule_wording",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"means": {"type": "string"}, "action": {"type": "string"}},
            "required": ["means", "action"],
            "additionalProperties": False,
        },
    },
}


def build_wording_messages(rule: str, field_means: str) -> list[dict[str, str]]:
    """The exact payload for a wording call. Nothing else is transmitted.

    Both inputs are ours: `rule` is a name from our own fixed vocabulary, and
    `field_means` is the description written in `canonical.py`. Neither is the
    clinic's heading and neither can be a cell.
    """
    return [
        {
            "role": "system",
            "content": (
                "A spreadsheet importer for Colombian medical clinics held a cell back "
                "for review. Explain the named rule to the receptionist who has to act "
                "on it, in Colombian Spanish, in two short sentences: what the rule "
                "means, and what they should do. Never tell them to correct an identity "
                "document or a phone number to a nearby valid value -- the corrected "
                "value would belong to a different person. You are given no patient "
                "data and must not ask for any."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"rule_name": rule, "field_means": field_means}, ensure_ascii=False
            ),
        },
    ]


def explain_rule(rule: str, field_means: str, *, settings: Settings | None = None) -> Explanation:
    """Word one refusal rule the guidance table has no entry for.

    Returns a `guidance.Explanation`. Imported inside the function because
    `guidance` imports this module for exactly this call, and the dependency
    only exists at call time.

    This decides nothing. The cell's status and the row's fate are settled
    before it runs; a failure raises `LLMUnavailable` and the caller falls back
    to the normalizer's own message.
    """
    from src.onboarding.guidance import Explanation

    settings = settings or get_settings()
    providers = _providers(settings)
    if not providers:
        raise LLMUnavailable("No model provider is configured.")

    messages = build_wording_messages(rule, field_means)
    digest = hashlib.sha256(
        json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    failures: list[str] = []
    for index, provider in enumerate(providers):
        started = time.monotonic()
        try:
            client = _openai_client(provider, settings)
            response = client.chat.completions.create(
                model=provider.model,
                messages=messages,
                response_format=_WORDING_SCHEMA,
                **_sampling(provider.model),
            )
            wording = json.loads(response.choices[0].message.content or "{}")
            means, action = str(wording["means"]), str(wording["action"])
        except Exception as error:  # provider errors are not ours to classify
            failures.append(f"{provider.name}: {type(error).__name__}")
            _log_call(
                provider,
                rule,
                digest,
                started,
                "error",
                str(error)[:200],
                purpose="rule_wording",
            )
            continue

        _log_call(
            provider,
            rule,
            digest,
            started,
            "ok",
            "",
            usage=getattr(response, "usage", None),
            attempt=index,
            purpose="rule_wording",
        )
        return Explanation(rule=rule, means=means, action=action, source=provider.name)

    raise LLMUnavailable("; ".join(failures))


def _openai_client(provider: _Provider, settings: Settings) -> Any:
    """One SDK for both providers: Groq speaks the OpenAI protocol."""
    from openai import OpenAI

    return OpenAI(
        api_key=provider.api_key,
        base_url=provider.base_url,
        timeout=settings.mapping_timeout_seconds,
        max_retries=2,
    )


def _log_call(
    provider: _Provider,
    question: ColumnQuestion | str,
    payload_sha256: str,
    started: float,
    outcome: str,
    detail: str,
    *,
    usage: Any = None,
    attempt: int = 0,
    purpose: str = "column_mapping",
) -> None:
    """Record the call for the audit trail.

    The prompt is absent rather than redacted: `payload_sha256` lets a reviewer
    recompute the digest from the column names in the import and confirm what
    was sent, without the payload ever being stored.
    """
    log.info(
        "llm.call",
        provider=provider.name,
        model=provider.model,
        purpose=purpose,
        # A rule name for the wording call, a heading for the mapping call.
        # Both are ours or the clinic's label, never a cell.
        column=question if isinstance(question, str) else question.header,
        attempt=attempt,
        latency_ms=round((time.monotonic() - started) * 1000),
        outcome=outcome,
        detail=detail or None,
        payload_sha256=payload_sha256,
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
    )


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    """Local statistics about a column. Derived here, never transmitted raw."""

    shape: str
    filled_percent: int
    distinct_count: int
    examples: tuple[str, ...] = field(default=())


def profile_column(values: list[str]) -> ColumnProfile:
    """Describe a column's values without keeping any of them.

    Everything returned is a count or a category. The `examples` are patterns
    built from the shape, so what leaves the machine describes the column
    without ever being a patient's.
    """
    filled = [v.strip() for v in values if v.strip()]
    if not filled:
        return ColumnProfile("empty", 0, 0)

    percent = round(100 * len(filled) / len(values))
    distinct = len(set(filled))

    digits = sum(1 for v in filled if v.isdigit())
    with_at = sum(1 for v in filled if "@" in v)
    dated = sum(1 for v in filled if _looks_like_date(v))
    timed = sum(1 for v in filled if ":" in v and len(v) <= 12)

    share = len(filled)
    if digits == share:
        lengths = {len(v) for v in filled}
        length = f"{lengths.pop()} digits" if len(lengths) == 1 else "digits of varying length"
        return ColumnProfile(length, percent, distinct, (_synthetic_digits(filled[0]),))
    if with_at > share * 0.8:
        return ColumnProfile("email addresses", percent, distinct, ("nombre@ejemplo.com",))
    if dated > share * 0.8:
        return ColumnProfile("dates", percent, distinct, ("2026-10-15",))
    if timed > share * 0.8:
        return ColumnProfile("times of day", percent, distinct, ("07:30",))
    if distinct <= 8 and share > 8:
        # A small set of repeating labels: a status or a category.
        return ColumnProfile(
            f"{distinct} repeating labels", percent, distinct, ("etiqueta-a", "etiqueta-b")
        )
    return ColumnProfile("free text", percent, distinct, ("texto",))


def _looks_like_date(value: str) -> bool:
    return bool(value) and sum(c in "/-" for c in value) == 2 and any(c.isdigit() for c in value)


def _synthetic_digits(sample: str) -> str:
    """A number of the same length, invented rather than taken from the file."""
    return "1234567890123456789012"[: len(sample)] or "1234567890"
