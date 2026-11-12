"""The tool registry cannot be driven by a sentence (rules 2 and 3).

These are unit tests over the registry's *shape*, which is where rule 3 is
actually enforced. The tools' behaviour against a real database is covered by
`tests/integration/test_conversation_graph.py`; what is asserted here is the
property no integration test can show: that there is nowhere for patient text
to go.

The structural check is deliberately written against the live signatures via
`inspect` rather than as a list of expected names. A list would pass forever
while someone added a `notes: str` parameter beside it.
"""

from __future__ import annotations

import datetime as dt
import inspect

import pytest

from src.conversation import tools
from src.conversation.facts import FactsBuilder

#: Argument names allowed to be `str`, each with a reason. Anything else that
#: is a string is a hole for patient text, so this list is the whole of the
#: exemption and every entry needs justifying.
ALLOWED_STRINGS = {
    # A code the graph chose from a closed set, never the patient's words.
    "reason_code",
    # A key the graph derives from thread and attempt, so a resumed node does
    # not double-book. Not patient-supplied.
    "idempotency_key",
}


def _tool_functions() -> list[tuple[str, object]]:
    return sorted(tools.REGISTRY.items())


@pytest.mark.parametrize(
    ("name", "func"), _tool_functions(), ids=lambda v: getattr(v, "__name__", v)
)
def test_no_tool_accepts_free_text(name: str, func: object) -> None:
    """Rule 3, enforced by shape.

    Every parameter is a UUID, a datetime, a date, a time, an int or the
    session. The two string exemptions are named above and are both
    graph-chosen codes. A tool with a free-text argument is how "ignora lo
    anterior" reaches a decision.
    """
    signature = inspect.signature(func)  # type: ignore[arg-type]
    for param in signature.parameters.values():
        if param.name in {"session", "self"}:
            continue
        annotation = str(param.annotation)
        if "str" not in annotation:
            continue
        assert param.name in ALLOWED_STRINGS, (
            f"{name}.{param.name} takes a string that is not an approved code: "
            "patient text must never reach a tool argument"
        )


@pytest.mark.parametrize(
    ("name", "func"), _tool_functions(), ids=lambda v: getattr(v, "__name__", v)
)
def test_every_tool_is_keyword_only_except_the_session(name: str, func: object) -> None:
    """Positional arguments are how a clinic_id and a patient_id get swapped.

    Both are UUIDs, so nothing would complain, and the result would be one
    clinic's scope applied to another's record.
    """
    signature = inspect.signature(func)  # type: ignore[arg-type]
    positional = [
        p.name
        for p in signature.parameters.values()
        if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and p.name != "session"
    ]
    assert positional == [], f"{name} has positional arguments: {positional}"


@pytest.mark.parametrize(
    ("name", "func"), _tool_functions(), ids=lambda v: getattr(v, "__name__", v)
)
def test_every_tool_requires_a_clinic(name: str, func: object) -> None:
    """Row-level security needs a clinic to compare against, and a tool that
    did not take one would read nothing while looking like an empty database
    (§7)."""
    signature = inspect.signature(func)  # type: ignore[arg-type]
    assert "clinic_id" in signature.parameters, f"{name} does not take a clinic_id"


@pytest.mark.parametrize(
    ("name", "func"), _tool_functions(), ids=lambda v: getattr(v, "__name__", v)
)
def test_every_tool_returns_a_tool_result(name: str, func: object) -> None:
    """A tool returning prose would be unchecked text one step from a patient."""
    signature = inspect.signature(func)  # type: ignore[arg-type]
    assert signature.return_annotation in {tools.ToolResult, "ToolResult"}, (
        f"{name} does not return ToolResult"
    )


def test_the_registry_is_the_whole_reach() -> None:
    """Exactly the four tools ADR-02 names, so adding a fifth is a visible
    change rather than an import somewhere."""
    assert set(tools.REGISTRY) == {
        "confirm_appointment",
        "cancel_appointment",
        "find_slots",
        "hold_slot",
    }


def test_hold_requires_an_idempotency_key() -> None:
    """§9: node resumption re-executes a node from its start. Without a
    required key, a resumed graph holds the same slot twice."""
    signature = inspect.signature(tools.hold_slot)
    key = signature.parameters["idempotency_key"]
    assert key.default is inspect.Parameter.empty, "the key must not be optional"


def test_an_offer_is_bounded() -> None:
    """A message listing ten times is one nobody reads, and a template allows
    ten buttons at most anyway (§9a)."""
    assert 1 <= tools.MAX_OFFERED_SLOTS <= 10


def test_tool_results_carry_facts_not_sentences() -> None:
    """The type a tool hands back builds `verified_facts` and nothing else."""
    result = tools.ToolResult(ok=True, facts=FactsBuilder())
    assert isinstance(result.facts, FactsBuilder)
    built = result.facts.build()
    assert built.is_empty()


def test_minutes_between_two_instants() -> None:
    start = dt.datetime(2026, 10, 23, 10, 0, tzinfo=dt.UTC)
    assert tools._minutes(start, start + dt.timedelta(minutes=30)) == 30
    assert tools._minutes(None, start) == 0


def test_a_uuid_is_what_identifies_an_appointment() -> None:
    """Sanity check on the premise: the graph resolves records, not the model.

    If this ever becomes a name or a phone number, rule 3 has been lost.
    """
    signature = inspect.signature(tools.cancel_appointment)
    assert "uuid" in str(signature.parameters["appointment_id"].annotation).lower()


def test_the_retry_budget_matches_adr_23() -> None:
    """ADR-23 says "at most two retries", which is three compositions.

    The first version counted attempts against the retry number and stopped
    after two drafts, giving the patient one fewer chance than the contract
    allows. It only showed up running the graph end to end, so the arithmetic
    is pinned here where it is cheap to check.
    """
    from src.conversation.graph import MAX_ATTEMPTS, MAX_RETRIES

    assert MAX_RETRIES == 2
    assert MAX_ATTEMPTS == MAX_RETRIES + 1 == 3
