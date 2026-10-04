"""The refusal guidance: written in Spanish, keyed on the rule, never the cell.

The guarantee worth testing is not that any particular sentence exists. It is
that **no patient value can reach a provider through this path**, and that the
table keeps up with the normalizers: a rule added without an entry leaves a
receptionist facing an English refusal and no instruction.
"""

from __future__ import annotations

import inspect
import json
import re

import pytest

from src.core.config import Settings
from src.onboarding import guidance, llm, normalizers


def _settings_with_a_key() -> Settings:
    """Settings that report a provider as configured, without one existing.

    `mapping_llm_available` is what `explain` consults; the call itself is
    stubbed in every test here, so no request is ever made.
    """
    return Settings(
        openai_api_key="sk-test-not-a-real-key",
        groq_api_key="",
        database_url="postgresql://u:p@localhost/x",
    )


def _emitted_rules() -> set[str]:
    """Every rule name the normalizers can stamp on a *refusal*.

    Read from the source rather than listed by hand, which is what makes the
    coverage test fail when a rule is added. A rule is a refusal when it is the
    first argument to `_review` or `_invalid`; `_valid` rules mark a successful
    conversion and never reach a review queue.
    """
    source = inspect.getsource(normalizers)
    found = re.findall(r"_(?:review|invalid)\(\s*\n?\s*[\"']([a-z_]+\.[a-z_]+)[\"']", source)
    assert found, "no rules were found; the regex no longer matches the source"
    return set(found)


def test_every_refusal_a_normalizer_can_emit_is_explained() -> None:
    """A rule with no entry is a receptionist with no instruction.

    `date.implausible` is included deliberately: it is built in `service._apply`
    rather than in a normalizer, so it is named here as well.
    """
    rules = _emitted_rules() | {"date.implausible", "date.ambiguous_column"}
    assert guidance.missing_rules(rules) == set()


def test_the_table_explains_nothing_that_cannot_happen() -> None:
    """A stale entry is dead code, and dead code is what the client reads for."""
    rules = _emitted_rules() | {"date.implausible", "date.ambiguous_column"}
    assert set(guidance.GUIDANCE) - rules == set()


def test_guidance_never_tells_anyone_to_repair_an_identifier() -> None:
    """The nearest valid cédula or phone number belongs to a stranger.

    `normalizers.phone` refuses rather than corrects for this reason; guidance
    that then said "fix it" would undo that in the one place a person acts.
    """
    for rule in ("phone.not_assigned", "document_number.scientific_notation"):
        explanation = guidance.explain(rule)
        assert explanation is not None
        text = f"{explanation.means} {explanation.action}".lower()
        assert "corrija" not in text and "corríjalo" not in text, (rule, text)


def test_an_unknown_rule_is_not_invented() -> None:
    """None is the honest answer; the caller still has the refusal itself."""
    assert guidance.explain("phone.something_new") is None


# ------------------------------------------------- what leaves the machine
def test_the_wording_payload_carries_no_cell_and_no_clinic_heading() -> None:
    """The whole point of keying on the rule rather than the message.

    Fourteen refusal messages embed the offending value -- that is what makes
    them actionable on screen -- so sending a message to a provider would send
    cédulas, phone numbers and names. The payload is built from the rule name
    and our own field description, and this asserts the built payload against a
    value that must not appear in it.
    """
    payload = json.dumps(
        llm.build_wording_messages(
            "phone.not_assigned", "The mobile number reminders are sent to."
        ),
        ensure_ascii=False,
    )
    for forbidden in ("3067891234", "Carlos", "1045678901", "CELULAR"):
        assert forbidden not in payload, forbidden
    assert "phone.not_assigned" in payload


def test_the_wording_call_needs_a_provider_and_says_so_when_there_is_none() -> None:
    """With no key the screen shows the normalizer's own message, as before."""
    from src.core.config import Settings

    settings = Settings(
        openai_api_key="", groq_api_key="", database_url="postgresql://u:p@localhost/x"
    )
    with pytest.raises(llm.LLMUnavailable):
        llm.explain_rule("phone.whatever", "a phone", settings=settings)


# ------------------------------------------- generated wording, table as floor
def test_generated_wording_is_preferred_when_a_provider_is_configured(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The client chose generated wording over the written table, cost accepted.

    What reaches the provider is unchanged -- a rule name and our own field
    description -- so this changes the words a receptionist reads and the bill,
    never what leaves the machine.
    """
    guidance._generated.cache_clear()
    monkeypatch.setattr(guidance, "get_settings", lambda: _settings_with_a_key(), raising=True)
    monkeypatch.setattr(
        llm,
        "explain_rule",
        lambda rule, means: guidance.Explanation(rule, "generado", "haga esto", "openai"),
    )
    explanation = guidance.explain("phone.not_assigned", "phone_e164")
    assert explanation is not None
    assert explanation.source == "openai"
    assert explanation.means == "generado"


def test_a_provider_failure_falls_back_to_the_written_table(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A model that is down must not leave a reviewer with no instruction."""
    guidance._generated.cache_clear()
    monkeypatch.setattr(guidance, "get_settings", lambda: _settings_with_a_key(), raising=True)

    def _down(rule: str, means: str) -> guidance.Explanation:
        raise llm.LLMUnavailable("both providers refused")

    monkeypatch.setattr(llm, "explain_rule", _down)
    explanation = guidance.explain("phone.not_assigned", "phone_e164")
    assert explanation is not None
    assert explanation.source == "table"
    assert "no lo aproxime" in explanation.action.lower()


def test_one_call_per_rule_however_many_cells_share_it(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A transform log holds thousands of cells and a handful of causes.

    Without the cache a 2,000-row file failing three ways would make 2,000
    calls for three answers, which is the one cost worth refusing.
    """
    guidance._generated.cache_clear()
    monkeypatch.setattr(guidance, "get_settings", lambda: _settings_with_a_key(), raising=True)
    calls: list[str] = []

    def _count(rule: str, means: str) -> guidance.Explanation:
        calls.append(rule)
        return guidance.Explanation(rule, "m", "a", "openai")

    monkeypatch.setattr(llm, "explain_rule", _count)
    for _ in range(50):
        guidance.explain("phone.not_assigned", "phone_e164")
    assert calls == ["phone.not_assigned"]


def test_with_no_key_the_table_answers_and_nothing_is_called(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """An import with no provider configured behaves exactly as it did before."""
    guidance._generated.cache_clear()

    def _never(rule: str, means: str) -> guidance.Explanation:
        raise AssertionError("no provider is configured; nothing should be asked")

    monkeypatch.setattr(llm, "explain_rule", _never)
    explanation = guidance.explain("phone.not_assigned", "phone_e164")
    assert explanation is not None
    assert explanation.source == "table"
