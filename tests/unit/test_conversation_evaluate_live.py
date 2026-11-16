"""The live measurement never scores a judge that did not answer.

No provider is reached: `judge` is replaced, so these run in the ordinary
suite (§6). They guard the measurement itself, because a rate-limited run
that counted every non-answer as a catch would report a perfect rate.
"""

from __future__ import annotations

import pytest

from src.conversation import evaluate_live
from src.conversation.judge import _unavailable
from src.conversation.validator import Verdict
from src.core.config import Settings

KEYS = Settings(_env_file=None, openai_api_key="x" * 20, groq_api_key="y" * 20)


def test_a_judge_that_does_not_answer_stops_the_measurement(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        evaluate_live, "judge", lambda draft, settings: _unavailable("RateLimitError")
    )

    assert evaluate_live.run(KEYS, repeat=1) == 3
    out = capsys.readouterr()
    assert "STOPPED" in out.err
    assert "catch rate" not in out.out


def test_a_judge_that_answers_is_measured(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A perfect judge, so the rates printed are known in advance."""
    bad = {case.draft for case in evaluate_live.BAD}
    monkeypatch.setattr(
        evaluate_live, "judge", lambda draft, settings: Verdict(approved=draft not in bad)
    )

    assert evaluate_live.run(KEYS, repeat=2) == 0
    out = capsys.readouterr().out
    assert "catch rate         : mean 100%, worst run 100%, over 2 runs" in out
    assert "correct-pass rate  : mean 100%, worst run 100%, over 2 runs" in out
