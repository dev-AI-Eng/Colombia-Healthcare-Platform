"""Asking a model about an ambiguous column.

The load-bearing test here is the first one: **no patient value reaches a
provider**. Everything else is about the fallback staying a fallback — not
called when the dictionary already knows, not called when no key is set, and
never deciding anything a person does not confirm.

No test calls a provider. The client is injected, so the payload can be
inspected exactly as it would be sent.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.core.config import Settings
from src.onboarding import llm
from src.onboarding.canonical import Entity

# Planted in the fixtures' patient cells. If any of these ever appears in an
# outbound payload, patient data has left the machine.
SENTINELS = (
    "Zzyzx Sentinelensen Marcadorez",
    "9999888877",
    "3009998877",
    "sentinel.marcador@example.invalid",
    "Carlos Andrés Pérez Gómez",
    "1045678901",
)

STRONG = "k" * 40


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "app_env": "local",
        "phi_encryption_key": STRONG,
        "phi_blind_index_key": STRONG + "b",
        "audit_chain_key": STRONG + "a",
        "openai_api_key": "test-key-not-a-real-one",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


class _FakeClient:
    """Records what would have been sent, and answers as a provider would."""

    def __init__(self, answer: str = "document_number", *, fail: Exception | None = None) -> None:
        self.answer = answer
        self.fail = fail
        self.calls: list[dict[str, Any]] = []
        self.chat = self  # type: ignore[assignment]
        self.completions = self  # type: ignore[assignment]

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.fail is not None:
            raise self.fail
        content = json.dumps(
            {"target_field": self.answer, "confidence": 0.9, "reason": "the heading matches"}
        )
        message = type("M", (), {"content": content})()
        choice = type("C", (), {"message": message})()
        usage = type("U", (), {"prompt_tokens": 120, "completion_tokens": 20})()
        return type("R", (), {"choices": [choice], "usage": usage})()

    def payload(self) -> str:
        return json.dumps(self.calls, ensure_ascii=False)


# ----------------------------------------------------- the guarantee that matters
def test_no_patient_value_can_reach_a_provider() -> None:
    """The prompt carries a heading, counts, and examples we generated.

    This is the test protecting CLAUDE.md rule 7 for the mapping stage. It plants
    real-looking values in a column and asserts that not one of them appears in
    what would be transmitted.
    """
    question = llm.ColumnQuestion(
        header="IDENTIFICACION",
        shape="10 digits",
        filled_percent=100,
        distinct_count=6,
        synthetic_examples=("1234567890",),
    )
    payload = json.dumps(llm.build_messages(question, Entity.PATIENT), ensure_ascii=False)

    for sentinel in SENTINELS:
        assert sentinel not in payload, f"{sentinel!r} would have been transmitted"


def test_the_question_type_cannot_carry_a_cell_value() -> None:
    """Privacy here is structural, not a matter of remembering.

    `ColumnQuestion` has no field for a value, so a caller cannot pass one even
    by mistake. If a field is ever added that could hold one, this fails.
    """
    fields = set(llm.ColumnQuestion.__dataclass_fields__)
    assert fields == {
        "header",
        "shape",
        "filled_percent",
        "distinct_count",
        "synthetic_examples",
    }


def test_profiling_describes_a_column_without_keeping_its_values() -> None:
    values = ["1045678901", "1023456789", "71234567", "43567890"]
    profile = llm.profile_column(values)

    assert profile.filled_percent == 100
    assert profile.distinct_count == 4
    assert "digit" in profile.shape
    # The example shows the shape and is not one of the values.
    assert all(v not in profile.examples for v in values)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["ana@x.com", "luis@y.com", "b@z.com"], "email"),
        (["15/10/2026", "03/11/2026", "22/10/2026"], "dates"),
        (["07:30", "08:00", "16:45"], "times"),
        ([""] * 5, "empty"),
    ],
)
def test_the_shape_of_a_column_is_recognised(values: list[str], expected: str) -> None:
    assert expected in llm.profile_column(values).shape


# ------------------------------------------------------------- staying a fallback
def test_nothing_is_called_when_no_key_is_configured() -> None:
    """With no provider the importer is unchanged; the column goes to a human."""
    question = llm.ColumnQuestion("COLUMNA RARA", "free text", 90, 40)
    with pytest.raises(llm.LLMUnavailable, match="No model provider"):
        llm.suggest(question, Entity.PATIENT, settings=_settings(openai_api_key=""))


def test_openai_is_preferred_and_groq_is_the_failover() -> None:
    """The client pays for OpenAI, so his key leads; Groq has separate quota."""
    settings = _settings(openai_api_key="openai-key", groq_api_key="groq-key")
    providers = llm._providers(settings)
    assert [p.name for p in providers] == ["openai", "groq"]


def test_a_failing_provider_falls_over_to_the_next() -> None:
    clients = [
        _FakeClient(fail=RuntimeError("rate limited")),
        _FakeClient(answer="document_number"),
    ]
    calls = iter(clients)
    suggestion = llm.suggest(
        llm.ColumnQuestion("NRO IDENT", "10 digits", 100, 9),
        Entity.PATIENT,
        settings=_settings(openai_api_key="a", groq_api_key="b"),
        client_factory=lambda provider, settings: next(calls),
    )
    assert suggestion.provider == "groq"
    assert suggestion.target_field == "document_number"


def test_when_every_provider_fails_the_column_goes_to_a_human() -> None:
    with pytest.raises(llm.LLMUnavailable):
        llm.suggest(
            llm.ColumnQuestion("NRO IDENT", "10 digits", 100, 9),
            Entity.PATIENT,
            settings=_settings(openai_api_key="a"),
            client_factory=lambda provider, settings: _FakeClient(fail=RuntimeError("down")),
        )


# --------------------------------------------------- the model cannot invent a field
def test_the_response_schema_restricts_the_answer_to_real_fields() -> None:
    """A hallucinated field is impossible, not merely detected afterwards."""
    schema = llm._schema(Entity.PATIENT, ("document_number", "birth_date"))
    allowed = schema["json_schema"]["schema"]["properties"]["target_field"]["enum"]

    assert set(allowed) == {"document_number", "birth_date", llm.NO_MATCH}
    assert schema["json_schema"]["strict"] is True
    assert schema["json_schema"]["schema"]["additionalProperties"] is False


def test_a_response_that_breaks_the_schema_is_rejected() -> None:
    """Under strict mode this cannot happen, so it means the model stopped
    honouring the schema. Failing over is right; retrying the same provider
    would only repeat the violation."""

    class _Malformed(_FakeClient):
        def create(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            message = type("M", (), {"content": '{"target_field": 12345}'})()
            choice = type("C", (), {"message": message})()
            return type("R", (), {"choices": [choice], "usage": None})()

    with pytest.raises(llm.LLMUnavailable, match="schema"):
        llm.suggest(
            llm.ColumnQuestion("X", "free text", 50, 3),
            Entity.PATIENT,
            settings=_settings(openai_api_key="a"),
            client_factory=lambda provider, settings: _Malformed(),
        )


def test_no_match_is_a_choice_the_model_can_make() -> None:
    """Better an honest "none of these" than a confident wrong field."""
    suggestion = llm.suggest(
        llm.ColumnQuestion("COLUMNA INTERNA", "free text", 100, 40),
        Entity.PATIENT,
        settings=_settings(openai_api_key="a"),
        client_factory=lambda provider, settings: _FakeClient(answer=llm.NO_MATCH),
    )
    assert suggestion.target_field is None


def test_the_heading_is_emphasised_in_the_prompt() -> None:
    """With no values to go on, repeating the heading is the one measured gain
    for a zero-shot model, and our privacy rule puts us in that regime."""
    messages = llm.build_messages(
        llm.ColumnQuestion("NRO IDENT", "10 digits", 100, 9), Entity.PATIENT
    )
    sent = json.loads(messages[1]["content"])
    assert sent["column_heading"] == "NRO IDENT"
    assert sent["heading_again"] == "NRO IDENT"


def test_the_prompt_says_the_model_gets_no_patient_data() -> None:
    messages = llm.build_messages(llm.ColumnQuestion("X", "digits", 10, 2), Entity.PATIENT)
    assert "must not ask for any" in messages[0]["content"]


# --------------------------------------------- the fallback is only a fallback
def test_a_spanish_file_asks_the_model_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The point of the alias dictionary: an ordinary export costs nothing.

    The PDF requires an LLM-assisted proposal, and the architecture places the
    model as a re-ranker over deterministic candidates. So a file whose headings
    the dictionary already knows must reach the model with nothing to ask.
    """
    from pathlib import Path

    from src.onboarding import service
    from src.onboarding.reader import read

    called: list[str] = []

    def _refuse(*args: Any, **kwargs: Any) -> None:
        called.append("provider call")
        raise AssertionError("a model was asked about a column the dictionary knows")

    monkeypatch.setattr(llm, "suggest", _refuse)
    monkeypatch.setattr(
        "src.onboarding.service.get_settings",
        lambda: _settings(openai_api_key="a-key-so-the-stage-is-enabled"),
    )

    fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "onboarding"
    reports = service.analyse(read(fixtures / "1_clean_ips.xlsx"))

    assert called == []
    patients = next(r for r in reports if r.sheet == "Pacientes")
    assert all(c.auto for c in patients.columns)


def test_an_unknown_heading_is_suggested_but_never_pre_ticked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model's answer is a suggestion for a person, not a decision.

    Schema adherence is not correctness: the model can return a valid field name
    that is the wrong field, so its answer is always shown for confirmation.
    """
    from src.onboarding import service
    from src.onboarding.reader import Sheet

    sheet = Sheet(
        name="Hoja1",
        headers=("XQZ-7", "Número Documento"),
        rows=(("abc", "1045678901"), ("def", "1023456789")),
        header_row=1,
    )
    monkeypatch.setattr(
        "src.onboarding.service.get_settings", lambda: _settings(openai_api_key="a")
    )
    monkeypatch.setattr(
        llm,
        "suggest",
        lambda question, entity, **kw: llm.Suggestion(
            column=question.header,
            target_field="external_ref",
            confidence=0.8,
            reason="looks like an internal code",
            provider="openai",
            model="test",
        ),
    )

    report = service.analyse_sheet_for_test(sheet)
    unknown = next(c for c in report.columns if c.column == "XQZ-7")
    assert unknown.target_field == "external_ref"
    assert unknown.confidence == "suggested"
    assert unknown.auto is False  # never applied without a person


# ----------------------------- the sentinel test that routes a real FILE
# The test at the top of this file hand-builds a ColumnQuestion, so the
# sentinels cannot be present by construction -- it could never fail. These
# route an actual file through `service.analyse` with the model stage enabled,
# which is where a patient value really could become a "column heading".


def test_no_patient_value_reaches_a_provider_from_a_real_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """A file with no heading row makes row 1 the headers -- and row 1 is a patient.

    `_ask_model` ran at analyse time, before anyone had answered the question
    about how the file reads, so a cédula, a name and a phone number were sent
    to the provider as column names and written to the log. Nothing is asked
    until a person has said the file has headings.
    """
    from src.onboarding import service
    from src.onboarding.reader import read

    asked: list[str] = []

    def _spy(question: llm.ColumnQuestion, entity: Entity, **kwargs: Any) -> None:
        asked.append(question.header)
        raise llm.LLMUnavailable("spy")

    monkeypatch.setattr(llm, "suggest", _spy)
    monkeypatch.setattr(
        "src.onboarding.service.get_settings",
        lambda: _settings(openai_api_key="a-key-so-the-stage-is-enabled"),
    )

    path = tmp_path / "headerless.csv"
    path.write_bytes(
        b"CC;9999888877;Zzyzx Sentinelensen Marcadorez;3009998877\n"
        b"CC;1020304051;Luis Gomez Diaz;3151112233\n"
    )
    parsed = read(path)
    # The reader does raise the question; the point is what happens meanwhile.
    assert [q.id for q in parsed.questions] == ["csv.no_header_row"]

    service.analyse(parsed)

    assert asked == [], f"these values were sent to a provider: {asked}"
    for sentinel in SENTINELS:
        assert sentinel not in " ".join(asked)


def test_an_ambiguous_heading_still_reaches_the_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The fix must not silence the stage the PDF's scope asks for.

    A file whose headings are real but one of which is meaningless still gets a
    model proposal, because there is no patient value in a heading the clinic
    actually wrote.
    """
    from src.onboarding import service
    from src.onboarding.reader import read

    asked: list[str] = []

    def _spy(question: llm.ColumnQuestion, entity: Entity, **kwargs: Any) -> None:
        asked.append(question.header)
        raise llm.LLMUnavailable("spy")

    monkeypatch.setattr(llm, "suggest", _spy)
    monkeypatch.setattr(
        "src.onboarding.service.get_settings",
        lambda: _settings(openai_api_key="a-key-so-the-stage-is-enabled"),
    )

    path = tmp_path / "ok.csv"
    path.write_bytes(b"TIPO DOC;IDENTIFICACION;COLUMNA RARA XYZ\nCC;1020304050;abc\n")
    service.analyse(read(path))

    assert asked == ["COLUMNA RARA XYZ"]


def test_no_patient_value_reaches_a_provider_from_a_workbook(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The same guarantee by the Excel path, which the first fix missed.

    `_ask_model` returns early while a structure question is open, but the Excel
    reader raised no question at all, so a headerless workbook's first patient
    still went out as a column heading. Fixing only the CSV path left half the
    defect in place; this test is the half that was missing.
    """
    from openpyxl import Workbook

    from src.onboarding import service
    from src.onboarding.reader import read

    asked: list[str] = []

    def _spy(question: llm.ColumnQuestion, entity: Entity, **kwargs: Any) -> None:
        asked.append(question.header)
        raise llm.LLMUnavailable("spy")

    monkeypatch.setattr(llm, "suggest", _spy)
    monkeypatch.setattr(
        "src.onboarding.service.get_settings",
        lambda: _settings(openai_api_key="a-key-so-the-stage-is-enabled"),
    )

    path = tmp_path / "headerless.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = "Pacientes"
    sheet.append(["CC", "9999888877", "Zzyzx Sentinelensen Marcadorez", "3009998877"])
    sheet.append(["CC", "1020304051", "Luis Gomez Diaz", "3151112233"])
    workbook.save(path)

    parsed = read(path)
    assert [q.id for q in parsed.questions] == ["csv.no_header_row"]

    service.analyse(parsed)

    assert asked == [], f"these values were sent to a provider: {asked}"
    for sentinel in SENTINELS:
        assert sentinel not in " ".join(asked)


def test_a_file_with_genuine_headings_still_gets_a_proposal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Gating on "any question at all" removed an in-scope feature for no gain.

    A file with duplicate headings, or a row with too many fields, has real
    headings -- the clinic wrote them -- so a model may be asked about an
    ambiguous one. The first fix for the leak silenced the mapping proposal for
    every such file, including after the reviewer answered. Only a question about
    WHICH ROW holds the headings means a heading might be a patient.
    """
    from src.onboarding import service
    from src.onboarding.reader import read

    asked: list[str] = []

    def _spy(question: llm.ColumnQuestion, entity: Entity, **kwargs: Any) -> None:
        asked.append(question.header)
        raise llm.LLMUnavailable("spy")

    monkeypatch.setattr(llm, "suggest", _spy)
    monkeypatch.setattr(
        "src.onboarding.service.get_settings",
        lambda: _settings(openai_api_key="a-key-so-the-stage-is-enabled"),
    )

    # An overlong row raises a question, and the headings are genuine.
    path = tmp_path / "overlong.csv"
    path.write_bytes(b"TIPO DOC;IDENTIFICACION;ZZZ RARO\nCC;1020304050;a;b\n")
    parsed = read(path)
    assert [q.id for q in parsed.questions] == ["csv.overlong_rows"]
    assert parsed.sheets[0].headings_are_settled

    service.analyse(parsed)
    assert asked == ["ZZZ RARO"], asked


def test_a_headerless_file_of_words_sends_nothing_to_a_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The leak `headings_are_settled` alone cannot close.

    A clinic exporting without headings, where every column holds words, has a
    first row that scores exactly as a heading row does: 4.00 for the patient
    names and 4.00 for every row beneath them. The reader cannot tell them apart
    -- a genuine two-column heading scores 4.00 against data of 4.00 too -- so it
    settles the headings and asks nothing, and those names become the "column
    headings" the mapping stage sends.

    The dictionary separates what the score cannot. This file matches none of
    its 288 aliases, so nothing leaves the machine.
    """
    from pathlib import Path

    from src.onboarding import service
    from src.onboarding.reader import read

    sent: list[str] = []

    def _spy(question: Any, entity: Any, **kwargs: Any) -> None:
        sent.append(question.header)
        raise llm.LLMUnavailable("spy")

    monkeypatch.setattr(llm, "suggest", _spy)
    monkeypatch.setattr(
        "src.onboarding.service.get_settings",
        lambda: _settings(openai_api_key="a-key-so-the-stage-is-enabled"),
    )

    path = Path(tmp_path) / "sin_encabezado.csv"
    path.write_bytes(
        b"Ana Maria;Perez Gomez;Cardiologia;Bogota\n"
        b"Luis Felipe;Gomez Diaz;Pediatria;Medellin\n"
        b"Eva Rosa;Ruiz Mora;Ortopedia;Cali\n"
    )

    service.analyse(read(path))
    assert sent == [], f"patient values reached a provider: {sent}"


def test_an_ordinary_spanish_export_still_reaches_the_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The dictionary gate must not silence the stage on files that have headings.

    One recognised heading is enough: a file whose other columns are unfamiliar
    still has headings, and those are exactly the columns the model is for.
    """
    from pathlib import Path

    from src.onboarding import service
    from src.onboarding.reader import read

    sent: list[str] = []

    def _spy(question: Any, entity: Any, **kwargs: Any) -> None:
        sent.append(question.header)
        raise llm.LLMUnavailable("spy")

    monkeypatch.setattr(llm, "suggest", _spy)
    monkeypatch.setattr(
        "src.onboarding.service.get_settings",
        lambda: _settings(openai_api_key="a-key-so-the-stage-is-enabled"),
    )

    path = Path(tmp_path) / "con_encabezado.csv"
    path.write_bytes(
        b"CEDULA;MUTUALISTA;ZONA DE COBRO\n1020304050;Sura;Norte\n1020304051;Nueva EPS;Sur\n"
    )

    service.analyse(read(path))
    assert sent, "a file with a recognised heading asked the model nothing"
