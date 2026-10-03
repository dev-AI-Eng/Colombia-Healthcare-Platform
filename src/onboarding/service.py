"""Turn an uploaded file into staged rows a person can review, then commit them.

The order matters and is not negotiable:

    analyse  -> read the file, propose a mapping, report what needs confirming
    validate -> run the normalizers over EVERY row, write staging rows
    commit   -> apply, in one transaction, only if nothing is invalid

Nothing reaches the clinic's real tables until `commit`, and `commit` refuses
while any row is invalid. That is the milestone's exit criterion: "no import
commits data that fails validation".

`validate` runs over 100% of rows, never a sample, and records what every rule
did to every cell (ADR-08a). The cost is one log row per cell; the benefit is
that a reviewer can answer "why is this patient's phone empty?" without
re-running anything.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from src.core.config import get_settings
from src.onboarding import normalizers as norm
from src.onboarding.canonical import (
    FIELDS_BY_ENTITY,
    Entity,
    Field,
    field_for,
    required_fields,
)
from src.onboarding.matcher import SHARED_FIELDS, Proposal, SheetMapping, guess_entity, match_sheet
from src.onboarding.reader import ReadResult, Sheet

# A column-level decision the file cannot make for itself. Held on the session
# so the human answers it once, and every row then follows the same rule.
type ColumnDecisions = dict[str, str]


@dataclass(frozen=True, slots=True)
class CellResult:
    """What one rule did to one cell. Written to the transform log."""

    row_number: int
    column: str
    target_field: str | None
    #: What the **file** held. A reviewer's answer never overwrites this: the
    #: log has to keep showing what arrived, or it stops being evidence.
    raw: str
    normalized: str | None
    rule: str
    status: norm.Status
    message: str = ""
    #: What a reviewer supplied in place of `raw`, when they answered this cell.
    #: Present only on a corrected cell, so the log distinguishes a value the
    #: clinic exported from one a person decided.
    corrected_from_review: str | None = None


@dataclass(slots=True)
class RowResult:
    row_number: int
    entity: Entity
    raw: dict[str, str]
    values: dict[str, Any] = field(default_factory=dict)
    cells: list[CellResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    reviews: list[str] = field(default_factory=list)

    @property
    def status(self) -> norm.Status:
        if self.errors:
            return norm.Status.INVALID
        if self.reviews:
            return norm.Status.REVIEW
        return norm.Status.VALID


@dataclass(frozen=True, slots=True)
class ColumnReport:
    """What the confirmation screen shows for one column.

    `percent_valid` is deliberately the headline rather than a preview of the
    first few rows: a preview looks fine while the hundredth row is broken
    (ADR-08a).
    """

    column: str
    target_field: str | None
    confidence: str
    reason: str
    auto: bool
    total: int = 0
    valid: int = 0
    review: int = 0
    invalid: int = 0

    @property
    def percent_valid(self) -> float:
        return 100.0 * self.valid / self.total if self.total else 0.0


@dataclass(frozen=True, slots=True)
class SheetReport:
    sheet: str
    entity: Entity
    entity_reason: str
    columns: tuple[ColumnReport, ...]
    missing_required: tuple[str, ...]
    total_rows: int
    valid_rows: int
    review_rows: int
    invalid_rows: int
    warnings: tuple[str, ...] = ()
    #: Column-level questions only a person can answer, e.g. an ambiguous date
    #: format. The import cannot be validated until each is answered.
    questions: tuple[str, ...] = ()


def analyse(result: ReadResult) -> tuple[SheetReport, ...]:
    """Propose a mapping for every sheet, without converting anything yet."""
    return tuple(_analyse_sheet(sheet) for sheet in result.sheets)


def _column_index(sheet: Sheet) -> dict[str, int]:
    """Position of each heading, keeping the FIRST where a heading repeats.

    A dict comprehension keeps the last, so a file with two columns called
    NOMBRES converted the second one twice and never read the first -- while the
    duplicate-headers question promises that declining "imports the first of
    each". Numbering the repeats is what the reviewer approves; declining has to
    mean what it says.
    """
    index: dict[str, int] = {}
    for position, header in enumerate(sheet.headers):
        index.setdefault(header, position)
    return index


def _ask_model(sheet: Sheet, entity: Entity, mapping: SheetMapping) -> dict[str, ColumnReport]:
    """Ask a model about the columns the deterministic stages could not resolve.

    The PDF's scope includes an LLM-assisted mapping proposal, and the
    architecture places it as a re-ranker over deterministic candidates rather
    than as the mapper. So this runs last, over the residue, and only when a key
    is configured: an ordinary Spanish export reaches it with nothing to ask.

    A failure here is not an error. The column keeps whatever the deterministic
    stages thought and goes to the person confirming the import, which is
    exactly where it would have gone with no provider at all.
    """
    from src.onboarding import llm

    settings = get_settings()
    if not settings.mapping_llm_available:
        return {}

    # A heading is only safely a heading once we know the file HAS headings.
    # While a question about which row holds them is open, the "headers" may be
    # row 1 of the data -- a cédula, a name, a phone -- and sending those as
    # column names would put patient values in a provider's payload and in our
    # logs.
    #
    # Only those questions silence this stage. A file with duplicate headings or
    # an overlong row has genuine headings, and gating on "any question at all"
    # removed an in-scope feature from those files for no safety gain.
    if not sheet.headings_are_settled:
        return {}

    unresolved = [p for p in mapping.proposals if not p.auto]
    if not unresolved:
        return {}

    taken = {p.field.name for p in mapping.proposals if p.auto and p.field}
    candidates = tuple(f.name for f in FIELDS_BY_ENTITY[entity] if f.name not in taken)
    index = _column_index(sheet)
    improved: dict[str, ColumnReport] = {}

    for proposal in unresolved:
        values = [row[index[proposal.column]] for row in sheet.rows]
        profile = llm.profile_column(values)
        question = llm.ColumnQuestion(
            header=proposal.column,
            shape=profile.shape,
            filled_percent=profile.filled_percent,
            distinct_count=profile.distinct_count,
            synthetic_examples=profile.examples,
        )
        try:
            suggestion = llm.suggest(question, entity, candidates=candidates)
        except llm.LLMUnavailable:
            continue  # the human decides, as they would have anyway
        if suggestion.target_field is None or suggestion.target_field in taken:
            continue
        taken.add(suggestion.target_field)
        improved[proposal.column] = ColumnReport(
            column=proposal.column,
            target_field=suggestion.target_field,
            # Never pre-ticked. A model's answer is a suggestion for a person to
            # confirm, not a decision: schema adherence is not correctness.
            confidence="suggested",
            reason=f"{suggestion.provider} suggests this: {suggestion.reason}",
            auto=False,
        )
    return improved


def reanalyse(sheet: Sheet, entity: Entity) -> SheetReport:
    """Re-propose a sheet's columns under an entity a reviewer chose.

    A sheet called AGENDA carrying patient columns is genuinely ambiguous, so
    when the reviewer says what it is, its columns are matched against that
    entity's fields rather than the one that was guessed.
    """
    mapping = match_sheet(sheet.headers, entity)
    return SheetReport(
        sheet=sheet.name,
        entity=entity,
        entity_reason="Chosen by the reviewer.",
        columns=tuple(_column_report(p) for p in mapping.proposals),
        missing_required=mapping.missing_required,
        total_rows=len(sheet.rows),
        valid_rows=0,
        review_rows=0,
        invalid_rows=0,
        warnings=sheet.warnings,
        questions=_column_questions(sheet, mapping),
    )


def analyse_sheet_for_test(sheet: Sheet) -> SheetReport:
    """Analyse one sheet. Exposed for tests that build a sheet directly."""
    return _analyse_sheet(sheet)


#: The entity a column is offered to when the sheet's own entity has no field
#: for it, in the order they are tried. A clinic's sheet is one row per visit:
#: the patient columns and the appointment columns sit side by side, and until
#: this existed the three columns belonging to the other entity were simply
#: dropped. HubSpot calls the same shape "one file, multiple objects": each
#: column is assigned an object as well as a property, and the parent's columns
#: repeat on every row.
#:
#: Patient first, because a sheet about anything else still names its patient,
#: and a patient is the record everything else refers to.
_SECONDARY_ENTITIES: Final[tuple[Entity, ...]] = (
    Entity.PATIENT,
    Entity.APPOINTMENT,
    Entity.DOCTOR,
)


def _secondary_proposals(
    sheet: Sheet, primary: Entity, mapping: SheetMapping
) -> dict[str, tuple[Entity, Proposal, str]]:
    """Columns the sheet's own entity cannot use, matched against the others.

    Only columns left over are offered, and only to the dictionary and fuzzy
    stages -- never to a model. A model asked "which appointment field is this"
    about a column that is really a patient's is being asked the wrong question,
    and the answer would be a guess dressed as a proposal.
    """
    unmapped = tuple(p.column for p in mapping.proposals if p.field is None)
    if not unmapped:
        return {}

    found: dict[str, tuple[Entity, Proposal, str]] = {}
    for entity in _SECONDARY_ENTITIES:
        if entity is primary:
            continue
        remaining = tuple(c for c in unmapped if c not in found)
        if not remaining:
            break
        for proposal in match_sheet(remaining, entity).proposals:
            # Only a confident match crosses an entity boundary. A fuzzy guess
            # that a patient column is really a doctor's would move a person's
            # name into the wrong table, which is worse than leaving it unmapped
            # for a reviewer to assign.
            if proposal.field is not None and proposal.auto:
                found[proposal.column] = (entity, proposal, proposal.field.name)
    return found


def _analyse_sheet(sheet: Sheet) -> SheetReport:
    entity, reason = guess_entity(sheet.name, sheet.headers)
    mapping = match_sheet(sheet.headers, entity)
    questions = _column_questions(sheet, mapping)
    suggested = _ask_model(sheet, entity, mapping)
    secondary = _secondary_proposals(sheet, entity, mapping)
    columns = tuple(
        _secondary_report(*secondary[p.column])
        if p.column in secondary
        else (suggested.get(p.column) or _column_report(p))
        for p in mapping.proposals
    )
    return SheetReport(
        sheet=sheet.name,
        entity=entity,
        entity_reason=reason,
        columns=columns,
        missing_required=mapping.missing_required,
        total_rows=len(sheet.rows),
        valid_rows=0,
        review_rows=0,
        invalid_rows=0,
        warnings=sheet.warnings,
        questions=questions,
    )


#: Separates an entity from a field in a qualified target, e.g.
#: "appointment.status". A bare name still means the sheet's own entity, so
#: every mapping written before this existed keeps its meaning.
ENTITY_SEPARATOR: Final = "."


def qualified(entity: Entity, field_name: str) -> str:
    """How a field of another entity is named on the confirmation screen."""
    return f"{entity.value}{ENTITY_SEPARATOR}{field_name}"


def split_target(target: str, default: Entity) -> tuple[Entity, str]:
    """The entity and field a mapping names, defaulting to the sheet's own.

    An unqualified name belongs to `default`, which is what every stored
    profile and every hand-written mapping contains.
    """
    head, separator, tail = target.partition(ENTITY_SEPARATOR)
    if not separator:
        return default, target
    try:
        return Entity(head), tail
    except ValueError:
        # Not an entity prefix. A field name containing a dot is not one we
        # define, but treating it as qualified would silently drop the column.
        return default, target


def _secondary_report(entity: Entity, proposal: Proposal, field_name: str) -> ColumnReport:
    """A column that belongs to an entity other than the sheet's own."""
    return ColumnReport(
        column=proposal.column,
        target_field=qualified(entity, field_name),
        confidence=proposal.confidence.value,
        reason=f"{proposal.reason} Belongs to the {entity.value} this row describes.",
        auto=proposal.auto,
    )


def _column_report(proposal: Proposal) -> ColumnReport:
    return ColumnReport(
        column=proposal.column,
        target_field=proposal.field.name if proposal.field else None,
        confidence=proposal.confidence.value,
        reason=proposal.reason,
        auto=proposal.auto,
    )


def _column_questions(sheet: Sheet, mapping: SheetMapping) -> tuple[str, ...]:
    """Ask about anything the column as a whole cannot decide.

    Only date order arises today: a column whose every value is ambiguous is
    either day-first or month-first, and picking one silently would misdate
    every row in it.
    """
    questions: list[str] = []
    index = _column_index(sheet)
    for proposal in mapping.proposals:
        if proposal.field is None or proposal.field.normalizer != "date":
            continue
        column_values = [row[index[proposal.column]] for row in sheet.rows]
        # Only ask when a value would actually be refused. An ISO column is
        # undecided too, because nothing in it is day-or-month, but every value
        # parses. Asking anyway teaches reviewers to click past questions,
        # which is how the one that matters gets missed.
        ambiguous = any(
            norm.date(value, order=norm.DayFirst.UNDECIDED).status is norm.Status.REVIEW
            for value in column_values
            if value.strip()
        )
        if ambiguous and norm.detect_day_first(column_values) is norm.DayFirst.UNDECIDED:
            questions.append(
                f"{proposal.column}: dates could be day/month or month/day. "
                "Which is it? (day_first or month_first)"
            )
    return tuple(questions)


def validate(
    sheet: Sheet,
    entity: Entity,
    mapping: dict[str, str | None],
    *,
    decisions: ColumnDecisions | None = None,
    excluded_rows: set[int] | None = None,
    corrections: dict[int, dict[str, str]] | None = None,
) -> tuple[list[RowResult], tuple[ColumnReport, ...]]:
    """Run every normalizer over every row of one sheet.

    `mapping` is what the human confirmed: column name to canonical field name,
    or None for a column that is deliberately not imported.

    `corrections` replaces a cell's text before conversion, keyed by row number
    then column. A reviewer answering "which of these three words is the
    surname?" supplies the value; it is still converted by the same normalizer
    as every other cell, so a corrected value that is itself invalid is caught
    exactly like an original one.
    """
    decisions = decisions or {}
    index = _column_index(sheet)

    # Date order is decided once per column, from the whole column, before any
    # row is converted. Deciding per row would let one file contain both
    # readings, which is how a birth date silently becomes a different date.
    # Which entity each confirmed column belongs to. A sheet is one row per
    # visit as often as it is one row per patient, so a column may name a field
    # of an entity other than the sheet's own: see `_SECONDARY_ENTITIES`.
    targets: dict[str, tuple[Entity, str]] = {
        column: split_target(target, entity) for column, target in mapping.items() if target
    }

    orders: dict[str, norm.DayFirst] = {}
    for column, target in mapping.items():
        column_entity, field_name = targets.get(column, (entity, ""))
        canonical = field_for(column_entity, field_name) if target else None
        if canonical and canonical.normalizer == "date":
            answer = decisions.get(column)
            orders[column] = (
                norm.DayFirst(answer)
                if answer in {"day_first", "month_first"}
                else norm.detect_day_first([row[index[column]] for row in sheet.rows])
            )

    rows: list[RowResult] = []
    counts: dict[str, Counter[str]] = {column: Counter() for column in mapping}

    excluded = excluded_rows or set()
    # The row number the reviewer sees, which is not the position in `rows`: a
    # blank line between two patients is dropped while reading, so counting from
    # the header puts every later row one out. Everything keyed on a row number
    # -- the hidden-row flag, `excluded_rows`, a refusal naming a line to fix --
    # then points at the wrong patient. The reader supplies the real numbers;
    # the count is the fallback for a sheet that dropped nothing.
    numbers = sheet.row_numbers or tuple(
        range(sheet.header_row + 1, sheet.header_row + 1 + len(sheet.rows))
    )
    for offset, raw_row in zip(numbers, sheet.rows, strict=True):
        # A row the reviewer marked as not a record — a totals line, the heading
        # of a second table — is left out entirely rather than counted as a
        # broken patient, which would block the import for no reason.
        if offset in excluded:
            continue
        corrected = (corrections or {}).get(offset, {})
        # Converted in the file's column order, not the mapping's. The mapping
        # arrives from JSONB, which does not preserve insertion order, and two
        # columns joined into one field (primerApellido + segundoApellido) would
        # otherwise be joined in whatever order the database handed back --
        # storing "Gomez Perez" for a patient whose file says "Perez Gomez".
        # In the file's column order, each heading once. A repeated heading
        # names one column as far as the mapping is concerned -- the first, per
        # `_column_index` -- so listing it twice converted that one column twice
        # and joined it to itself ("Ana Ana") for a shared name field.
        in_file_order = [
            (header, mapping[header])
            for header in dict.fromkeys(sheet.headers)
            if header in mapping
        ]
        raw_cells = {
            header: corrected.get(header, raw_row[index[header]]) for header in sheet.headers
        }
        # One source row becomes one result per entity it describes. A sheet of
        # visits holds a patient and their appointment side by side, and the
        # patient repeats on every row they appear in; `find_duplicates` and the
        # apply step both key on identity, so the repeat resolves to one record.
        by_entity: dict[Entity, RowResult] = {
            entity: RowResult(row_number=offset, entity=entity, raw=dict(raw_cells))
        }
        row = by_entity[entity]
        for column, target in in_file_order:
            if target is None:
                continue
            column_entity, field_name = targets.get(column, (entity, target))
            canonical = field_for(column_entity, field_name)
            if canonical is None:
                continue
            if column_entity not in by_entity:
                by_entity[column_entity] = RowResult(
                    row_number=offset, entity=column_entity, raw=dict(raw_cells)
                )
            row = by_entity[column_entity]
            original = raw_row[index[column]]
            # The reviewer's answer, where they gave one, so the correction is
            # what gets converted rather than only what gets displayed.
            answer = corrected.get(column)
            raw_value = original if answer is None else answer
            outcome = _apply(canonical, raw_value, orders.get(column, norm.DayFirst.UNDECIDED))
            counts[column][outcome.status.value] += 1
            row.cells.append(
                CellResult(
                    row_number=offset,
                    column=column,
                    target_field=target,
                    raw=original,
                    normalized=_text(outcome.value),
                    rule=outcome.rule,
                    status=outcome.status,
                    message=outcome.message,
                    corrected_from_review=answer,
                )
            )
            if outcome.status is norm.Status.VALID:
                if field_name in SHARED_FIELDS and field_name in row.values:
                    # A second column for the same name field: joined in the
                    # file's column order, so "primerNombre" then
                    # "segundoNombre" reads as the person writes their name.
                    existing = str(row.values[field_name]).strip()
                    addition = str(_text(outcome.value) or "").strip()
                    row.values[field_name] = f"{existing} {addition}".strip()
                else:
                    row.values[field_name] = outcome.value
            elif outcome.status is norm.Status.INVALID:
                row.errors.append(f"{column}: {outcome.message}")
            else:
                row.reviews.append(f"{column}: {outcome.message}")

        # The reader's warning says hidden rows are "imported and flagged for
        # review", and until this was added nothing flagged them: a hidden row
        # whose cells all converted was committed as valid. Hiding a row is not
        # deleting it, and the usual reason is a cancellation the clinic never
        # removed, so a person decides.
        for produced in by_entity.values():
            if offset in sheet.hidden_row_numbers:
                produced.reviews.append(
                    "This row is hidden in the file. Hiding is not deleting, so confirm "
                    "whether it should be imported."
                )
        rows.extend(by_entity.values())

    # In the file's own column order, and including the columns nobody mapped.
    # Iterating the mapping instead would reorder the screen against the
    # spreadsheet the reviewer is comparing it to, and would hide every unmapped
    # column so they could not be mapped at all.
    reports = tuple(
        ColumnReport(
            column=header,
            target_field=mapping.get(header),
            confidence="confirmed" if mapping.get(header) else "not imported",
            reason=(
                "Confirmed by the reviewer."
                if mapping.get(header)
                else "Deliberately not imported."
            ),
            auto=True,
            total=sum(counts[header].values()) if header in counts else 0,
            valid=counts[header][norm.Status.VALID.value] if header in counts else 0,
            review=counts[header][norm.Status.REVIEW.value] if header in counts else 0,
            invalid=counts[header][norm.Status.INVALID.value] if header in counts else 0,
        )
        for header in sheet.headers
    )
    return rows, reports


def _apply(canonical: Field, raw: str, order: norm.DayFirst) -> norm.Outcome[Any]:
    """Run the one normalizer bound to this field.

    The binding lives in `canonical.py` and is fixed in code. Nothing here
    chooses a transform from the data, which is the whole point of ADR-08a.
    """
    match canonical.normalizer:
        case "document_type":
            return norm.document_type(raw)
        case "document_number":
            return norm.document_number(raw)
        case "phone":
            return norm.phone(raw)
        case "date":
            converted = norm.date(raw, order=order)
            # The range belongs to the field: `date()` serves a birth date and an
            # appointment date, which disagree about whether next year is wrong.
            if (
                converted.status is norm.Status.VALID
                and converted.value is not None
                and (complaint := norm.plausible_date(converted.value, canonical.name))
            ):
                return norm.Outcome(norm.Status.REVIEW, None, "date.implausible", complaint)
            return converted
        case "time":
            return norm.time_of_day(raw)
        case "status":
            return norm.appointment_status(raw)
        case "full_name":
            return norm.split_full_name(raw)
        case "boolean":
            return norm.boolean(raw)
        case "weekday":
            return norm.weekday(raw)
        case "consent_purpose":
            return norm.consent_purpose(raw)
        case "consent_evidence":
            return norm.consent_evidence(raw)
        case _:
            # No normalizer: the value is stored as written, trimmed. An empty
            # optional field is not an error.
            text = raw.strip()
            # A cell beginning with one of these is a formula to Excel,
            # LibreOffice and Sheets. `split_full_name` refuses them, but most
            # free-text fields -- given_names, family_names, a doctor's or a
            # specialty's name -- have no normalizer and reach here, so the
            # same payload was stored verbatim through any of them. Inert in
            # the database, executable the moment anyone exports the sheet.
            if text and text[0] in norm.FORMULA_LEADS:
                return norm.Outcome(
                    norm.Status.REVIEW,
                    None,
                    "text.looks_like_a_formula",
                    f"{raw!r} starts with {text[0]!r}, so a spreadsheet would read it "
                    "as a formula rather than a value. Confirm what it should say.",
                )
            return norm.Outcome(norm.Status.VALID, text or None, "text.as_written")


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, norm.SplitName):
        return f"{value.given_names} | {value.family_names}"
    if isinstance(value, dt.date | dt.time):
        return value.isoformat()
    return str(value)


#: What makes two rows the same record, per entity. Patient identity is the
#: document pair because that is what `_apply_patients` matches on: keying
#: duplicate detection on anything else would report rows as distinct that the
#: database then merges, or the reverse.
#: Fields that name a record defined elsewhere, and what to call them in a
#: refusal. Appointments are the only entity with references today; availability
#: joins here when M2 computes it.
REFERENCE_FIELDS: Final[dict[Entity, tuple[tuple[str, str], ...]]] = {
    Entity.APPOINTMENT: (
        ("doctor_ref", "the doctor"),
        ("patient_document", "the patient document"),
    ),
    Entity.AVAILABILITY: (("doctor_ref", "the doctor"),),
}

IDENTITY_FIELDS: Final[dict[Entity, tuple[str, ...]]] = {
    # What `_apply_patients` matches on.
    Entity.PATIENT: ("document_type", "document_number"),
    # A doctor has no document column in the canonical schema, so the clinic's
    # own code identifies them where there is one, and the name otherwise.
    Entity.DOCTOR: ("external_ref",),
    Entity.SPECIALTY: ("name",),
}


def find_duplicates(rows: list[RowResult], entity: Entity) -> dict[int, str]:
    """Row numbers that repeat an identity already seen in this file.

    Returned per row rather than as a count, so the message can name the row the
    duplicate collides with — "the same cédula as row 14" is actionable, while
    "3 duplicates" sends a receptionist hunting.

    The first occurrence is not a duplicate: it is the record, and the later ones
    are the repeats. Rows missing part of the identity are skipped, because an
    empty cédula is already a per-cell refusal and reporting it twice would make
    the screen noisier without telling anyone anything new.
    """
    fields = IDENTITY_FIELDS.get(entity)
    if not fields:
        return {}

    first_seen: dict[tuple[str, ...], int] = {}
    duplicates: dict[int, str] = {}
    for row in rows:
        # An empty cell is None, and `str(None)` is "none" after casefolding:
        # it passes the `all(key)` check below, so two rows missing the same
        # identifier would be reported as duplicates of each other.
        key = tuple(
            "" if (v := row.values.get(f)) is None else str(v).strip().casefold() for f in fields
        )
        if not all(key):
            continue
        if key in first_seen:
            shown = " ".join(k for k in key if k)
            duplicates[row.row_number] = (
                f"the same {' and '.join(fields).replace('_', ' ')} as row "
                f"{first_seen[key]} ({shown})"
            )
            continue
        first_seen[key] = row.row_number
    return duplicates


def find_dangling_references(
    rows: list[RowResult], entity: Entity, *, known: Mapping[str, set[str]]
) -> dict[int, str]:
    """Row numbers whose reference points at something the import does not have.

    A sheet of appointments naming a doctor the file never defines would import
    as an appointment with nobody attending it. `known` carries the values each
    referenced field actually has, gathered from the other sheets in the same
    file plus what the clinic already holds, so a doctor already in the database
    is not reported as missing.
    """
    checks = REFERENCE_FIELDS.get(entity, ())
    if not checks:
        return {}

    dangling: dict[int, str] = {}
    for row in rows:
        for field_name, label in checks:
            raw = row.values.get(field_name)
            # `str(None)` is "None", which is not empty, so an empty cell was
            # reported to the receptionist as a missing doctor named 'None'.
            value = "" if raw is None else str(raw).strip()
            if not value:
                continue  # an absent reference is a per-cell concern, not this one
            if value.casefold() not in known.get(field_name, set()):
                dangling[row.row_number] = (
                    f"{label} {value!r} is not in this file and not already in the clinic's records"
                )
                break
    return dangling


#: How many unconvertible rows a sheet may carry and still import the rest.
#: Agreed with the client in writing, and the shape matters more than the
#: numbers:
#:
#:   * a flat share punishes a small file -- a clinic with 18 doctors is refused
#:     over one typo, because 1 of 18 is 6%;
#:   * a flat count lets a systematically broken export through -- 3 bad rows in
#:     4,000 is noise, 3 in 6 is a file with the wrong column order;
#:   * so the allowance is the larger of the two, and then capped, because
#:     without the cap a 6-row file with 3 failures imports: 3 does not exceed a
#:     floor of 3, and the share is never consulted.
#:
#: The cap is what keeps the rule honest at every size. Industrial acceptance
#: sampling (ANSI/ASQ Z1.4) works the same way: the reject number scales with
#: the lot rather than being one fixed percentage.
INVALID_ROW_FLOOR: Final = 3
INVALID_ROW_SHARE: Final = 0.02
INVALID_ROW_CAP: Final = 0.20


def tolerable_invalid_rows(total_rows: int) -> int:
    """How many rows may fail before the sheet itself is refused.

    Zero for an empty sheet: there is nothing to import, so nothing to tolerate.
    """
    if total_rows <= 0:
        return 0
    allowance = max(INVALID_ROW_FLOOR, round(INVALID_ROW_SHARE * total_rows))
    # Never more than the cap, however small the file. `int` truncates, so a
    # 6-row sheet allows 1, not 1.2.
    return min(allowance, int(INVALID_ROW_CAP * total_rows))


def summarise(
    rows: list[RowResult], report: SheetReport, columns: tuple[ColumnReport, ...]
) -> SheetReport:
    """Fold row outcomes back into the sheet report the screen renders."""
    tally = Counter(row.status.value for row in rows)
    return SheetReport(
        sheet=report.sheet,
        entity=report.entity,
        entity_reason=report.entity_reason,
        columns=columns,
        missing_required=report.missing_required,
        total_rows=len(rows),
        valid_rows=tally[norm.Status.VALID.value],
        review_rows=tally[norm.Status.REVIEW.value],
        invalid_rows=tally[norm.Status.INVALID.value],
        warnings=report.warnings,
        questions=report.questions,
    )


def _column_to_dict(column: ColumnReport) -> dict[str, Any]:
    return {
        "column": column.column,
        "target_field": column.target_field,
        "confidence": column.confidence,
        "reason": column.reason,
        "auto": column.auto,
        "total": column.total,
        "valid": column.valid,
        "review": column.review,
        "invalid": column.invalid,
    }


def report_to_dict(report: SheetReport) -> dict[str, Any]:
    """Serialise a sheet report for the session's `report` column."""
    return {
        "sheet": report.sheet,
        "entity": report.entity.value,
        "entity_reason": report.entity_reason,
        "columns": [_column_to_dict(c) for c in report.columns],
        "missing_required": list(report.missing_required),
        "total_rows": report.total_rows,
        "valid_rows": report.valid_rows,
        "review_rows": report.review_rows,
        "invalid_rows": report.invalid_rows,
        "warnings": list(report.warnings),
        "questions": list(report.questions),
    }


def report_from_dict(data: dict[str, Any]) -> SheetReport:
    return SheetReport(
        sheet=data["sheet"],
        entity=Entity(data["entity"]),
        entity_reason=data.get("entity_reason", ""),
        columns=tuple(ColumnReport(**column) for column in data.get("columns", ())),
        missing_required=tuple(data.get("missing_required", ())),
        total_rows=data.get("total_rows", 0),
        valid_rows=data.get("valid_rows", 0),
        review_rows=data.get("review_rows", 0),
        invalid_rows=data.get("invalid_rows", 0),
        warnings=tuple(data.get("warnings", ())),
        questions=tuple(data.get("questions", ())),
    )


def apply_profile(
    report: SheetReport, mapping: dict[str, str | None], entity: Entity | None = None
) -> SheetReport:
    """Re-state a report under a mapping a person confirmed.

    A confirmed column is no longer a proposal, so its confidence becomes
    "confirmed" and it is pre-ticked. A column the reviewer cleared is shown as
    deliberately not imported, rather than as something the matcher failed on.

    `entity` is the sheet type the reviewer confirmed. It must be restored here
    rather than left to the heuristic: a sheet named AGENDA holding patients is
    guessed as `appointment` on every upload, and the stored mapping then targets
    fields the guessed entity does not have. It is also what `missing_required`
    below is computed against, so it has to be settled before that runs.
    """
    entity = entity or report.entity
    columns = tuple(
        ColumnReport(
            column=c.column,
            target_field=mapping.get(c.column),
            confidence="confirmed" if mapping.get(c.column) else "not imported",
            reason=(
                "Confirmed for this clinic."
                if mapping.get(c.column)
                else "Deliberately not imported."
            ),
            auto=True,
            total=c.total,
            valid=c.valid,
            review=c.review,
            invalid=c.invalid,
        )
        for c in report.columns
    )
    assigned = {target for target in mapping.values() if target}
    missing = tuple(
        name for name in (f.name for f in required_fields(entity)) if name not in assigned
    )
    if entity is Entity.PATIENT and (
        "full_name" in assigned or {"given_names", "family_names"} <= assigned
    ):
        missing = tuple(m for m in missing if m not in {"full_name", "given_names", "family_names"})
    return SheetReport(
        sheet=report.sheet,
        entity=entity,
        entity_reason=(
            report.entity_reason
            if entity is report.entity
            else "Confirmed for this clinic on an earlier import."
        ),
        columns=columns,
        missing_required=missing,
        total_rows=report.total_rows,
        valid_rows=report.valid_rows,
        review_rows=report.review_rows,
        invalid_rows=report.invalid_rows,
        warnings=report.warnings,
        questions=report.questions,
    )
