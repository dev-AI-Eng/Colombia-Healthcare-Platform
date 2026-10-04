"""Response shapes for the onboarding API.

Written to be read in Swagger UI by a person deciding whether an import is safe,
so every field is one they would ask about: what did you think this column was,
how sure are you, how much of it converted, and what still needs me.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from src.onboarding import service


class ColumnOut(BaseModel):
    column: str = Field(description="The heading exactly as the file spells it.")
    target_field: str | None = Field(description="The canonical field it will be imported into.")
    confidence: str = Field(description="exact, strong, weak or none.")
    reason: str = Field(description="Why this column was proposed for that field.")
    auto: bool = Field(description="Whether the proposal is pre-ticked for confirmation.")
    total: int = 0
    valid: int = 0
    review: int = 0
    invalid: int = 0
    percent_valid: float = Field(
        default=0.0,
        description="Share of rows that converted. Shown instead of a preview of "
        "the first rows, which looks fine while row 100 is broken.",
    )

    @classmethod
    def build(cls, report: service.ColumnReport) -> ColumnOut:
        return cls(
            column=report.column,
            target_field=report.target_field,
            confidence=report.confidence,
            reason=report.reason,
            auto=report.auto,
            total=report.total,
            valid=report.valid,
            review=report.review,
            invalid=report.invalid,
            percent_valid=round(report.percent_valid, 1),
        )


class SheetOut(BaseModel):
    sheet: str
    entity: str = Field(description="What this sheet is taken to hold.")
    entity_reason: str
    columns: list[ColumnOut]
    missing_required: list[str] = Field(
        description="Required fields no column was found for. The import cannot proceed "
        "until these are mapped or the file is re-exported."
    )
    questions: list[str] = Field(
        description="Column-level decisions only a person can make, such as whether "
        "a date column is day/month or month/day."
    )
    warnings: list[str] = Field(
        description="Structural findings: hidden rows, a header below title rows, "
        "an unusual encoding."
    )
    total_rows: int
    valid_rows: int
    review_rows: int
    invalid_rows: int

    @classmethod
    def build(cls, report: service.SheetReport) -> SheetOut:
        return cls(
            sheet=report.sheet,
            entity=report.entity.value,
            entity_reason=report.entity_reason,
            columns=[ColumnOut.build(c) for c in report.columns],
            missing_required=list(report.missing_required),
            questions=list(report.questions),
            warnings=list(report.warnings),
            total_rows=report.total_rows,
            valid_rows=report.valid_rows,
            review_rows=report.review_rows,
            invalid_rows=report.invalid_rows,
        )


class StructureQuestionOut(BaseModel):
    """One thing about the file's shape that a reviewer must decide."""

    id: str
    sheet: str
    finding: str
    if_approved: str
    if_declined: str
    answered: bool | None = Field(
        default=None,
        description="True approved, False declined, null not yet answered.",
    )


class UploadOut(BaseModel):
    session_id: uuid.UUID
    filename: str
    file_sha256: str = Field(description="Identifies a byte-identical re-upload.")
    encoding: str | None = None
    delimiter: str | None = None
    sheets: list[SheetOut]
    structure_questions: list[StructureQuestionOut] = Field(
        default_factory=list,
        description="Things about the file's shape that must be approved or declined "
        "before the import can be trusted. Declining reads the file as written.",
    )
    status: str
    reused_profiles: list[str] = Field(
        default_factory=list,
        description="Sheets whose mapping came from a profile this clinic already "
        "confirmed. These need no review and no model call.",
    )
    duplicate_of: uuid.UUID | None = Field(
        default=None,
        description="A byte-identical file this clinic already committed. Importing "
        "it again would duplicate work someone has already done.",
    )


class CellOut(BaseModel):
    """One cell, and what the rule made of it. The ADR-08a transform log."""

    row_number: int
    column: str
    target_field: str | None
    raw: str = Field(description="What the file held, never a reviewer's answer.")
    corrected_from_review: str | None = Field(
        default=None,
        description="What a reviewer supplied in place of `raw`, if they answered this cell.",
    )
    normalized: str | None
    rule: str = Field(description="The named rule that produced this result.")
    status: str
    message: str = ""
    means: str = Field(
        default="",
        description="What the rule means, in Spanish, for the receptionist acting on "
        "it. Written against the rule name, so it never contains the cell's value.",
    )
    action: str = Field(
        default="",
        description="What to do about it. Never 'correct the value' for a document or "
        "a phone number: the nearest valid one belongs to somebody else.",
    )


class RowOut(BaseModel):
    row_number: int = Field(
        description="The row in the source file, so a person can go and fix it."
    )
    status: str
    values: dict[str, Any]
    errors: list[str]
    reviews: list[str]


class ValidationOut(BaseModel):
    session_id: uuid.UUID
    status: str
    sheets: list[SheetOut]
    can_commit: bool = Field(
        description="False while any row is invalid. Commit refuses in that state, "
        "which is the milestone's exit criterion."
    )
    blocking: list[str] = Field(description="Why it cannot be committed yet.")
    tolerated: list[str] = Field(
        default_factory=list,
        description="Failures small enough that the rest of the file still imports. "
        "Reported, never hidden: the rows themselves are in the review queue.",
    )


class CommitOut(BaseModel):
    session_id: uuid.UUID
    status: str
    committed: dict[str, int] = Field(description="Rows written, per sheet.")
    created: dict[str, int] = Field(
        default_factory=dict, description="Rows that became a new record, per sheet."
    )
    updated: dict[str, int] = Field(
        default_factory=dict,
        description="Rows that replaced an existing record, per sheet. A clinic that "
        "has maintained a record by hand since the last import loses those edits, so "
        "the count is reported separately rather than folded into `committed`.",
    )
    skipped: dict[str, int] = Field(
        description="Rows not written per sheet, because they await review."
    )
    conflicts: list[str] = Field(
        default_factory=list,
        description="Rows a person must decide about, such as a match against a "
        "patient who was previously deleted.",
    )
    message: str


class ProfileOut(BaseModel):
    """A mapping this clinic confirmed, reused when a file of that shape returns."""

    id: uuid.UUID
    name: str
    header_fingerprint: str = Field(
        description="The sorted, normalized headers hashed: what identifies a file "
        "of this shape even if columns are reordered or recapitalised."
    )
    entity: str
    mapping: dict[str, str]
    version: int = Field(description="Incremented each time a reviewer changes it.")
    updated_at: datetime


class StructureAnswerIn(BaseModel):
    """Approve or decline one structure question.

    Approving applies the fix the question describes. Declining discards that
    one finding and reads the file as written, leaving every other question and
    every other row untouched.
    """

    id: str
    approved: bool


class CorrectionIn(BaseModel):
    """One reviewer's answer to one row the file could not decide.

    The correction supplies the cell's text, not its converted value: it goes
    through the same normalizer as every other cell, so answering "Pérez Gómez
    is the surname" cannot introduce a value the rules would have rejected.
    """

    sheet: str
    row_number: int = Field(description="The row number as the file shows it.")
    cells: dict[str, str] = Field(
        description="Column name to the corrected text for that cell.",
        min_length=1,
    )


class MappingIn(BaseModel):
    """A reviewer's corrections: column heading to canonical field, or null to skip."""

    sheet: str
    mapping: dict[str, str | None]
    entity: str | None = Field(
        default=None,
        description="What this sheet holds: patient, doctor, specialty, availability "
        "or appointment. A sheet called AGENDA carrying patient columns is genuinely "
        "ambiguous, so the reviewer decides rather than the matcher guessing harder. "
        "Send 'skip' to leave the sheet out of the import entirely, which is how a "
        "stale sheet left in the workbook is dealt with.",
    )
    decisions: dict[str, str] = Field(
        default_factory=dict,
        description="Answers to the sheet's questions, e.g. {'FECHA': 'day_first'}.",
    )
    excluded_rows: list[int] = Field(
        default_factory=list,
        description="Rows in the file that are not records at all: a TOTAL line, a "
        "heading for a second table pasted below the first, a note to staff. They "
        "are left out of the import rather than reported as broken patients.",
    )
