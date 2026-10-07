"""Tables that hold an import in progress (migration 0005).

These live in the `onboarding` schema rather than `app` because a staged row is
a clinic's *file*, not yet the clinic's data: nothing here has been accepted by
a person, and the clinic-facing API must not be able to read it as though it
had. They still carry `clinic_id` and still have a row-level policy, so one
clinic can never see another's upload.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.core.db import Base


class ImportProfile(Base):
    """A mapping a person confirmed once, reused when a file of that shape returns.

    `header_fingerprint` identifies "a file shaped like this one": the sorted,
    normalized headers hashed. A second upload with the same shape reuses the
    confirmed mapping and calls no model at all, which is the PDF's exit
    criterion for repeat imports.

    `mapping` holds the column-to-field mapping only, never a rule derived from
    the values that happened to be in the file (ADR-08a).
    """

    __tablename__ = "import_profiles"
    __table_args__ = (
        Index(
            "uq_import_profiles_clinic_fingerprint",
            "clinic_id",
            "header_fingerprint",
            unique=True,
        ),
        {"schema": "onboarding"},
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    clinic_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("app.clinics.id"))
    name: Mapped[str] = mapped_column(String(120))
    header_fingerprint: Mapped[str] = mapped_column(String(64))
    mapping: Mapped[dict[str, Any]] = mapped_column(JSONB)
    version: Mapped[int] = mapped_column(Integer, server_default="1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ImportSession(Base):
    """One uploaded file, from arrival to commit or refusal."""

    __tablename__ = "import_sessions"
    __table_args__ = (
        Index("ix_import_sessions_clinic_id_started_at", "clinic_id", "started_at"),
        {"schema": "onboarding"},
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    clinic_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("app.clinics.id"))
    profile_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("onboarding.import_profiles.id")
    )
    filename: Mapped[str] = mapped_column(String(255))
    file_size: Mapped[int] = mapped_column(BigInteger)
    #: Identifies a byte-identical re-upload, so the same file is not imported twice.
    file_sha256: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(20))
    total_rows: Mapped[int | None] = mapped_column(Integer)
    valid_rows: Mapped[int | None] = mapped_column(Integer)
    invalid_rows: Mapped[int | None] = mapped_column(Integer)
    review_rows: Mapped[int | None] = mapped_column(Integer)
    #: Structure findings and per-column proposals: what the review screen renders.
    report: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    actor_id: Mapped[str | None] = mapped_column(String(120))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)


class StagingRow(Base):
    """One row of the file, as read and as converted, before anything is committed."""

    __tablename__ = "staging_rows"
    __table_args__ = (
        Index("ix_staging_rows_session_id_status", "session_id", "status"),
        {"schema": "onboarding"},
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    clinic_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("app.clinics.id"))
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("onboarding.import_sessions.id", ondelete="CASCADE")
    )
    entity: Mapped[str] = mapped_column(String(30))
    sheet: Mapped[str | None] = mapped_column(String(120))
    #: 1-based row number in the source file, so an error report can point a
    #: receptionist at the row to fix.
    row_number: Mapped[int] = mapped_column(Integer)
    raw: Mapped[dict[str, Any]] = mapped_column(JSONB)
    normalized: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(20))
    errors: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class TransformLogEntry(Base):
    """What one rule did to one cell (ADR-08a).

    This is the evidence that no transform was inferred from the data: every
    converted value carries the name of the rule that produced it, so a reviewer
    can ask why a value became what it became without re-running the import.
    """

    __tablename__ = "transform_log"
    __table_args__ = (
        Index("ix_transform_log_session_id_status", "session_id", "status"),
        {"schema": "onboarding"},
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    clinic_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("app.clinics.id"))
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("onboarding.import_sessions.id", ondelete="CASCADE")
    )
    row_number: Mapped[int] = mapped_column(Integer)
    #: Which sheet the cell came from. A workbook's sheets are validated one at
    #: a time and each pass clears its own previous attempt, so without this the
    #: clear took every other sheet's log with it.
    sheet: Mapped[str | None] = mapped_column(String(120))
    column_name: Mapped[str] = mapped_column(String(255))
    target_field: Mapped[str | None] = mapped_column(String(80))
    #: What the file held. A reviewer's answer never overwrites it.
    raw_value: Mapped[str | None] = mapped_column(Text)
    #: What a reviewer supplied instead, when they answered this cell. Null
    #: otherwise, so a corrected value is always distinguishable from an
    #: exported one.
    corrected_from_review: Mapped[str | None] = mapped_column(Text)
    normalized_value: Mapped[str | None] = mapped_column(Text)
    rule: Mapped[str] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(20))
    message: Mapped[str | None] = mapped_column(Text)
