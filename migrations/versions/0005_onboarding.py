"""Onboarding schema and the canonical fields the clinic's own export carries.

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-23

Two concerns, kept in one revision because the staging tables are meaningless
without the columns they finally write into.

`onboarding` is a schema of its own rather than part of `app`: a staging row is
a clinic's file, not yet clinic data, and the clinic-facing API must not be able
to read it as though it were. Its tables still carry `clinic_id` and still get a
row-level policy, so one clinic can never see another's upload.

The `app` additions come from the target schema the client supplied. They are
real typed columns rather than a JSON blob so that validation, indexes and the
review API work on them the same way they work on every other field.

Document types are deliberately NOT extended with a CHECK constraint here:
Resolución 948 de 2026 moved the RIPS code list into a technical document that
MinSalud can revise without issuing a new resolution, so the list is
configuration (src/registry/models.py) rather than schema.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from src.core.config import get_settings

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")

# Every new table that holds clinic data and therefore needs a clinic policy.
_SCOPED_TABLES = ("import_sessions", "import_profiles", "staging_rows", "transform_log")


def _runtime_role() -> str:
    role = get_settings().app_db_user
    if not _IDENTIFIER.match(role):
        raise ValueError(f"APP_DB_USER is not a plain SQL identifier: {role!r}")
    return role


def _enum_check(column: str, values: Sequence[str], name: str) -> sa.CheckConstraint:
    quoted = ", ".join(f"'{value}'" for value in values)
    return sa.CheckConstraint(f"{column} IN ({quoted})", name=name)


def upgrade() -> None:
    role = _runtime_role()

    # ------------------------------------------------------------ app additions
    op.create_table(
        "specialties",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("clinic_id", sa.Uuid(), sa.ForeignKey("app.clinics.id"), nullable=False),
        # The clinic's own code for the specialty ("E01"), used to resolve
        # references inside the uploaded file and to match on re-import.
        sa.Column("external_ref", sa.String(64), nullable=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        schema="app",
    )
    op.create_index(
        "uq_specialties_clinic_id_external_ref",
        "specialties",
        ["clinic_id", "external_ref"],
        unique=True,
        postgresql_where=sa.text("external_ref IS NOT NULL"),
        schema="app",
    )

    op.add_column("patients", sa.Column("eps", sa.String(120), nullable=True), schema="app")
    op.add_column(
        "patients", sa.Column("secondary_contact_name", sa.String(200), nullable=True), schema="app"
    )
    # Encrypted like every other direct identifier (src/core/crypto.py).
    op.add_column(
        "patients",
        sa.Column("secondary_contact_phone", sa.LargeBinary(), nullable=True),
        schema="app",
    )
    op.add_column(
        "patients", sa.Column("telegram_chat_id", sa.String(64), nullable=True), schema="app"
    )
    op.add_column("patients", sa.Column("external_ref", sa.String(64), nullable=True), schema="app")

    op.add_column("doctors", sa.Column("office_number", sa.String(32), nullable=True), schema="app")
    op.add_column("doctors", sa.Column("external_ref", sa.String(64), nullable=True), schema="app")
    op.add_column(
        "doctors",
        sa.Column("specialty_id", sa.Uuid(), sa.ForeignKey("app.specialties.id"), nullable=True),
        schema="app",
    )

    for table in ("patients", "doctors"):
        op.create_index(
            f"uq_{table}_clinic_id_external_ref",
            table,
            ["clinic_id", "external_ref"],
            unique=True,
            postgresql_where=sa.text("external_ref IS NOT NULL"),
            schema="app",
        )

    op.add_column(
        "appointments", sa.Column("consultation_type", sa.String(40), nullable=True), schema="app"
    )
    op.add_column("appointments", sa.Column("source", sa.String(60), nullable=True), schema="app")
    op.add_column(
        "appointments",
        sa.Column("cancellation_reason", sa.String(200), nullable=True),
        schema="app",
    )
    op.add_column(
        "appointments",
        sa.Column(
            "rescheduled_from_appointment_id",
            sa.Uuid(),
            sa.ForeignKey("app.appointments.id"),
            nullable=True,
        ),
        schema="app",
    )
    op.add_column(
        "appointments", sa.Column("preferred_channel", sa.String(30), nullable=True), schema="app"
    )
    op.add_column(
        "appointments",
        sa.Column("reminder_48h_sent", sa.Boolean(), server_default=sa.false(), nullable=False),
        schema="app",
    )
    op.add_column(
        "appointments",
        sa.Column("reminder_24h_sent", sa.Boolean(), server_default=sa.false(), nullable=False),
        schema="app",
    )
    op.add_column(
        "appointments", sa.Column("external_ref", sa.String(64), nullable=True), schema="app"
    )
    op.create_index(
        "uq_appointments_clinic_id_external_ref",
        "appointments",
        ["clinic_id", "external_ref"],
        unique=True,
        postgresql_where=sa.text("external_ref IS NOT NULL"),
        schema="app",
    )

    # --------------------------------------------------------- onboarding schema
    op.execute("CREATE SCHEMA IF NOT EXISTS onboarding")

    op.create_table(
        "import_profiles",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("clinic_id", sa.Uuid(), sa.ForeignKey("app.clinics.id"), nullable=False),
        sa.Column("name", sa.String(120), nullable=False),
        # Sorted, normalized header names hashed: identifies "a file shaped like
        # this one" so a repeat upload reuses the confirmed mapping (PDF exit
        # criterion) without calling a model at all.
        sa.Column("header_fingerprint", sa.String(64), nullable=False),
        # ADR-08a: the mapping only. Never a rule derived from sampled values.
        sa.Column("mapping", postgresql.JSONB(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        schema="onboarding",
    )
    op.create_index(
        "uq_import_profiles_clinic_fingerprint",
        "import_profiles",
        ["clinic_id", "header_fingerprint"],
        unique=True,
        schema="onboarding",
    )

    op.create_table(
        "import_sessions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("clinic_id", sa.Uuid(), sa.ForeignKey("app.clinics.id"), nullable=False),
        sa.Column(
            "profile_id",
            sa.Uuid(),
            sa.ForeignKey("onboarding.import_profiles.id"),
            nullable=True,
        ),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("file_size", sa.BigInteger(), nullable=False),
        # Identifies a byte-identical re-upload, so the same file is not
        # imported twice by accident.
        sa.Column("file_sha256", sa.String(64), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("total_rows", sa.Integer(), nullable=True),
        sa.Column("valid_rows", sa.Integer(), nullable=True),
        sa.Column("invalid_rows", sa.Integer(), nullable=True),
        sa.Column("review_rows", sa.Integer(), nullable=True),
        # Structure findings and per-column proposals; what the review screen renders.
        sa.Column("report", postgresql.JSONB(), nullable=True),
        sa.Column("actor_id", sa.String(120), nullable=True),
        sa.Column(
            "started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        _enum_check(
            "status",
            # `needs_review` is the state a blocked import rests in: validated,
            # but with rows or questions a person must resolve before it can be
            # committed. It is distinct from `failed`, which is unreadable.
            [
                "received",
                "analyzed",
                "mapped",
                "validated",
                "needs_review",
                "committed",
                "failed",
            ],
            "import_session_status_valid",
        ),
        schema="onboarding",
    )
    op.create_index(
        "ix_import_sessions_clinic_id_started_at",
        "import_sessions",
        ["clinic_id", "started_at"],
        schema="onboarding",
    )

    op.create_table(
        "staging_rows",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("clinic_id", sa.Uuid(), sa.ForeignKey("app.clinics.id"), nullable=False),
        sa.Column(
            "session_id",
            sa.Uuid(),
            sa.ForeignKey("onboarding.import_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # Which canonical entity this row becomes.
        sa.Column("entity", sa.String(30), nullable=False),
        sa.Column("sheet", sa.String(120), nullable=True),
        # 1-based row number in the source file, so an error report can point a
        # receptionist at the row to fix.
        sa.Column("row_number", sa.Integer(), nullable=False),
        sa.Column("raw", postgresql.JSONB(), nullable=False),
        sa.Column("normalized", postgresql.JSONB(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("errors", postgresql.JSONB(), nullable=True),
        _enum_check("status", ["valid", "invalid", "review"], "staging_row_status_valid"),
        schema="onboarding",
    )
    op.create_index(
        "ix_staging_rows_session_id_status",
        "staging_rows",
        ["session_id", "status"],
        schema="onboarding",
    )

    # ADR-08a: one entry per row per column, recording what the deterministic
    # normalizer did. This is the evidence that no transform was inferred from a
    # sample: every value carries the rule that produced it.
    op.create_table(
        "transform_log",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("clinic_id", sa.Uuid(), sa.ForeignKey("app.clinics.id"), nullable=False),
        sa.Column(
            "session_id",
            sa.Uuid(),
            sa.ForeignKey("onboarding.import_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("row_number", sa.Integer(), nullable=False),
        sa.Column("column_name", sa.String(255), nullable=False),
        sa.Column("target_field", sa.String(80), nullable=True),
        sa.Column("raw_value", sa.Text(), nullable=True),
        sa.Column("normalized_value", sa.Text(), nullable=True),
        sa.Column("rule", sa.String(80), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("message", sa.Text(), nullable=True),
        _enum_check("status", ["valid", "invalid", "review"], "transform_log_status_valid"),
        schema="onboarding",
    )
    op.create_index(
        "ix_transform_log_session_id_status",
        "transform_log",
        ["session_id", "status"],
        schema="onboarding",
    )

    # ------------------------------------------------------------------ security
    # `app.specialties` is new in this revision. Migration 0001 granted ALL
    # TABLES IN SCHEMA app, which applies only to the tables that existed then,
    # so a table added later needs its own grant or the runtime role cannot
    # read it at all.
    op.execute(f'GRANT SELECT, INSERT, UPDATE, DELETE ON app.specialties TO "{role}"')

    op.execute(f'GRANT USAGE ON SCHEMA onboarding TO "{role}"')
    op.execute(
        f'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA onboarding TO "{role}"'
    )
    op.execute(f'GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA onboarding TO "{role}"')

    for table in _SCOPED_TABLES:
        op.execute(f"ALTER TABLE onboarding.{table} ENABLE ROW LEVEL SECURITY")
        op.execute(
            f"""
            CREATE POLICY clinic_isolation ON onboarding.{table}
            FOR ALL TO "{role}"
            USING (clinic_id = app.current_clinic())
            WITH CHECK (clinic_id = app.current_clinic())
            """
        )

    # app.specialties is clinic data like every other table in that schema.
    op.execute("ALTER TABLE app.specialties ENABLE ROW LEVEL SECURITY")
    op.execute(
        f"""
        CREATE POLICY clinic_isolation ON app.specialties
        FOR ALL TO "{role}"
        USING (clinic_id = app.current_clinic())
        WITH CHECK (clinic_id = app.current_clinic())
        """
    )


def downgrade() -> None:
    role = _runtime_role()

    op.execute("DROP POLICY IF EXISTS clinic_isolation ON app.specialties")
    for table in _SCOPED_TABLES:
        op.execute(f"DROP POLICY IF EXISTS clinic_isolation ON onboarding.{table}")
    op.execute(f'REVOKE ALL ON app.specialties FROM "{role}"')
    op.execute(f'REVOKE ALL ON ALL TABLES IN SCHEMA onboarding FROM "{role}"')
    op.execute(f'REVOKE ALL ON SCHEMA onboarding FROM "{role}"')

    op.drop_table("transform_log", schema="onboarding")
    op.drop_table("staging_rows", schema="onboarding")
    op.drop_table("import_sessions", schema="onboarding")
    op.drop_table("import_profiles", schema="onboarding")
    op.execute("DROP SCHEMA IF EXISTS onboarding")

    op.drop_index("uq_appointments_clinic_id_external_ref", "appointments", schema="app")
    for column in (
        "consultation_type",
        "source",
        "cancellation_reason",
        "rescheduled_from_appointment_id",
        "preferred_channel",
        "reminder_48h_sent",
        "reminder_24h_sent",
        "external_ref",
    ):
        op.drop_column("appointments", column, schema="app")

    for table in ("patients", "doctors"):
        op.drop_index(f"uq_{table}_clinic_id_external_ref", table, schema="app")
    op.drop_column("doctors", "specialty_id", schema="app")
    op.drop_column("doctors", "external_ref", schema="app")
    op.drop_column("doctors", "office_number", schema="app")
    for column in (
        "eps",
        "secondary_contact_name",
        "secondary_contact_phone",
        "telegram_chat_id",
        "external_ref",
    ):
        op.drop_column("patients", column, schema="app")

    op.drop_index("uq_specialties_clinic_id_external_ref", "specialties", schema="app")
    op.drop_table("specialties", schema="app")
