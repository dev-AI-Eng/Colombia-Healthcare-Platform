"""The queue of things a human has to deal with.

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-09

Scheduling can reach a state it must not resolve on its own: a patient needs
an appointment and no slot exists that they can accept. The scope calls for a
fallback in that case, and a fallback that only sends a message is a dead end
-- nobody is told, and nothing records that the patient is still waiting. This
table is what staff work from.

It is created in M2 because M2 is the first milestone that can reach such a
state, but the reason enum covers M3's cases too (missing consent, an
unrecognised sender, a clinical concern, the evaluator giving up, a patient
asking for a person). One queue, one screen, one place to look.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from src.core.config import get_settings

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")

#: Mirrors `EscalationReason`. Stored as VARCHAR with a CHECK, per the project's
#: enum convention: a database enum type cannot have a value removed.
_REASONS = (
    "no_acceptable_slot",
    "missing_consent",
    "unrecognised_sender",
    "clinical_concern",
    "evaluator_exhausted",
    "patient_asked_for_a_human",
)
_STATUSES = ("open", "resolved")


def _runtime_role() -> str:
    role = get_settings().app_db_user
    if not _IDENTIFIER.match(role):
        raise ValueError(f"APP_DB_USER is not a plain SQL identifier: {role!r}")
    return role


def _enum_check(column: str, values: Sequence[str], name: str) -> sa.CheckConstraint:
    quoted = ", ".join(f"'{value}'" for value in values)
    return sa.CheckConstraint(f"{column} IN ({quoted})", name=name)


def upgrade() -> None:
    op.create_table(
        "escalations",
        sa.Column("id", sa.Uuid(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("clinic_id", sa.Uuid(), sa.ForeignKey("app.clinics.id"), nullable=False),
        # Null when the sender could not be identified, which is itself a reason
        # to escalate: somebody messaged the clinic and nobody knows who.
        sa.Column("patient_id", sa.Uuid(), sa.ForeignKey("app.patients.id"), nullable=True),
        sa.Column("appointment_id", sa.Uuid(), sa.ForeignKey("app.appointments.id"), nullable=True),
        sa.Column("reason", sa.String(25), nullable=False),
        sa.Column("status", sa.String(8), nullable=False, server_default="open"),
        # Spanish: a receptionist reads this, not a developer.
        sa.Column("detail_es", sa.String(500), nullable=False),
        sa.Column(
            "context",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        _enum_check("reason", _REASONS, "reason_valid"),
        _enum_check("status", _STATUSES, "status_valid"),
        schema="app",
    )
    # The queue is read as "what is open for this clinic", so that is the index.
    op.create_index(
        "ix_escalations_clinic_id_status", "escalations", ["clinic_id", "status"], schema="app"
    )

    role = _runtime_role()
    op.execute(f'GRANT SELECT, INSERT, UPDATE ON app.escalations TO "{role}"')
    # No DELETE: an escalation is resolved, never erased. A queue that can be
    # emptied silently is a queue that cannot be audited.
    op.execute("ALTER TABLE app.escalations ENABLE ROW LEVEL SECURITY")
    op.execute(
        f"""
        CREATE POLICY clinic_isolation ON app.escalations
        FOR ALL TO "{role}"
        USING (clinic_id = app.current_clinic())
        WITH CHECK (clinic_id = app.current_clinic())
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS clinic_isolation ON app.escalations")
    op.drop_index("ix_escalations_clinic_id_status", table_name="escalations", schema="app")
    op.drop_table("escalations", schema="app")
