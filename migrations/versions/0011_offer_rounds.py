"""Remember the dates each reschedule round offered.

Revision ID: 0011
Revises: 0010
Create Date: 2026-11-18

A reschedule now offers five dates, and if none suits, five different ones,
then stops. Both rules need to know what was offered before: the offered
times are holds, which are deleted when released and swept when they expire,
so by the time a patient answers "ninguna" they may be gone. One row per
round keeps the dates and the moment they were offered.

Rows are written and read, never changed or deleted by the runtime role: a
round that was offered stays offered.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from src.core.config import get_settings

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")


def _runtime_role() -> str:
    role = get_settings().app_db_user
    if not _IDENTIFIER.match(role):
        raise ValueError(f"APP_DB_USER is not a plain SQL identifier: {role!r}")
    return role


def upgrade() -> None:
    op.create_table(
        "offer_rounds",
        sa.Column("id", sa.Uuid(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("clinic_id", sa.Uuid(), sa.ForeignKey("app.clinics.id"), nullable=False),
        sa.Column(
            "appointment_id", sa.Uuid(), sa.ForeignKey("app.appointments.id"), nullable=False
        ),
        sa.Column("patient_id", sa.Uuid(), sa.ForeignKey("app.patients.id"), nullable=False),
        sa.Column("offered_dates", postgresql.ARRAY(sa.Date()), nullable=False),
        sa.Column("offered_at", sa.DateTime(timezone=True), nullable=False),
        schema="app",
    )
    op.create_index(
        "ix_offer_rounds_clinic_id_appointment_id",
        "offer_rounds",
        ["clinic_id", "appointment_id"],
        schema="app",
    )

    role = _runtime_role()
    op.execute(f'GRANT SELECT, INSERT ON app.offer_rounds TO "{role}"')
    op.execute("ALTER TABLE app.offer_rounds ENABLE ROW LEVEL SECURITY")
    op.execute(
        f"""
        CREATE POLICY clinic_isolation ON app.offer_rounds
        FOR ALL TO "{role}"
        USING (clinic_id = app.current_clinic())
        WITH CHECK (clinic_id = app.current_clinic())
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS clinic_isolation ON app.offer_rounds")
    op.drop_index(
        "ix_offer_rounds_clinic_id_appointment_id", table_name="offer_rounds", schema="app"
    )
    op.drop_table("offer_rounds", schema="app")
