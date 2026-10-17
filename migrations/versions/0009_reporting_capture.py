"""M4 reporting capture: appointment events, actual times, patient zone.

Revision ID: 0009
Revises: 0008
Create Date: 2026-10-17

ADR-14: management reporting is computed from an append-only
`appointment_events` table, written in the same transaction as each status
change. Current status alone loses history -- an appointment rescheduled twice
and then attended reads as `completed`, and no report built on that can say it
was ever rescheduled.

Three additions, one concern each:

**`app.appointment_events`** is append-only in the same sense as the audit log:
the runtime role gets INSERT and SELECT and nothing else. A report whose
underlying events can be edited is a report that proves nothing. `doctor_id`
and `location_id` are denormalised onto the event deliberately -- a doctor who
moves to another sede must not retroactively move their own history.

**`checked_in_at` / `started_at` / `ended_at`** on appointments. Rev3 is
explicit that "the system cannot detect on its own when a consultation starts
or ends", so each is null until a person presses a button. Reports show
coverage alongside any number derived from them: punctuality computed from a
tenth of the rows misleads more than no number at all.

**`municipality_divipola` / `neighborhood`** on patients, for the geographic
no-show view. The municipality is the five-digit DANE DIVIPOLA code rather than
a name, because "Bogotá D.C." has several spellings and a code has one.
Collecting these must be covered by the clinic's privacy notice and the
patient's authorisation (Ley 1581, arts. 4(b) and 12) -- a client-side
dependency, not something this migration can satisfy.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from src.core.config import get_settings

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: Mirrors `AppointmentEventKind`, in its declaration order.
_KINDS = (
    "offered",
    "confirmed",
    "rescheduled",
    "cancelled",
    "checked_in",
    "started",
    "completed",
    "no_show",
)


def _runtime_role() -> str:
    return get_settings().app_db_user


def _kind_check() -> sa.CheckConstraint:
    values = ", ".join(f"'{kind}'" for kind in _KINDS)
    return sa.CheckConstraint(f"kind IN ({values})", name="kind_valid")


def upgrade() -> None:
    # ----------------------------------------------- actual consultation times
    for column in ("checked_in_at", "started_at", "ended_at"):
        op.add_column(
            "appointments",
            sa.Column(column, sa.DateTime(timezone=True), nullable=True),
            schema="app",
        )

    # ------------------------------------------------------- patient zone
    op.add_column(
        "patients",
        sa.Column("municipality_divipola", sa.String(length=5), nullable=True),
        schema="app",
    )
    op.add_column(
        "patients", sa.Column("neighborhood", sa.String(length=120), nullable=True), schema="app"
    )

    # ------------------------------------------------------ appointment events
    op.create_table(
        "appointment_events",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("clinic_id", sa.Uuid(), sa.ForeignKey("app.clinics.id"), nullable=False),
        sa.Column(
            "appointment_id", sa.Uuid(), sa.ForeignKey("app.appointments.id"), nullable=False
        ),
        sa.Column("doctor_id", sa.Uuid(), sa.ForeignKey("app.doctors.id"), nullable=False),
        sa.Column("location_id", sa.Uuid(), sa.ForeignKey("app.locations.id"), nullable=False),
        # VARCHAR(11), matching what `string_enum` derives from the longest
        # value ("rescheduled"). Written explicitly rather than guessed:
        # `test_migrations` compares this against the model and a wider column
        # is a real difference, not a harmless one.
        sa.Column("kind", sa.String(length=11), nullable=False),
        sa.Column("actor", sa.String(length=16), nullable=False),
        sa.Column("reason", sa.String(length=200), nullable=True),
        sa.Column(
            "occurred_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        _kind_check(),
        schema="app",
    )
    # The funnel is read as "everything for this clinic in this date range",
    # and the drill-down as "everything for this appointment".
    op.create_index(
        "ix_appointment_events_clinic_id_occurred_at",
        "appointment_events",
        ["clinic_id", "occurred_at"],
        schema="app",
    )
    op.create_index(
        "ix_appointment_events_appointment_id",
        "appointment_events",
        ["appointment_id"],
        schema="app",
    )

    role = _runtime_role()
    # INSERT and SELECT only. No UPDATE, no DELETE: the same privilege barrier
    # the audit log has, for the same reason -- a funnel computed from rows the
    # application could rewrite is not evidence of anything.
    op.execute(f'GRANT SELECT, INSERT ON app.appointment_events TO "{role}"')
    op.execute(f'GRANT USAGE, SELECT ON SEQUENCE app.appointment_events_id_seq TO "{role}"')
    op.execute("ALTER TABLE app.appointment_events ENABLE ROW LEVEL SECURITY")
    op.execute(
        f"""
        CREATE POLICY clinic_isolation ON app.appointment_events
        FOR ALL TO "{role}"
        USING (clinic_id = app.current_clinic())
        WITH CHECK (clinic_id = app.current_clinic())
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS clinic_isolation ON app.appointment_events")
    op.drop_index(
        "ix_appointment_events_appointment_id", table_name="appointment_events", schema="app"
    )
    op.drop_index(
        "ix_appointment_events_clinic_id_occurred_at",
        table_name="appointment_events",
        schema="app",
    )
    op.drop_table("appointment_events", schema="app")

    op.drop_column("patients", "neighborhood", schema="app")
    op.drop_column("patients", "municipality_divipola", schema="app")
    for column in ("ended_at", "started_at", "checked_in_at"):
        op.drop_column("appointments", column, schema="app")
