"""Clinic ownership on appointment references, and one attendance escalation.

Revision ID: 0009
Revises: 0008
Create Date: 2026-10-20

Two findings from the client's QA audit, fixed where they belong: in the
database, not in application code that a second caller could forget.

**Composite foreign keys (QA finding 4).** `appointments` referenced its
patient, doctor, location and appointment type by id alone, so the database
checked that each row existed but not that it belonged to the same clinic. That
was demonstrated rather than theorised: as the runtime role scoped to clinic A,
an appointment was inserted in clinic A referencing clinic B's doctor, and
PostgreSQL accepted it. Row-level security compares the *row's own* clinic_id
against the scope; it never looks at what the row points at.

Each referenced table therefore gains a `UNIQUE (clinic_id, id)` -- redundant
given the primary key, but a composite foreign key needs a matching unique
constraint to target -- and `appointments` now references `(clinic_id, id)` on
all four. A mismatched reference becomes a foreign-key violation, which is the
one place the application cannot talk its way around.

**One attendance escalation per appointment (QA finding 10).** The sweep read
the existing escalations, then inserted for the rest. Two overlapping sweeps
could both read "none" and both insert. The partial unique index makes that
impossible rather than unlikely, and it is partial because the rule applies to
`attendance_unrecorded` only: a patient may legitimately have several
escalations of other kinds.

**What is not here, deliberately.** The expired-hold problem (QA finding 2)
cannot be fixed with a constraint predicate. `now()` is STABLE rather than
IMMUTABLE, and PostgreSQL refuses it in an index predicate -- tested, not
assumed: "functions in index predicate must be marked IMMUTABLE". So that fix
lives in the booking transaction, which reclaims an expired hold before
inserting over it.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: (table, the appointments column that points at it).
_REFERENCES: tuple[tuple[str, str], ...] = (
    ("patients", "patient_id"),
    ("doctors", "doctor_id"),
    ("locations", "location_id"),
    ("appointment_types", "appointment_type_id"),
)


def upgrade() -> None:
    for table, column in _REFERENCES:
        # The target a composite foreign key needs. Redundant against the
        # primary key as a uniqueness claim, but a foreign key can only
        # reference a unique constraint, and (clinic_id, id) is not one yet.
        op.create_unique_constraint(
            f"uq_{table}_clinic_id_id", table, ["clinic_id", "id"], schema="app"
        )
        # Replace the id-only key with one that carries the clinic. Dropped
        # first: two foreign keys on the same column would both have to pass,
        # which is harmless but misleading to read.
        op.drop_constraint(f"fk_appointments_{column}_{table}", "appointments", schema="app")
        op.create_foreign_key(
            f"fk_appointments_clinic_id_{column}_{table}",
            "appointments",
            table,
            ["clinic_id", column],
            ["clinic_id", "id"],
            source_schema="app",
            referent_schema="app",
        )

    # One attendance escalation per appointment, enforced rather than checked.
    # Partial: the rule is specific to `attendance_unrecorded`, and a patient
    # may have several escalations of other reasons.
    op.execute(
        """
        CREATE UNIQUE INDEX uq_escalations_attendance_per_appointment
        ON app.escalations (clinic_id, appointment_id)
        WHERE reason = 'attendance_unrecorded' AND appointment_id IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS app.uq_escalations_attendance_per_appointment")

    for table, column in reversed(_REFERENCES):
        op.drop_constraint(
            f"fk_appointments_clinic_id_{column}_{table}", "appointments", schema="app"
        )
        op.create_foreign_key(
            f"fk_appointments_{column}_{table}",
            "appointments",
            table,
            [column],
            ["id"],
            source_schema="app",
            referent_schema="app",
        )
        op.drop_constraint(f"uq_{table}_clinic_id_id", table, schema="app")
