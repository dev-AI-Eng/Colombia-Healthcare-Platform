"""Allow the attendance sweep's escalation reason.

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-10

M2's scope names an end-of-day attendance sweep that flags appointments with
no recorded check-in "for staff to confirm as no-show". It flags rather than
decides, because nobody pressing Check-in does not distinguish a patient who
never arrived from one who arrived while reception was busy.

So the sweep writes an escalation, and that needs a reason the CHECK will
accept. `attendance_unrecorded` is 21 characters, inside the column's existing
VARCHAR(25), so only the constraint changes and no data is rewritten.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: Mirrors `EscalationReason`, in its declaration order.
_REASONS_AFTER = (
    "no_acceptable_slot",
    "attendance_unrecorded",
    "missing_consent",
    "unrecognised_sender",
    "clinical_concern",
    "evaluator_exhausted",
    "patient_asked_for_a_human",
)

_REASONS_BEFORE = tuple(r for r in _REASONS_AFTER if r != "attendance_unrecorded")


#: Both operations take the SHORT name. Alembic applies the metadata naming
#: convention ("ck_%(table_name)s_%(constraint_name)s") to the drop as well as
#: the create, so the constraint stored as `ck_escalations_reason_valid` is
#: addressed here as `reason_valid`. Passing the full name produces
#: `ck_escalations_ck_escalations_reason_valid` and fails.
_NAME = "reason_valid"


def _replace_reason_check(values: tuple[str, ...]) -> None:
    quoted = ", ".join(f"'{value}'" for value in values)
    op.drop_constraint(_NAME, "escalations", schema="app", type_="check")
    op.create_check_constraint(_NAME, "escalations", f"reason IN ({quoted})", schema="app")


def upgrade() -> None:
    _replace_reason_check(_REASONS_AFTER)


def downgrade() -> None:
    # Rows carrying the reason this migration added would violate the narrower
    # constraint, so they go first. They are a question for staff, not patient
    # data: the appointment they point at is untouched, and the next sweep
    # raises them again.
    op.execute("DELETE FROM app.escalations WHERE reason = 'attendance_unrecorded'")
    _replace_reason_check(_REASONS_BEFORE)
