"""Contact details for doctors.

Revision ID: 0010
Revises: 0009
Create Date: 2026-10-28

The client asked for a doctor's phone, a second phone and an email on the
doctors screen, so that staff can reach a doctor who is running late or has to
be moved. `doctors` carried a name, a specialty and an office number and nothing
else.

All three are encrypted at rest like every other direct identifier and are
nullable: a clinic that does not hold a number for a doctor is not wrong, and
the importer has never had a column to put one in. There is no blind index
because nothing looks a doctor up by phone.

Numbered 0010 because 0009 is `0009_qa_integrity`. Another branch also wrote a
0009 and the two cannot coexist -- `alembic upgrade head` refuses with
"Multiple head revisions" -- so this one chains after it, and the branch that
merges later renumbers.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | None = None
depends_on: str | None = None

_COLUMNS = ("phone", "secondary_phone", "email")


def upgrade() -> None:
    for name in _COLUMNS:
        op.add_column(
            "doctors",
            sa.Column(name, sa.LargeBinary(), nullable=True),
            schema="app",
        )


def downgrade() -> None:
    for name in reversed(_COLUMNS):
        op.drop_column("doctors", name, schema="app")
