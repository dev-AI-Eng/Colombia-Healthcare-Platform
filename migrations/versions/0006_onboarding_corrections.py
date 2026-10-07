"""Bring a database already at 0005 up to what 0005 should have created.

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-03

Four changes were made by editing 0005 in place after it had already run
somewhere. Alembic stamps a database by revision, not by content, so a database
stamped 0005 never receives them: it is reported as current and silently lacks
the column the code selects. The symptom is every `validate` answering HTTP 500
with `UndefinedColumn: transform_log.sheet`, with nothing at startup explaining
why. This revision applies the four, and `test_migrations.py` compares the
resulting schema against the models so the pair cannot drift again.

Each step is written to be safe on a database that already has the change. The
CHECK value and the grant are created by 0005 on disk, so a fresh install
applies them twice; the two columns are created only here, since aa79672 removed
them from 0005. Either way the second application must not fail, which is why
every step is `IF NOT EXISTS` or drops the constraint before recreating it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from alembic import op

from src.core.config import get_settings

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")

#: The status values an import session may hold. `needs_review` is the state a
#: blocked import rests in: validated, but with rows or questions a person must
#: resolve before it can be committed. It is distinct from `failed`, which means
#: the file could not be read at all.
_SESSION_STATUSES = (
    "received",
    "analyzed",
    "mapped",
    "validated",
    "needs_review",
    "committed",
    "failed",
)


def _runtime_role() -> str:
    role = get_settings().app_db_user
    if not _IDENTIFIER.match(role):
        raise ValueError(f"APP_DB_USER is not a plain SQL identifier: {role!r}")
    return role


def upgrade() -> None:
    role = _runtime_role()

    # 1 and 2. The transform log is the ADR-08a evidence of which rule produced
    # every cell. `sheet` is what makes the per-sheet clear in `replace_staging`
    # correct: without it, revalidating one sheet of a workbook cleared the log
    # for all of them. `corrected_from_review` holds what a reviewer supplied, so
    # `raw_value` can keep showing what the clinic's file actually contained.
    op.execute("ALTER TABLE onboarding.transform_log ADD COLUMN IF NOT EXISTS sheet VARCHAR(120)")
    op.execute(
        "ALTER TABLE onboarding.transform_log ADD COLUMN IF NOT EXISTS corrected_from_review TEXT"
    )

    # 3. A CHECK constraint cannot be altered in place, so it is replaced. An
    # import blocked on an unanswered question could not be stored without
    # `needs_review`, which is the state the review screen reads. The stored name
    # carries the convention's `ck_<table>_` prefix, not the bare name 0005
    # passes to `_enum_check`.
    op.execute(
        "ALTER TABLE onboarding.import_sessions DROP CONSTRAINT IF EXISTS "
        "ck_import_sessions_import_session_status_valid"
    )
    allowed = ", ".join(f"'{status}'" for status in _SESSION_STATUSES)
    op.create_check_constraint(
        "import_session_status_valid",
        "import_sessions",
        f"status IN ({allowed})",
        schema="onboarding",
    )

    # 4. Migration 0001 granted ALL TABLES IN SCHEMA app, which applies only to
    # the tables existing at that moment. `app.specialties` arrived in 0005, so
    # without its own grant the runtime role cannot read it at all.
    op.execute(f'GRANT SELECT, INSERT, UPDATE, DELETE ON app.specialties TO "{role}"')


def downgrade() -> None:
    """Drop only the two columns. The constraint and the grant belong to 0005.

    Reverting those as well would be wrong twice over. 0005 on disk creates the
    `needs_review` value and the specialties grant, so undoing them here leaves
    revision 0005 in a state a fresh `upgrade 0005` never produces -- the schema
    would differ depending on which direction it was reached from. And narrowing
    the CHECK fails outright on any database holding a blocked import, because
    those rows carry exactly the status being removed.
    """
    op.drop_column("transform_log", "corrected_from_review", schema="onboarding")
    op.drop_column("transform_log", "sheet", schema="onboarding")
