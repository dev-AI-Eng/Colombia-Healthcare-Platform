"""The hand-written migration against the models, its downgrade, and runtime-role limits.

Alembic's comparison covers tables, columns, types, nullability, indexes, unique
constraints and foreign keys. It does not compare CHECK or exclusion constraints,
so those are compared separately by name. Constraint definitions and server
defaults are not compared.
"""

from __future__ import annotations

import pathlib
import re
from typing import Any

import psycopg
import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Connection
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlalchemy.schema import CreateTable

import src.audit.models
import src.identity.models
import src.registry.models
import src.scheduling.models  # noqa: F401
from src import bootstrap
from src.core.config import Settings
from src.core.db import Base

# Every schema whose tables are declared as models and created by a migration.
# `conversation` is absent on purpose: LangGraph owns those tables and creates
# them itself, so comparing them against our models would always differ.
_SCHEMAS = {"app", "audit", "onboarding"}


def _include_name(name: str | None, type_: str, parent_names: Any) -> bool:
    if type_ == "schema":
        return name in _SCHEMAS
    return True


def _diff(connection: Connection) -> list[Any]:
    context = MigrationContext.configure(
        connection,
        opts={
            "include_schemas": True,
            "include_name": _include_name,
            "compare_type": True,
            "compare_server_default": False,
        },
    )
    return list(compare_metadata(context, Base.metadata))


async def test_migrated_schema_matches_models(settings: Settings) -> None:
    engine = create_async_engine(settings.admin_database_url, poolclass=NullPool)
    try:
        async with engine.connect() as connection:
            differences = await connection.run_sync(_diff)
    finally:
        await engine.dispose()
    assert differences == []


def _model_check_and_exclusion_constraints() -> set[tuple[str, str]]:
    names: set[tuple[str, str]] = set()
    for table in Base.metadata.sorted_tables:
        ddl = str(CreateTable(table).compile(dialect=postgresql.dialect()))
        for name in re.findall(r"CONSTRAINT (\w+) (?:CHECK|EXCLUDE)", ddl):
            names.add((f"{table.schema}.{table.name}", name))
    return names


def test_check_and_exclusion_constraints_match_models(settings: Settings) -> None:
    expected = _model_check_and_exclusion_constraints()
    with psycopg.connect(settings.conninfo(admin=True)) as conn:
        rows = conn.execute(
            "SELECT conrelid::regclass::text, conname FROM pg_constraint"
            " WHERE contype IN ('c', 'x') AND connamespace::regnamespace::text IN ('app', 'audit')"
        ).fetchall()
    assert len(expected) >= 20, "constraint extraction from the models found too few"
    assert set(rows) == expected


def test_downgrade_then_upgrade_round_trips(settings: Settings) -> None:
    config = Config("alembic.ini")
    command.downgrade(config, "base")
    with psycopg.connect(settings.conninfo(admin=True), autocommit=True) as conn:
        remaining = conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema IN ('app', 'audit')"
        ).fetchone()
    assert remaining == (0,)
    bootstrap.run_migrations()


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE audit.access_log SET purpose = 'x'",
        "DELETE FROM audit.access_log",
        "TRUNCATE audit.access_log",
    ],
)
def test_runtime_role_cannot_rewrite_the_audit_log(settings: Settings, statement: str) -> None:
    """The runtime role lacks the privileges to rewrite the audit log."""
    with (
        psycopg.connect(settings.conninfo(admin=False), autocommit=True) as conn,
        pytest.raises(psycopg.errors.InsufficientPrivilege),
    ):
        conn.execute(statement)


def test_owner_cannot_update_audit_rows_either(owner_connection: psycopg.Connection) -> None:
    """The trigger rejects UPDATE even from the owner account."""
    owner_connection.execute(
        "INSERT INTO audit.access_log (occurred_at, actor_kind, action, resource, purpose, row_hash)"
        " VALUES (now(), 'system', 'read', 'patients', 'test', '\\x00')"
    )
    with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
        owner_connection.execute("UPDATE audit.access_log SET purpose = 'tampered'")


def test_runtime_role_cannot_create_tables(settings: Settings) -> None:
    with (
        psycopg.connect(settings.conninfo(admin=False), autocommit=True) as conn,
        pytest.raises(psycopg.errors.InsufficientPrivilege),
    ):
        conn.execute("CREATE TABLE app.intruder (id int)")


def test_the_migration_history_is_a_single_unambiguous_chain() -> None:
    """One head, and no revision id used twice.

    Two branches that each add a migration numbered 0009 produce two files
    declaring `revision = "0009"` with `down_revision = "0008"`. Measured on
    this repository's Alembic (1.20.0), with M2's `0009_qa_integrity` and M4's
    `0009_reporting_capture` in one directory:

        $ alembic heads
        UserWarning: Revision 0009 is present more than once
        0009 (head)
        0009 (head)

        $ alembic upgrade head
        FAILED: Multiple head revisions are present for given argument 'head'

    So the duplicate is a deploy blocker rather than a silent skip: `main.py`
    and CI both run `upgrade head` and stop. It is caught today, but badly --
    provisioning is a session fixture, so every test in this file errors at
    setup and none of them names the duplicate. This test names it.

    Both assertions are kept deliberately. `get_heads()` is what catches the
    duplicate-id case, because the two files collapse into one id that is
    reported twice. Reading the ids out of the files is what names *which*
    files collided, which is the part a developer needs. Asking Alembic which
    ids repeat cannot answer that: `walk_revisions()` yields the duplicated id
    once, since the second file has already overwritten the first in its map.
    """
    script = ScriptDirectory.from_config(Config("alembic.ini"))

    # Read the ids out of the files rather than out of Alembic's revision map.
    # `walk_revisions()` yields a duplicated id once, not twice -- the second
    # file has already overwritten the first in the map -- so asking Alembic
    # which ids repeat cannot answer the question. Measured: with two 0009
    # files present, walking the revisions reported no duplicates at all while
    # `get_heads()` returned `['0009', '0009']`.
    versions = pathlib.Path(script.versions).glob("*.py")
    declared: dict[str, list[str]] = {}
    pattern = re.compile(r"^revision(?::\s*str)?\s*=\s*[\"'](?P<id>[^\"']+)[\"']", re.MULTILINE)
    for path in versions:
        found = pattern.search(path.read_text(encoding="utf-8"))
        assert found is not None, f"{path.name} declares no `revision`"
        declared.setdefault(found.group("id"), []).append(path.name)
    duplicates = {rev: files for rev, files in sorted(declared.items()) if len(files) > 1}
    assert not duplicates, (
        f"these revision ids are declared by more than one migration: {duplicates}. "
        "Alembic will apply only one of them and warn rather than fail. Renumber "
        "the newer migration and point its down_revision at the older one."
    )

    heads = script.get_heads()
    assert len(heads) == 1, (
        f"the migration history has {len(heads)} heads ({heads}); it must be a single "
        "chain so that one upgrade order is defined"
    )
