"""Alembic environment.

Migrations run as the owner account (`postgres_*` settings), never as the
runtime role. The connection URL comes from settings, so no credential is
written to alembic.ini.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig
from typing import Any

from alembic import context
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

# Importing the model modules registers every table on Base.metadata.
import src.audit.models
import src.identity.models
import src.onboarding.models
import src.registry.models
import src.scheduling.models  # noqa: F401
from src.core.config import get_settings
from src.core.db import Base, configure_event_loop_policy

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

_OPTIONS: dict[str, Any] = {
    "target_metadata": Base.metadata,
    "include_schemas": True,
    "compare_type": True,
}


def run_migrations_offline() -> None:
    url = get_settings().admin_database_url.render_as_string(hide_password=False)
    context.configure(url=url, literal_binds=True, **_OPTIONS)
    with context.begin_transaction():
        context.run_migrations()


def _run_with_connection(connection: Connection) -> None:
    context.configure(connection=connection, **_OPTIONS)
    with context.begin_transaction():
        context.run_migrations()


async def _run_async() -> None:
    engine = create_async_engine(get_settings().admin_database_url, poolclass=NullPool)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(_run_with_connection)
    finally:
        await engine.dispose()


def run_migrations_online() -> None:
    configure_event_loop_policy()
    asyncio.run(_run_async())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
