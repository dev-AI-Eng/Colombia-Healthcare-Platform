"""Integration fixtures.

Once per run, the test database is dropped, recreated and provisioned through
the same code path as `main.py`: runtime role, migrations, checkpoint schema.
If PostgreSQL is unreachable, every integration test is skipped with the reason.

Before each test, all application and audit rows are removed with TRUNCATE, run
as the owner account. TRUNCATE does not fire the audit table's row-level
append-only trigger.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator

import psycopg
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src import bootstrap
from src.api.app import create_app
from src.audit.context import ActorKind, AuditContext, audit_context
from src.conversation.checkpointer import ensure_checkpoint_schema
from src.core.config import Settings, get_settings

_TABLES = (
    # Onboarding first: its rows reference clinics and are cascaded from sessions.
    "onboarding.transform_log",
    "onboarding.staging_rows",
    "onboarding.import_sessions",
    "onboarding.import_profiles",
    "app.appointments",
    "app.availability_exceptions",
    "app.availability_rules",
    "app.appointment_types",
    "app.consents",
    "app.phone_bindings",
    "app.patients",
    "app.doctors",
    "app.specialties",
    "app.locations",
    "app.clinics",
    "audit.access_log",
    "audit.chain_anchor",
)


@pytest.fixture(scope="session")
def settings() -> Settings:
    return get_settings()


@pytest.fixture(scope="session", autouse=True)
def provisioned_database(settings: Settings) -> None:
    try:
        bootstrap.reset_database(settings)
    except bootstrap.ProvisioningError as exc:
        pytest.skip(f"PostgreSQL unavailable for integration tests: {exc}")
    bootstrap.ensure_runtime_role(settings)
    bootstrap.run_migrations()
    asyncio.run(ensure_checkpoint_schema(settings))


@pytest.fixture(autouse=True)
def empty_tables(provisioned_database: None, settings: Settings) -> None:
    with psycopg.connect(settings.conninfo(admin=True), autocommit=True) as conn:
        conn.execute(f"TRUNCATE {', '.join(_TABLES)} RESTART IDENTITY")


@pytest.fixture
def owner_connection(settings: Settings) -> Iterator[psycopg.Connection]:
    with psycopg.connect(settings.conninfo(admin=True), autocommit=True) as conn:
        yield conn


@pytest.fixture
async def session(settings: Settings) -> AsyncIterator[AsyncSession]:
    """A session connected as the runtime role, as the application connects."""
    engine = create_async_engine(settings.app_database_url, poolclass=NullPool)
    maker = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    try:
        async with maker() as db_session:
            yield db_session
    finally:
        await engine.dispose()


@pytest.fixture
def staff_context() -> Iterator[AuditContext]:
    context = AuditContext(
        actor_kind=ActorKind.STAFF, actor_id="tester", purpose="integration_test"
    )
    with audit_context(context):
        yield context


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app()) as test_client:
        yield test_client
