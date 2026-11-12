"""Synthetic seeding, the LangGraph checkpointer, readiness, and runtime-role provisioning."""

from __future__ import annotations

import operator
from datetime import date, datetime
from typing import Annotated, TypedDict

import psycopg
from fastapi.testclient import TestClient
from langgraph.graph import END, START, StateGraph
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import bootstrap
from src.audit.context import AuditContext
from src.audit.models import AccessLogEntry
from src.audit.service import verify_chain
from src.conversation.checkpointer import (
    CHECKPOINT_SCHEMA,
    open_checkpointer,
    sweep_expired_threads,
)
from src.core.config import Settings
from src.core.tenancy import ClinicScope, apply_clinic_scope
from src.core.timezones import BOGOTA
from src.devdata.seed import (
    APPOINTMENTS_PER_DOCTOR,
    SHARED_HANDSET_PATIENTS,
    seed_synthetic_data,
)
from src.identity.models import PhoneBinding
from src.registry.models import Patient
from src.scheduling.models import Appointment


async def test_seed_creates_expected_records_and_audits_them(
    session: AsyncSession, staff_context: AuditContext
) -> None:
    report = await seed_synthetic_data(session, patients=20, doctors=3, today=date(2026, 10, 1))
    await session.commit()

    assert report is not None
    # The commit ended the transaction the seeder scoped; the reads below need it back.
    await apply_clinic_scope(session, ClinicScope(clinic_id=report.clinic_id))
    assert (report.patients, report.doctors) == (20, 3)
    assert report.appointments == 3 * APPOINTMENTS_PER_DOCTOR
    assert len((await session.scalars(select(Patient))).all()) == 20
    assert len((await session.scalars(select(PhoneBinding))).all()) == 20 + SHARED_HANDSET_PATIENTS

    created = (
        await session.scalars(select(AccessLogEntry).where(AccessLogEntry.action == "create"))
    ).all()
    assert len(created) == 20 + (20 + SHARED_HANDSET_PATIENTS) + 20 + report.appointments
    assert (await verify_chain(session)).intact

    # 2026-10-01 is a Thursday, so the first appointment day is Friday 2 October.
    first = min(a.during.lower for a in (await session.scalars(select(Appointment))).all())
    assert first == datetime(2026, 10, 2, 8, 0, tzinfo=BOGOTA)


async def test_seed_refuses_a_non_empty_database(
    session: AsyncSession, staff_context: AuditContext
) -> None:
    assert await seed_synthetic_data(session, patients=2, doctors=1) is not None
    await session.commit()
    assert await seed_synthetic_data(session, patients=2, doctors=1) is None


async def test_seeded_appointments_never_overlap(
    session: AsyncSession, staff_context: AuditContext
) -> None:
    # The exclusion constraint would reject overlaps; this documents that the
    # seeder produces a valid schedule rather than relying on luck.
    report = await seed_synthetic_data(session, patients=10, doctors=2)
    await session.commit()
    assert report is not None
    await apply_clinic_scope(session, ClinicScope(clinic_id=report.clinic_id))
    rows = (await session.scalars(select(Appointment))).all()
    assert len(rows) == 2 * APPOINTMENTS_PER_DOCTOR


class _Counter(TypedDict):
    visits: Annotated[int, operator.add]


async def test_runtime_role_can_persist_and_resume_graph_state(settings: Settings) -> None:
    graph = StateGraph(_Counter)
    graph.add_node("visit", lambda state: {"visits": 1})
    graph.add_edge(START, "visit")
    graph.add_edge("visit", END)
    config = {"configurable": {"thread_id": "integration-test-thread"}}

    async with open_checkpointer(settings) as checkpointer:
        compiled = graph.compile(checkpointer=checkpointer)
        await compiled.ainvoke({"visits": 0}, config)
        await compiled.ainvoke({"visits": 0}, config)
        state = await compiled.aget_state(config)

    assert state.values["visits"] == 2

    with psycopg.connect(settings.conninfo(admin=True)) as conn:
        schema = conn.execute(
            "SELECT table_schema FROM information_schema.tables WHERE table_name = 'checkpoints'"
        ).fetchall()
    assert schema == [(CHECKPOINT_SCHEMA,)]


def test_readiness_uses_the_runtime_role(client: TestClient) -> None:
    body = client.get("/readyz").json()
    assert body == {
        "status": "ok",
        "database": "ok",
        "app_env": "ci",
        "real_patient_data_allowed": False,
    }


def test_ensure_runtime_role_is_idempotent_and_password_works(settings: Settings) -> None:
    bootstrap.ensure_runtime_role(settings)
    bootstrap.ensure_runtime_role(settings)
    with psycopg.connect(settings.conninfo(admin=False)) as conn:
        assert conn.execute("SELECT current_user").fetchone() == (settings.app_db_user,)


async def _run_one_checkpoint(settings: Settings, thread_id: str, marker: object) -> None:
    """Run a one-node graph that puts `marker` in the `note` channel.

    A str lands inline in the `checkpoints` row; anything else goes through the
    serializer into `checkpoint_blobs`.
    """
    graph = StateGraph(_Note)
    graph.add_node("note", lambda state: {"note": marker})
    graph.add_edge(START, "note")
    graph.add_edge("note", END)
    async with open_checkpointer(settings) as checkpointer:
        await graph.compile(checkpointer=checkpointer).ainvoke(
            {"note": ""}, {"configurable": {"thread_id": thread_id}}
        )


class _Note(TypedDict):
    note: object


async def test_checkpoint_blobs_are_stored_encrypted(settings: Settings) -> None:
    """Values the checkpointer puts in `checkpoint_blobs` are ciphertext.

    This is where message objects go, which is where conversation content will
    live. `test_inlined_primitive_channel_values_are_not_encrypted` covers what
    this does not reach.
    """
    marker = "Ana Gómez Ruiz cita 9 de la mañana"
    await _run_one_checkpoint(settings, "encryption-test-thread", [marker])

    with psycopg.connect(settings.conninfo(admin=True)) as conn:
        blobs = conn.execute(
            f"SELECT blob FROM {CHECKPOINT_SCHEMA}.checkpoint_blobs WHERE blob IS NOT NULL"  # noqa: S608
        ).fetchall()

    assert blobs, "no blob values were written, so this check would prove nothing"
    stored = b"".join(bytes(row[0]) for row in blobs)
    assert marker.encode("utf-8") not in stored
    assert all(bytes(row[0])[0] == 1 for row in blobs), "not our ciphertext key version"


async def test_inlined_primitive_channel_values_are_not_encrypted(settings: Settings) -> None:
    """A deliberate record of a LangGraph behaviour the serializer cannot reach.

    `AsyncPostgresSaver.aput` writes channel values that are None, str, int,
    float or bool straight into the `checkpoints` JSONB column, and only routes
    the rest through the serializer. An encrypting serializer therefore does not
    protect them.

    The graph that would write patient data is M3. Until it exists this test
    pins the behaviour, so a channel holding a patient's name as a bare string
    cannot be added without this failing first. If a LangGraph release starts
    routing primitives through the serializer too, this also fails, and the
    restriction can be lifted.
    """
    marker = "Ana Gómez Ruiz"
    await _run_one_checkpoint(settings, "inlining-test-thread", marker)

    with psycopg.connect(settings.conninfo(admin=True)) as conn:
        rows = conn.execute(
            f"SELECT checkpoint FROM {CHECKPOINT_SCHEMA}.checkpoints"  # noqa: S608
            " WHERE thread_id = 'inlining-test-thread'"
        ).fetchall()

    inlined = [row[0]["channel_values"].get("note") for row in rows]
    assert marker in inlined, (
        "LangGraph no longer inlines primitive channel values; checkpoint state "
        "may now be encrypted end to end, so lift the restriction in CLAUDE.md"
    )


async def test_the_sweep_deletes_only_idle_threads(settings: Settings) -> None:
    await _run_one_checkpoint(settings, "fresh-thread", "reciente")
    await _run_one_checkpoint(settings, "stale-thread", "antiguo")

    older_than_retention = settings.checkpoint_retention_days + 1
    with psycopg.connect(settings.conninfo(admin=True), autocommit=True) as conn:
        for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
            conn.execute(
                f"UPDATE {CHECKPOINT_SCHEMA}.{table}"  # noqa: S608
                f" SET created_at = now() - make_interval(days => %s)"
                " WHERE thread_id = 'stale-thread'",
                (older_than_retention,),
            )

    assert await sweep_expired_threads(settings) == 1

    with psycopg.connect(settings.conninfo(admin=True)) as conn:
        remaining = conn.execute(
            f"SELECT DISTINCT thread_id FROM {CHECKPOINT_SCHEMA}.checkpoints"  # noqa: S608
        ).fetchall()
    assert ("stale-thread",) not in remaining
    assert ("fresh-thread",) in remaining


async def test_seeded_patients_hold_a_document_their_age_allows(
    session: AsyncSession, staff_context: AuditContext
) -> None:
    """The demo data must obey the rule the importer enforces.

    The seeder drew the document type by weight, independent of the birth date
    beside it, so 36 of 200 seeded patients held a document their age rules out
    -- a cédula on a five-year-old. The client then asked for exactly that check
    on the import, and the database he would be shown broke it on a fifth of its
    rows. Two hundred, not twenty: the defect was a minority of draws.
    """
    from src.core.colombian_ids import check_document_for_age

    await seed_synthetic_data(session, patients=200, doctors=2)
    today = datetime.now(BOGOTA).date()

    clinic_id = await session.scalar(select(Patient.clinic_id).limit(1))
    assert clinic_id is not None
    await apply_clinic_scope(session, ClinicScope(clinic_id=clinic_id))
    rows = (await session.execute(select(Patient.document_type, Patient.birth_date))).all()

    assert len(rows) >= 200
    wrong = [
        (str(kind), born)
        for kind, born in rows
        if born is not None and check_document_for_age(str(kind), born, today) is not None
    ]
    assert not wrong, f"{len(wrong)} of {len(rows)} seeded patients break the age rule: {wrong[:3]}"


async def test_seeded_patients_and_doctors_carry_the_contact_fields(
    session: AsyncSession, staff_context: AuditContext
) -> None:
    """EPS and a secondary contact on the patients, a phone and an email on the
    doctors -- the fields the client asked about, so the demo has something to
    show. Not every record has a secondary contact, which is the point of the
    column being optional."""
    from src.registry.models import Doctor

    await seed_synthetic_data(session, patients=60, doctors=4)
    clinic_id = await session.scalar(select(Patient.clinic_id).limit(1))
    assert clinic_id is not None
    await apply_clinic_scope(session, ClinicScope(clinic_id=clinic_id))

    patients = (await session.scalars(select(Patient))).all()
    assert all(p.eps for p in patients)
    with_contact = [p for p in patients if p.secondary_contact_phone]
    assert 0 < len(with_contact) < len(patients)
    assert all(p.secondary_contact_name for p in with_contact)

    doctors = (await session.scalars(select(Doctor))).all()
    assert all(d.phone and d.email for d in doctors)
    assert any(d.secondary_phone for d in doctors)
    assert any(d.secondary_phone is None for d in doctors)
