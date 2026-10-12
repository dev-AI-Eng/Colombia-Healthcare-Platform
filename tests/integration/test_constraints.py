"""Database-enforced scheduling rules (ADR-16, ADR-17). These must fail if a
constraint is dropped or weakened."""

from __future__ import annotations

import asyncio
from datetime import time, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.core.config import Settings
from src.core.tenancy import ClinicScope, apply_clinic_scope
from src.scheduling.models import AppointmentStatus, AvailabilityRule
from tests.integration.factories import appointment, at, create_graph

HOLD_EXPIRY = at(1, 8)
# exclusion_violation, deadlock_detected
LOST_RACE_SQLSTATES = {"23P01", "40P01"}


async def test_overlapping_active_appointments_are_rejected(session: AsyncSession) -> None:
    graph = await create_graph(session)
    session.add(appointment(graph, at(1, 9)))
    await session.flush()
    session.add(appointment(graph, at(1, 9, 10)))
    with pytest.raises(IntegrityError, match="no_overlapping_active_appointments"):
        await session.flush()


async def test_back_to_back_appointments_are_allowed(session: AsyncSession) -> None:
    graph = await create_graph(session)
    session.add_all([appointment(graph, at(1, 9)), appointment(graph, at(1, 9, 20))])
    await session.flush()


async def test_hold_blocks_a_booking(session: AsyncSession) -> None:
    graph = await create_graph(session)
    session.add(
        appointment(graph, at(1, 11), status=AppointmentStatus.HOLD, expires_at=HOLD_EXPIRY)
    )
    await session.flush()
    session.add(appointment(graph, at(1, 11)))
    with pytest.raises(IntegrityError, match="no_overlapping_active_appointments"):
        await session.flush()


@pytest.mark.parametrize(
    "inactive",
    [AppointmentStatus.CANCELLED, AppointmentStatus.NO_SHOW, AppointmentStatus.RESCHEDULED],
)
async def test_inactive_statuses_free_the_slot(
    session: AsyncSession, inactive: AppointmentStatus
) -> None:
    graph = await create_graph(session)
    session.add_all([appointment(graph, at(1, 14), status=inactive), appointment(graph, at(1, 14))])
    await session.flush()


async def test_hold_without_expiry_is_rejected(session: AsyncSession) -> None:
    graph = await create_graph(session)
    session.add(appointment(graph, at(2, 9), status=AppointmentStatus.HOLD))
    with pytest.raises(IntegrityError, match="expiry_only_for_holds"):
        await session.flush()


async def test_expiry_on_a_non_hold_is_rejected(session: AsyncSession) -> None:
    graph = await create_graph(session)
    session.add(appointment(graph, at(2, 9), expires_at=HOLD_EXPIRY))
    with pytest.raises(IntegrityError, match="expiry_only_for_holds"):
        await session.flush()


async def test_availability_window_must_be_ordered(session: AsyncSession) -> None:
    graph = await create_graph(session)
    session.add(
        AvailabilityRule(
            clinic_id=graph.clinic_id,
            doctor_id=graph.doctor_id,
            location_id=graph.location_id,
            weekday=0,
            start_time=time(12, 0),
            end_time=time(8, 0),
            valid_from=at(1, 0).date(),
        )
    )
    with pytest.raises(IntegrityError, match="time_window_ordered"):
        await session.flush()


@pytest.mark.concurrency
async def test_concurrent_bookings_of_one_slot_admit_exactly_one(
    session: AsyncSession, settings: Settings
) -> None:
    """Separate connections race for overlapping times; the database admits one.

    Losers fail with exclusion_violation or, when their inserts wait on each
    other, deadlock_detected. Any other failure fails the test.
    """
    graph = await create_graph(session)
    await session.commit()

    engine = create_async_engine(settings.app_database_url, poolclass=NullPool)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    start = at(3, 10)

    async def attempt(offset_minutes: int) -> str | None:
        """Return None on success, or the SQLSTATE of a lost race."""
        async with maker() as racer:
            await apply_clinic_scope(racer, ClinicScope(clinic_id=graph.clinic_id))
            racer.add(appointment(graph, start + timedelta(minutes=offset_minutes)))
            try:
                await racer.commit()
            except DBAPIError as exc:
                sqlstate = getattr(exc.orig, "sqlstate", None)
                if sqlstate not in LOST_RACE_SQLSTATES:
                    raise
                return str(sqlstate)
            return None

    try:
        outcomes = await asyncio.gather(*(attempt(i) for i in range(12)))
    finally:
        await engine.dispose()

    assert outcomes.count(None) == 1
    assert set(outcomes) - {None} <= LOST_RACE_SQLSTATES


async def test_raw_sql_insert_cannot_double_book(session: AsyncSession) -> None:
    """The constraint holds against SQL that never passes through the ORM.

    The scope's exit criterion names "direct database inserts that bypass the
    service layer". The ORM tests above already bypass `book()`, but they still
    go through SQLAlchemy's mapper; this one writes the row with literal SQL, so
    a reader does not have to trust that the mapper adds nothing. Anyone with a
    psql prompt and the runtime role gets the same refusal.
    """
    graph = await create_graph(session)
    await session.flush()

    insert = text(
        """
        INSERT INTO app.appointments (
            id, clinic_id, patient_id, doctor_id, location_id,
            appointment_type_id, during, status
        ) VALUES (
            gen_random_uuid(), :clinic_id, :patient_id, :doctor_id, :location_id,
            :type_id, tstzrange(:start, :end, '[)'), 'scheduled'
        )
        """
    )
    params = {
        "clinic_id": graph.clinic_id,
        "patient_id": graph.patient_id,
        "doctor_id": graph.doctor_id,
        "location_id": graph.location_id,
        "type_id": graph.appointment_type_id,
    }
    start = at(5, 9)
    await session.execute(insert, {**params, "start": start, "end": start + timedelta(minutes=20)})

    overlap = start + timedelta(minutes=10)
    with pytest.raises(IntegrityError, match="no_overlapping_active_appointments"):
        await session.execute(
            insert, {**params, "start": overlap, "end": overlap + timedelta(minutes=20)}
        )
