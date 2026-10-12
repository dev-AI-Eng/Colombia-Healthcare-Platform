"""The scheduled housekeeping run.

The sweep is not what makes an expired hold release its slot -- `availability`
does that the moment `expires_at` passes. What these prove is that the sweep
cannot do harm: it must not touch a live hold, a real booking, or another
clinic's rows, and it must be safe to run twice.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.audit.context import AuditContext
from src.core.tenancy import ClinicScope, apply_clinic_scope
from src.scheduling import booking, sweeper
from src.scheduling.models import Appointment, AppointmentStatus
from tests.integration.factories import Graph, at, create_graph

SLOT = at(7, 9)
LATER = at(7, 11)
#: After the first hold lapses (08:15) and before the second (10:00).
AFTER_THE_FIRST = at(7, 9, 30)


@pytest.fixture(autouse=True)
def _audited(staff_context: AuditContext) -> AuditContext:
    """Holding writes an audit entry, which needs an actor bound."""
    return staff_context


async def _hold(session: AsyncSession, graph: Graph, start, minutes: int):  # type: ignore[no-untyped-def]
    return await booking.hold(
        session,
        clinic_id=graph.clinic_id,
        patient_id=graph.patient_id,
        doctor_id=graph.doctor_id,
        location_id=graph.location_id,
        appointment_type_id=graph.appointment_type_id,
        start=start,
        end=start + timedelta(minutes=20),
        now=at(7, 8),
        minutes=minutes,
    )


async def _count(session: AsyncSession) -> int:
    return (await session.execute(select(func.count()).select_from(Appointment))).scalar_one()


async def test_the_sweep_removes_a_lapsed_hold(session: AsyncSession) -> None:
    graph = await create_graph(session)
    await _hold(session, graph, SLOT, minutes=15)  # lapses 08:15

    holds, clinics, _due, _flagged = await sweeper.sweep_every_clinic(session, now=AFTER_THE_FIRST)
    assert (holds, clinics) == (1, 1)
    assert await _count(session) == 0


async def test_the_sweep_leaves_a_live_hold_alone(session: AsyncSession) -> None:
    """Deleting a hold a patient is still deciding on would lose their slot."""
    graph = await create_graph(session)
    await _hold(session, graph, SLOT, minutes=120)  # lapses 10:00

    removed, *_ = await sweeper.sweep_every_clinic(session, now=AFTER_THE_FIRST)
    assert removed == 0
    assert await _count(session) == 1


async def test_the_sweep_never_touches_a_real_booking(session: AsyncSession) -> None:
    """A booking has no expiry, so nothing about it can look lapsed."""
    graph = await create_graph(session)
    await booking.book(
        session,
        clinic_id=graph.clinic_id,
        patient_id=graph.patient_id,
        doctor_id=graph.doctor_id,
        location_id=graph.location_id,
        appointment_type_id=graph.appointment_type_id,
        start=SLOT,
        end=SLOT + timedelta(minutes=20),
    )
    removed, *_ = await sweeper.sweep_every_clinic(
        session,
        now=at(8, 9),  # a day later
    )
    assert removed == 0
    assert await _count(session) == 1


async def test_running_the_sweep_twice_is_harmless(session: AsyncSession) -> None:
    """Two scheduled runs can overlap; the second must find nothing to do."""
    graph = await create_graph(session)
    await _hold(session, graph, SLOT, minutes=15)

    first, *_ = await sweeper.sweep_every_clinic(session, now=AFTER_THE_FIRST)
    second, *_ = await sweeper.sweep_every_clinic(session, now=AFTER_THE_FIRST)
    assert (first, second) == (1, 0)


async def test_the_sweep_covers_every_clinic(session: AsyncSession) -> None:
    """A single-clinic sweep would leave every other clinic's table growing.

    Each clinic is walked under its own scope rather than swept in one
    statement: a job able to reach across clinics would be the only code that
    can, which is not a privilege worth creating for housekeeping.
    """
    first = await create_graph(session, name="Clínica Uno")
    await _hold(session, first, SLOT, minutes=15)
    await session.commit()

    second = await create_graph(session, name="Clínica Dos")
    await _hold(session, second, SLOT, minutes=15)
    await session.commit()

    removed, clinics, *_ = await sweeper.sweep_every_clinic(session, now=AFTER_THE_FIRST)
    assert clinics >= 2
    assert removed == 2, "both clinics' lapsed holds should have gone"

    for graph in (first, second):
        await apply_clinic_scope(session, ClinicScope(clinic_id=graph.clinic_id))
        assert await _count(session) == 0


async def test_the_slot_is_bookable_again_after_a_sweep(session: AsyncSession) -> None:
    """The visible outcome: the time a lapsed hold occupied comes back."""
    graph = await create_graph(session)
    await _hold(session, graph, SLOT, minutes=15)
    await sweeper.sweep_every_clinic(session, now=AFTER_THE_FIRST)

    booked = await booking.book(
        session,
        clinic_id=graph.clinic_id,
        patient_id=graph.patient_id,
        doctor_id=graph.doctor_id,
        location_id=graph.location_id,
        appointment_type_id=graph.appointment_type_id,
        start=SLOT,
        end=SLOT + timedelta(minutes=20),
    )
    assert booked.ok
    assert booked.appointment is not None
    assert booked.appointment.status is AppointmentStatus.SCHEDULED
