"""Every status change records an event, in the same transaction (ADR-14).

The guarantee being tested is not "events get written" — it is that an event
and the change it records cannot come apart. A funnel missing the cancellations
that happened during a rolled-back transaction is a funnel that is quietly
wrong, and quietly is the problem.

Two further properties here, both load-bearing for the report:

**Patient and staff actions are distinguishable.** "Patients cancel 30% of
appointments" and "staff cancel 30%" are different problems with different
answers, and a funnel that merges them hides the one a manager is looking for.

**The table is append-only to the runtime role.** A report computed from rows
the application could rewrite is not evidence of anything.
"""

from __future__ import annotations

import psycopg
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.audit.context import AuditContext
from src.core.config import Settings
from src.core.tenancy import ClinicScope, apply_clinic_scope
from src.scheduling import booking
from src.scheduling.events import ACTOR_PATIENT, ACTOR_STAFF, record_event
from src.scheduling.models import (
    Appointment,
    AppointmentEvent,
    AppointmentEventKind,
    AppointmentStatus,
)
from tests.integration.factories import Graph, appointment, at, create_graph


@pytest.fixture(autouse=True)
def _audited(staff_context: AuditContext) -> AuditContext:
    return staff_context


async def _booked(session: AsyncSession, graph: Graph, day: int = 3) -> Appointment:
    row = appointment(graph, at(day, 9, 0))
    session.add(row)
    await session.flush()
    return row


async def _events(session: AsyncSession) -> list[AppointmentEvent]:
    return list(
        (await session.execute(select(AppointmentEvent).order_by(AppointmentEvent.id)))
        .scalars()
        .all()
    )


# ---------------------------------------------------- the ADR-14 guarantee


async def test_cancelling_records_an_event(session: AsyncSession) -> None:
    graph = await create_graph(session)
    row = await _booked(session, graph)

    await booking.cancel(
        session, clinic_id=graph.clinic_id, appointment_id=row.id, reason="patient_request"
    )

    events = await _events(session)
    assert len(events) == 1
    assert events[0].kind is AppointmentEventKind.CANCELLED
    assert events[0].appointment_id == row.id
    assert events[0].reason == "patient_request"


async def test_rescheduling_records_an_event_against_the_original(
    session: AsyncSession,
) -> None:
    """The funnel counts "this appointment was moved". The replacement's own
    events tell the rest of the story, so recording against the new row would
    double-count the move."""
    graph = await create_graph(session)
    row = await _booked(session, graph)

    result = await booking.reschedule(
        session,
        clinic_id=graph.clinic_id,
        appointment_id=row.id,
        start=at(4, 11, 0),
        end=at(4, 11, 20),
    )
    assert result.appointment is not None

    events = await _events(session)
    moved = [e for e in events if e.kind is AppointmentEventKind.RESCHEDULED]
    assert len(moved) == 1
    assert moved[0].appointment_id == row.id, "recorded against the replacement, not the original"


async def test_the_event_denormalises_the_doctor_and_the_site(
    session: AsyncSession,
) -> None:
    """A doctor who moves to another sede must not retroactively move their
    own history, so the event carries both rather than joining for them."""
    graph = await create_graph(session)
    row = await _booked(session, graph)

    await booking.cancel(session, clinic_id=graph.clinic_id, appointment_id=row.id)

    event = (await _events(session))[0]
    assert event.doctor_id == graph.doctor_id
    assert event.location_id == graph.location_id


async def test_an_event_does_not_survive_a_rolled_back_change(
    session: AsyncSession,
) -> None:
    """The guarantee, stated as the failure it prevents.

    `record_event` flushes and never commits, so the event lives in the
    caller's transaction. Rolling that transaction back must take the event
    with it -- otherwise the funnel counts a cancellation that never happened.
    """
    graph = await create_graph(session)
    row = await _booked(session, graph)
    await session.commit()
    # A commit ends the transaction and clears `app.clinic_id`, which is
    # transaction-local by design (§7). Without re-applying it the policies
    # match no row and `cancel` has nothing to cancel.
    await apply_clinic_scope(session, ClinicScope(clinic_id=graph.clinic_id))

    await booking.cancel(session, clinic_id=graph.clinic_id, appointment_id=row.id)
    assert len(await _events(session)) == 1, "the event should exist before the rollback"

    appointment_id = row.id  # read before the rollback expires the instance
    await session.rollback()

    # Re-scoped, because the rollback ended that transaction too.
    await apply_clinic_scope(session, ClinicScope(clinic_id=graph.clinic_id))

    # Both gone together: no event, and the appointment is active again.
    assert await _events(session) == []
    refreshed = (
        await session.execute(select(Appointment).where(Appointment.id == appointment_id))
    ).scalar_one()
    assert refreshed.status is not AppointmentStatus.CANCELLED


# ------------------------------------------------- patient versus staff


async def test_a_patient_cancellation_is_distinguishable_from_a_staff_one(
    session: AsyncSession,
) -> None:
    """The split a manager actually needs. Merging them hides whether the
    clinic or its patients are driving the cancellations."""
    graph = await create_graph(session)
    theirs = await _booked(session, graph, day=3)
    ours = await _booked(session, graph, day=5)

    await booking.cancel(
        session, clinic_id=graph.clinic_id, appointment_id=theirs.id, actor=ACTOR_PATIENT
    )
    await booking.cancel(
        session, clinic_id=graph.clinic_id, appointment_id=ours.id, actor=ACTOR_STAFF
    )

    actors = {e.appointment_id: e.actor for e in await _events(session)}
    assert actors[theirs.id] == ACTOR_PATIENT
    assert actors[ours.id] == ACTOR_STAFF


async def test_an_unknown_actor_is_refused_rather_than_defaulted(
    session: AsyncSession,
) -> None:
    """Defaulting would make the split wrong, and wrong quietly."""
    graph = await create_graph(session)
    row = await _booked(session, graph)

    with pytest.raises(ValueError, match="actor must be one of"):
        await record_event(
            session,
            appointment=row,
            kind=AppointmentEventKind.CANCELLED,
            actor="receptionist",
        )


# --------------------------------------------------------- append-only


async def test_the_runtime_role_cannot_rewrite_an_event(
    session: AsyncSession, settings: Settings
) -> None:
    """The same privilege barrier the audit log has.

    A funnel computed from rows the application could edit is not evidence of
    anything, so the grant is INSERT and SELECT with no UPDATE and no DELETE.
    """
    graph = await create_graph(session)
    row = await _booked(session, graph)
    await booking.cancel(session, clinic_id=graph.clinic_id, appointment_id=row.id)
    await session.commit()

    with psycopg.connect(settings.conninfo(admin=False), autocommit=True) as conn:
        conn.execute(f"SET app.clinic_id = '{graph.clinic_id}'")
        for statement in (
            "UPDATE app.appointment_events SET kind = 'completed'",
            "DELETE FROM app.appointment_events",
        ):
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                conn.execute(statement)


async def test_events_are_isolated_per_clinic(session: AsyncSession) -> None:
    """Row-level security, as for every other clinic table (§7). One clinic's
    funnel must not contain another's rows."""
    first = await create_graph(session, name="Clínica Uno")
    row = await _booked(session, first)
    await booking.cancel(session, clinic_id=first.clinic_id, appointment_id=row.id)
    await session.flush()

    second = await create_graph(session, name="Clínica Dos")
    await apply_clinic_scope(session, ClinicScope(clinic_id=second.clinic_id))

    visible = (
        await session.execute(select(func.count()).select_from(AppointmentEvent))
    ).scalar_one()
    assert visible == 0, "another clinic's events were visible"
