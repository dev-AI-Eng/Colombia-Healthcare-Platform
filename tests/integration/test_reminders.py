"""The two scheduled sweeps over appointments: reminders due, and attendance.

Neither sweep sends anything or decides anything. The reminder sweep answers
"which appointments are due a reminder, and have not had one", and marks the
ones a caller has dispatched; the dispatch itself is M4's, behind the channel
layer. The attendance sweep answers "which of yesterday's appointments has
nobody recorded as attended", and flags them for a human to confirm.

The distinction is the point and is tested here: software cannot observe
whether a patient walked in. It can only notice that nobody said they did.
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.audit.context import AuditContext
from src.core.tenancy import ClinicScope, apply_clinic_scope
from src.scheduling import reminders
from src.scheduling.models import (
    Appointment,
    AppointmentStatus,
    Escalation,
    EscalationReason,
)
from tests.integration.factories import Graph, appointment, at, create_graph

#: The sweep runs at this instant in every test, so "due" is unambiguous.
NOW = at(10, 8)


@pytest.fixture(autouse=True)
def _audited(staff_context: AuditContext) -> AuditContext:
    """Reading and writing appointments is a patient-data access."""
    return staff_context


async def _add(
    session: AsyncSession,
    graph: Graph,
    start: dt.datetime,
    *,
    status: AppointmentStatus = AppointmentStatus.CONFIRMED,
    sent_48h: bool = False,
    sent_24h: bool = False,
) -> Appointment:
    row = appointment(graph, start, status=status)
    row.reminder_48h_sent = sent_48h
    row.reminder_24h_sent = sent_24h
    session.add(row)
    await session.flush()
    return row


# ----------------------------------------------------------------- reminders


async def test_an_appointment_two_days_out_is_due_its_48h_reminder(
    session: AsyncSession,
) -> None:
    graph = await create_graph(session)
    # 08:00 on the 10th; due 07:00 on the 12th, so 47 hours away: inside 48,
    # outside 24.
    await _add(session, graph, at(12, 7))

    due = await reminders.reminders_due(session, clinic_id=graph.clinic_id, now=NOW)

    assert [d.kind for d in due] == ["48h"]


async def test_an_appointment_tomorrow_is_due_its_24h_reminder(
    session: AsyncSession,
) -> None:
    graph = await create_graph(session)
    # 23 hours away: inside the 24h window.
    await _add(session, graph, at(11, 7), sent_48h=True)

    due = await reminders.reminders_due(session, clinic_id=graph.clinic_id, now=NOW)

    assert [d.kind for d in due] == ["24h"]


async def test_a_reminder_already_sent_is_not_due_again(session: AsyncSession) -> None:
    """The flag is what stops a patient being messaged twice.

    A sweep that ran every few minutes and ignored the flag would message the
    same patient on every run.
    """
    graph = await create_graph(session)
    await _add(session, graph, at(12, 7), sent_48h=True)

    due = await reminders.reminders_due(session, clinic_id=graph.clinic_id, now=NOW)

    assert due == []


async def test_an_appointment_further_out_than_the_window_is_not_due(
    session: AsyncSession,
) -> None:
    graph = await create_graph(session)
    await _add(session, graph, at(20, 9))

    assert await reminders.reminders_due(session, clinic_id=graph.clinic_id, now=NOW) == []


async def test_an_appointment_in_the_past_is_never_due(session: AsyncSession) -> None:
    """A reminder for an appointment that has already happened is noise."""
    graph = await create_graph(session)
    await _add(session, graph, at(9, 9))

    assert await reminders.reminders_due(session, clinic_id=graph.clinic_id, now=NOW) == []


@pytest.mark.parametrize(
    "inactive",
    [
        AppointmentStatus.CANCELLED,
        AppointmentStatus.NO_SHOW,
        AppointmentStatus.RESCHEDULED,
        AppointmentStatus.COMPLETED,
    ],
)
async def test_a_cancelled_appointment_is_never_reminded(
    session: AsyncSession, inactive: AppointmentStatus
) -> None:
    """Reminding somebody of an appointment they cancelled is the worst case.

    It reads as the clinic having ignored the cancellation.
    """
    graph = await create_graph(session)
    await _add(session, graph, at(12, 7), status=inactive)

    assert await reminders.reminders_due(session, clinic_id=graph.clinic_id, now=NOW) == []


async def test_a_hold_is_never_reminded(session: AsyncSession) -> None:
    """A hold is an offer nobody has accepted, so there is nothing to confirm."""
    graph = await create_graph(session)
    row = appointment(graph, at(12, 7), status=AppointmentStatus.HOLD)
    row.expires_at = at(10, 9)
    session.add(row)
    await session.flush()

    assert await reminders.reminders_due(session, clinic_id=graph.clinic_id, now=NOW) == []


async def test_marking_a_reminder_sent_stops_it_coming_back(
    session: AsyncSession,
) -> None:
    """What a dispatcher calls once the message is actually away."""
    graph = await create_graph(session)
    row = await _add(session, graph, at(12, 7))

    due = await reminders.reminders_due(session, clinic_id=graph.clinic_id, now=NOW)
    assert len(due) == 1

    await reminders.mark_sent(
        session, clinic_id=graph.clinic_id, appointment_id=row.id, kind=due[0].kind
    )
    await session.flush()

    assert await reminders.reminders_due(session, clinic_id=graph.clinic_id, now=NOW) == []
    assert row.reminder_48h_sent is True


async def test_the_due_record_carries_what_a_message_needs(
    session: AsyncSession,
) -> None:
    """The sweep's output has to be enough to send from, without a second query."""
    graph = await create_graph(session)
    row = await _add(session, graph, at(12, 7))

    (due,) = await reminders.reminders_due(session, clinic_id=graph.clinic_id, now=NOW)

    assert due.appointment_id == row.id
    assert due.patient_id == graph.patient_id
    assert due.doctor_id == graph.doctor_id
    assert due.starts_at == at(12, 7)
    assert due.kind == "48h"


# ---------------------------------------------------------------- attendance


async def test_yesterdays_unmarked_appointment_is_flagged_for_a_human(
    session: AsyncSession,
) -> None:
    """The sweep does not decide a no-show. It asks somebody to.

    Rev2 is explicit: software cannot observe attendance, only record it. So
    the row keeps its status and an escalation carries the question.
    """
    graph = await create_graph(session)
    row = await _add(session, graph, at(9, 9))

    flagged = await reminders.flag_unrecorded_attendance(
        session, clinic_id=graph.clinic_id, now=NOW
    )
    await session.flush()

    assert flagged == 1
    # The status is untouched: nobody has said what happened yet.
    assert row.status == AppointmentStatus.CONFIRMED

    escalations = list(
        (
            await session.scalars(
                select(Escalation).where(
                    Escalation.reason == EscalationReason.ATTENDANCE_UNRECORDED
                )
            )
        ).all()
    )
    assert len(escalations) == 1
    assert escalations[0].appointment_id == row.id
    assert "asistió" in escalations[0].detail_es


async def test_an_appointment_marked_checked_in_is_not_flagged(
    session: AsyncSession,
) -> None:
    graph = await create_graph(session)
    await _add(session, graph, at(9, 9), status=AppointmentStatus.CHECKED_IN)

    assert (
        await reminders.flag_unrecorded_attendance(session, clinic_id=graph.clinic_id, now=NOW) == 0
    )


@pytest.mark.parametrize(
    "settled",
    [
        AppointmentStatus.COMPLETED,
        AppointmentStatus.NO_SHOW,
        AppointmentStatus.CANCELLED,
        AppointmentStatus.RESCHEDULED,
    ],
)
async def test_an_appointment_already_settled_is_not_flagged(
    session: AsyncSession, settled: AppointmentStatus
) -> None:
    """Somebody has already said what happened; there is nothing to ask."""
    graph = await create_graph(session)
    await _add(session, graph, at(9, 9), status=settled)

    assert (
        await reminders.flag_unrecorded_attendance(session, clinic_id=graph.clinic_id, now=NOW) == 0
    )


async def test_an_appointment_still_in_the_future_is_not_flagged(
    session: AsyncSession,
) -> None:
    graph = await create_graph(session)
    await _add(session, graph, at(12, 7))

    assert (
        await reminders.flag_unrecorded_attendance(session, clinic_id=graph.clinic_id, now=NOW) == 0
    )


async def test_an_appointment_earlier_today_is_not_flagged_yet(
    session: AsyncSession,
) -> None:
    """The sweep is end-of-day. A patient due at 07:00 may still be in reception."""
    graph = await create_graph(session)
    await _add(session, graph, at(10, 7))

    assert (
        await reminders.flag_unrecorded_attendance(session, clinic_id=graph.clinic_id, now=NOW) == 0
    )


async def test_running_the_attendance_sweep_twice_flags_once(
    session: AsyncSession,
) -> None:
    """The sweep runs on a timer, so a second run must not duplicate the queue."""
    graph = await create_graph(session)
    await _add(session, graph, at(9, 9))

    first = await reminders.flag_unrecorded_attendance(session, clinic_id=graph.clinic_id, now=NOW)
    await session.flush()
    second = await reminders.flag_unrecorded_attendance(session, clinic_id=graph.clinic_id, now=NOW)
    await session.flush()

    assert (first, second) == (1, 0)
    rows = list(
        (
            await session.scalars(
                select(Escalation).where(
                    Escalation.reason == EscalationReason.ATTENDANCE_UNRECORDED
                )
            )
        ).all()
    )
    assert len(rows) == 1


async def test_neither_sweep_reaches_another_clinic(session: AsyncSession) -> None:
    """Row-level security scopes both sweeps, and this is what proves it."""
    mine = await create_graph(session)
    await _add(session, mine, at(12, 7))
    await _add(session, mine, at(9, 9))
    await session.commit()

    theirs = await create_graph(session, name="Clínica Ajena")
    await _add(session, theirs, at(12, 7))
    await _add(session, theirs, at(9, 9))
    await session.commit()

    await apply_clinic_scope(session, ClinicScope(clinic_id=mine.clinic_id))
    due = await reminders.reminders_due(session, clinic_id=mine.clinic_id, now=NOW)
    flagged = await reminders.flag_unrecorded_attendance(session, clinic_id=mine.clinic_id, now=NOW)

    assert len(due) == 1
    assert flagged == 1
    assert due[0].patient_id == mine.patient_id
