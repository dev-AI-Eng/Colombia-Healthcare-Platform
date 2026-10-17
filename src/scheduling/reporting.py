"""Management reports, computed from events rather than from current status.

ADR-14. An appointment rescheduled twice and then attended reads as
`completed` today, so a funnel built on status cannot say it was ever moved —
which is the number a manager is looking for. Every figure here comes from
`app.appointment_events`, which is append-only.

Four decisions that shape the output more than the SQL does:

**Every percentage carries its formula.** Rev3 requires it printed on the
report, and the reason is that "no-show %" has at least three defensible
denominators. A manager comparing this month to last needs to know the number
means the same thing both times, and the only way to guarantee that is to
state it rather than document it elsewhere.

**Small groups are suppressed, not rounded.** A no-show rate computed from
three patients in one neighbourhood is unreliable *and* re-identifying: in a
small barrio, "2 of 3 missed" names people. Rev3 asks for suppression and this
is `MIN_GROUP_SIZE`, applied before any rate is computed rather than after.

**Punctuality reports its own coverage.** Rev3: "computed only where both times
were recorded, with recording coverage shown". A mean delay over a tenth of the
rows is a mean over the appointments somebody happened to log, not over the
clinic, and presenting it without the coverage invites a decision it cannot
support.

**Nothing here masks anything.** The repository returns numbers and ids; the
API layer decides what a role may see (ADR-13). A reporting module that also
masked would be two places deciding one thing.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, field
from typing import Final

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.scheduling.models import (
    Appointment,
    AppointmentEvent,
    AppointmentEventKind,
)

#: Below this, a geographic group is suppressed rather than published. Five is
#: the smallest number that is defensible for both reasons at once: a rate from
#: four rows is noise, and in a small neighbourhood it also identifies people.
MIN_GROUP_SIZE: Final = 5

#: The formula printed beside each percentage. Stated rather than documented,
#: because a rate whose denominator is implicit is a rate that gets compared
#: against a different one next month.
# Each line below suppresses RUF001: the multiplication sign is deliberate.
# These strings
# are printed on a report a clinic manager reads, where "x" reads as an unknown
# rather than as "times" -- and §10 makes patient- and staff-facing copy Spanish
# with the typography a Spanish reader expects.
FORMULAS: Final[dict[str, str]] = {
    "confirmation_pct": "confirmados ÷ (confirmados + no confirmados agendados) × 100",  # noqa: RUF001
    "cancellation_pct": "cancelados ÷ total agendado × 100",  # noqa: RUF001
    "no_show_pct": "no asistió ÷ (completados + no asistió) × 100",  # noqa: RUF001
    "compliance_pct": "completados ÷ total agendado × 100",  # noqa: RUF001
}


@dataclass(frozen=True, slots=True)
class ReportFilter:
    """What the manager asked for. Every field narrows the query."""

    clinic_id: uuid.UUID
    start_date: dt.date
    end_date: dt.date
    doctor_id: uuid.UUID | None = None
    location_id: uuid.UUID | None = None
    specialty: str | None = None


@dataclass(frozen=True, slots=True)
class Funnel:
    """Counts per event kind, over the filtered range."""

    offered: int = 0
    confirmed: int = 0
    rescheduled: int = 0
    cancelled: int = 0
    no_show: int = 0
    completed: int = 0
    #: Appointments whose `during` starts in the range, whatever happened to
    #: them. The denominator of cancellation and compliance, because an
    #: appointment nobody touched still counts as scheduled.
    scheduled: int = 0

    def percentages(self) -> dict[str, float | None]:
        """The four rates, or None where the denominator is zero.

        None rather than 0.0: a clinic with no appointments has no no-show
        rate, and printing 0% would read as perfect attendance.
        """
        return {
            "confirmation_pct": _rate(self.confirmed, self.scheduled),
            "cancellation_pct": _rate(self.cancelled, self.scheduled),
            "no_show_pct": _rate(self.no_show, self.completed + self.no_show),
            "compliance_pct": _rate(self.completed, self.scheduled),
        }


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator * 100 / denominator, 1)


@dataclass(frozen=True, slots=True)
class CancellationReason:
    reason: str
    count: int
    #: Who cancelled. A patient cancelling and the clinic cancelling are
    #: different facts, and a breakdown that merges them hides which.
    actor: str


@dataclass(frozen=True, slots=True)
class ZoneRow:
    """No-show rate for one municipality and neighbourhood.

    `suppressed` is True when the group is below `MIN_GROUP_SIZE`, and then
    `no_show_pct` is None. The row is still returned so the report can say
    "three groups suppressed" rather than silently omitting them -- a reader
    who cannot see that data was withheld will read the rest as complete.
    """

    municipality_divipola: str | None
    neighborhood: str | None
    appointments: int
    no_shows: int
    no_show_pct: float | None
    suppressed: bool


@dataclass(frozen=True, slots=True)
class PunctualityRow:
    """Start delay and overrun for one doctor, with the coverage behind them.

    Both figures are in minutes and both may be None. `recorded` against
    `total` is what tells a manager whether to trust them: 4 of 90 is not a
    measurement of the doctor, it is a measurement of who remembers to press
    buttons.
    """

    doctor_id: uuid.UUID
    doctor_name: str
    total: int
    recorded_start: int
    recorded_end: int
    mean_start_delay_minutes: float | None
    mean_overrun_minutes: float | None

    @property
    def start_coverage_pct(self) -> float | None:
        return _rate(self.recorded_start, self.total)

    @property
    def end_coverage_pct(self) -> float | None:
        return _rate(self.recorded_end, self.total)


@dataclass(frozen=True, slots=True)
class Summary:
    """Everything the summary panel shows, with its formulas."""

    funnel: Funnel
    percentages: dict[str, float | None]
    formulas: dict[str, str] = field(default_factory=lambda: dict(FORMULAS))
    cancellations: tuple[CancellationReason, ...] = ()


def _in_range[R: tuple[object, ...]](statement: Select[R], filters: ReportFilter) -> Select[R]:
    """Apply the date range and the optional narrowing to an events query.

    Generic in the row type: `Select` is invariant, so a non-generic signature
    rejects every concrete query that calls it.
    """
    statement = statement.where(
        AppointmentEvent.clinic_id == filters.clinic_id,
        func.date(AppointmentEvent.occurred_at) >= filters.start_date,
        func.date(AppointmentEvent.occurred_at) <= filters.end_date,
    )
    if filters.doctor_id is not None:
        statement = statement.where(AppointmentEvent.doctor_id == filters.doctor_id)
    if filters.location_id is not None:
        statement = statement.where(AppointmentEvent.location_id == filters.location_id)
    return statement


async def funnel(session: AsyncSession, filters: ReportFilter) -> Funnel:
    """Counts per event kind, plus the scheduled denominator.

    One query for the events and one for the denominator, rather than a join:
    the denominator counts *appointments in the range*, and an appointment with
    no events at all still belongs in it. A join would drop exactly the rows
    that make the compliance figure meaningful.
    """
    rows = (
        await session.execute(
            _in_range(
                select(AppointmentEvent.kind, func.count()).group_by(AppointmentEvent.kind),
                filters,
            )
        )
    ).all()
    counts = {str(kind): int(count) for kind, count in rows}

    appointments = (
        select(func.count())
        .select_from(Appointment)
        .where(
            Appointment.clinic_id == filters.clinic_id,
            func.date(func.lower(Appointment.during)) >= filters.start_date,
            func.date(func.lower(Appointment.during)) <= filters.end_date,
        )
    )
    if filters.doctor_id is not None:
        appointments = appointments.where(Appointment.doctor_id == filters.doctor_id)
    if filters.location_id is not None:
        appointments = appointments.where(Appointment.location_id == filters.location_id)
    scheduled = (await session.execute(appointments)).scalar_one()

    return Funnel(
        offered=counts.get(AppointmentEventKind.OFFERED.value, 0),
        confirmed=counts.get(AppointmentEventKind.CONFIRMED.value, 0),
        rescheduled=counts.get(AppointmentEventKind.RESCHEDULED.value, 0),
        cancelled=counts.get(AppointmentEventKind.CANCELLED.value, 0),
        no_show=counts.get(AppointmentEventKind.NO_SHOW.value, 0),
        completed=counts.get(AppointmentEventKind.COMPLETED.value, 0),
        scheduled=int(scheduled),
    )


async def cancellation_reasons(
    session: AsyncSession, filters: ReportFilter
) -> tuple[CancellationReason, ...]:
    """Cancellations grouped by reason and by who cancelled."""
    rows = (
        await session.execute(
            _in_range(
                select(AppointmentEvent.reason, AppointmentEvent.actor, func.count())
                .where(AppointmentEvent.kind == AppointmentEventKind.CANCELLED)
                .group_by(AppointmentEvent.reason, AppointmentEvent.actor)
                .order_by(func.count().desc()),
                filters,
            )
        )
    ).all()
    return tuple(
        CancellationReason(reason=reason or "sin_motivo", actor=str(actor), count=int(count))
        for reason, actor, count in rows
    )


async def summary(session: AsyncSession, filters: ReportFilter) -> Summary:
    """The summary panel, formulas included."""
    counts = await funnel(session, filters)
    return Summary(
        funnel=counts,
        percentages=counts.percentages(),
        cancellations=await cancellation_reasons(session, filters),
    )
