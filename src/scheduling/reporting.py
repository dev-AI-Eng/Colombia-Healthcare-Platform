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

from src.audit.context import AccessAction
from src.audit.service import Access, record_accesses
from src.registry.models import Doctor, Patient
from src.scheduling.models import (
    Appointment,
    AppointmentEvent,
    AppointmentEventKind,
    AppointmentStatus,
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


async def zones(session: AsyncSession, filters: ReportFilter) -> tuple[ZoneRow, ...]:
    """No-show rate by municipality and neighbourhood, small groups suppressed.

    Suppression happens before the rate is computed, not after: computing
    "2 of 3 missed" and then hiding it still means the number existed in a
    response somebody could inspect. A group below `MIN_GROUP_SIZE` gets None.

    The null-zone group is returned rather than dropped. Most patients have no
    recorded zone until a clinic's import carries one, and a clinic with no
    zone data should see "we have no zone data" instead of an empty table that
    reads as no no-shows.
    """
    statement = (
        select(
            Patient.municipality_divipola,
            Patient.neighborhood,
            func.count().label("appointments"),
            func.count()
            .filter(AppointmentEvent.kind == AppointmentEventKind.NO_SHOW)
            .label("no_shows"),
        )
        .select_from(AppointmentEvent)
        .join(Appointment, Appointment.id == AppointmentEvent.appointment_id)
        .join(Patient, Patient.id == Appointment.patient_id)
        .where(
            AppointmentEvent.kind.in_(
                (AppointmentEventKind.COMPLETED, AppointmentEventKind.NO_SHOW)
            )
        )
        .group_by(Patient.municipality_divipola, Patient.neighborhood)
        .order_by(func.count().desc())
    )
    rows = (await session.execute(_in_range(statement, filters))).all()

    out: list[ZoneRow] = []
    for municipality, neighborhood, appointments, no_shows in rows:
        total = int(appointments)
        suppressed = total < MIN_GROUP_SIZE
        out.append(
            ZoneRow(
                municipality_divipola=municipality,
                neighborhood=neighborhood,
                appointments=total,
                no_shows=int(no_shows),
                no_show_pct=None if suppressed else _rate(int(no_shows), total),
                suppressed=suppressed,
            )
        )
    return tuple(out)


async def punctuality(session: AsyncSession, filters: ReportFilter) -> tuple[PunctualityRow, ...]:
    """Start delay and overrun per doctor, computed only where recorded.

    Rev3: "computed only where both times were recorded, with recording
    coverage shown". The averages use the rows that have the times; the
    coverage counts all of them. Reporting one without the other is how a mean
    over four appointments gets read as a fact about a doctor.

    Delay is `started_at` minus the planned start, so a doctor who starts early
    shows a negative delay rather than zero -- running ahead is information.
    """
    planned_start = func.lower(Appointment.during)
    planned_end = func.upper(Appointment.during)
    delay_minutes = func.extract("epoch", Appointment.started_at - planned_start) / 60.0
    overrun_minutes = func.extract("epoch", Appointment.ended_at - planned_end) / 60.0

    statement = (
        select(
            Doctor.id,
            Doctor.full_name,
            func.count().label("total"),
            func.count().filter(Appointment.started_at.is_not(None)).label("recorded_start"),
            func.count().filter(Appointment.ended_at.is_not(None)).label("recorded_end"),
            func.avg(delay_minutes).label("mean_delay"),
            func.avg(overrun_minutes).label("mean_overrun"),
        )
        .select_from(Appointment)
        .join(Doctor, Doctor.id == Appointment.doctor_id)
        .where(
            Appointment.clinic_id == filters.clinic_id,
            func.date(planned_start) >= filters.start_date,
            func.date(planned_start) <= filters.end_date,
        )
        .group_by(Doctor.id, Doctor.full_name)
        .order_by(Doctor.full_name)
    )
    if filters.doctor_id is not None:
        statement = statement.where(Appointment.doctor_id == filters.doctor_id)
    if filters.location_id is not None:
        statement = statement.where(Appointment.location_id == filters.location_id)

    rows = (await session.execute(statement)).all()
    return tuple(
        PunctualityRow(
            doctor_id=doctor_id,
            doctor_name=name,
            total=int(total),
            recorded_start=int(recorded_start),
            recorded_end=int(recorded_end),
            mean_start_delay_minutes=None if mean_delay is None else round(float(mean_delay), 1),
            mean_overrun_minutes=None if mean_overrun is None else round(float(mean_overrun), 1),
        )
        for (
            doctor_id,
            name,
            total,
            recorded_start,
            recorded_end,
            mean_delay,
            mean_overrun,
        ) in rows
    )


#: What a patient has said about an upcoming appointment. `NO_RESPONSE` is the
#: follow-up list: reminded, and still silent.
RESPONSE_CONFIRMED: Final = "confirmed"
RESPONSE_CANCELLED: Final = "cancelled"
RESPONSE_NO_RESPONSE: Final = "no_response"
RESPONSE_NOT_YET_REMINDED: Final = "not_yet_reminded"

RESPONSES: Final[frozenset[str]] = frozenset(
    {
        RESPONSE_CONFIRMED,
        RESPONSE_CANCELLED,
        RESPONSE_NO_RESPONSE,
        RESPONSE_NOT_YET_REMINDED,
    }
)


@dataclass(frozen=True, slots=True)
class FollowUpRow:
    appointment_id: uuid.UUID
    patient_id: uuid.UUID
    patient_name: str
    starts_at: dt.datetime
    doctor_name: str
    response: str
    reminded: bool


def _response_of(appointment: Appointment, *, reminded: bool) -> str:
    """A status plus the reminder flags, as one of four answers."""
    if appointment.status is AppointmentStatus.CONFIRMED:
        return RESPONSE_CONFIRMED
    if appointment.status in (AppointmentStatus.CANCELLED, AppointmentStatus.RESCHEDULED):
        return RESPONSE_CANCELLED
    return RESPONSE_NO_RESPONSE if reminded else RESPONSE_NOT_YET_REMINDED


async def follow_up(
    session: AsyncSession,
    filters: ReportFilter,
    *,
    response: str | None = None,
    limit: int = 200,
) -> tuple[FollowUpRow, ...]:
    """Who confirmed, cancelled, or has not answered.

    The third category is the point. A `scheduled` appointment means one of
    two different things -- "we reminded them and they went quiet" or "we have
    not reminded them yet" -- and only the first is a phone call. The reminder
    flags separate them, which is why this report works before M4's channels
    exist at all.
    """
    planned_start = func.lower(Appointment.during)
    statement = (
        select(Appointment, Doctor.full_name, Patient)
        .join(Doctor, Doctor.id == Appointment.doctor_id)
        .join(Patient, Patient.id == Appointment.patient_id)
        .where(
            Appointment.clinic_id == filters.clinic_id,
            Patient.deleted_at.is_(None),
            func.date(planned_start) >= filters.start_date,
            func.date(planned_start) <= filters.end_date,
        )
        .order_by(planned_start)
        .limit(limit)
    )
    if filters.doctor_id is not None:
        statement = statement.where(Appointment.doctor_id == filters.doctor_id)
    if filters.location_id is not None:
        statement = statement.where(Appointment.location_id == filters.location_id)

    out: list[FollowUpRow] = []
    for appointment, doctor_name, patient in (await session.execute(statement)).all():
        reminded = bool(appointment.reminder_48h_sent or appointment.reminder_24h_sent)
        state = _response_of(appointment, reminded=reminded)
        if response is not None and state != response:
            continue
        out.append(
            FollowUpRow(
                appointment_id=appointment.id,
                patient_id=patient.id,
                patient_name=f"{patient.given_names} {patient.family_names}".strip(),
                starts_at=appointment.during.lower,
                doctor_name=doctor_name,
                response=state,
                reminded=reminded,
            )
        )

    # One audit entry per patient named (rule 8). This report returns patient
    # names, so it is a disclosure however aggregated the screen looks.
    if out:
        await record_accesses(
            session,
            AccessAction.READ,
            [
                Access(
                    resource="appointments",
                    resource_id=str(row.appointment_id),
                    patient_id=row.patient_id,
                )
                for row in out
            ],
        )
    return tuple(out)
