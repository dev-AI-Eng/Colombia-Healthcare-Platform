"""Attendance actions and management reports over HTTP (M4).

Two groups of routes, both thin shells over `src.scheduling`:

**Attendance** — the Check-in / Start / End / No-show buttons Rev3 names.
Every refusal maps to a status code a screen can act on rather than to a
generic 400: a stale screen, a double click and the buttons pressed out of
order are three different things a receptionist needs told apart.

**Reports** — the summary panel, the geographic no-show view, doctor
punctuality, and the follow-up list the client asked for. Masking is decided
here rather than in `reporting.py` (ADR-13), because a reporting module that
also masked would be two places deciding one thing.

Every report view is audit-logged. Rev3 requires it, and the reason is that a
report *is* patient data in aggregate: a no-show list by neighbourhood is a
disclosure even though no single row is a patient record.

The clinic-admin role gate is M5's. Until then the whole `/review` surface is
loopback-only in synthetic-data mode, which is what makes these routes safe to
exist at all — see `review/__init__.py`.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field, model_validator

from src.api.dependencies import ClinicScopeDep, SessionDep
from src.api.review.auth import RoleDep
from src.api.review.live import TOPIC_APPOINTMENTS, notify
from src.audit.context import AccessAction
from src.audit.service import Access, record_access
from src.core.timezones import BOGOTA
from src.scheduling import attendance, reporting
from src.scheduling.attendance import Refusal, Step
from src.scheduling.reporting import FORMULAS, MIN_GROUP_SIZE, ReportFilter

router = APIRouter()

#: Refusal to status code. Each one is a different thing to tell a
#: receptionist, so each gets its own code rather than a shared 400.
_REFUSAL_STATUS: dict[Refusal, int] = {
    Refusal.NOT_FOUND: status.HTTP_404_NOT_FOUND,
    # The appointment exists but is cancelled, completed or missed. A conflict
    # rather than a not-found: the row is there, the action is not available.
    Refusal.NOT_ACTIVE: status.HTTP_409_CONFLICT,
    Refusal.OUT_OF_ORDER: status.HTTP_409_CONFLICT,
    # Already recorded. Also a conflict, and deliberately not an error the
    # screen shows in red: pressing twice is not a mistake worth alarming
    # anybody about.
    Refusal.ALREADY_RECORDED: status.HTTP_409_CONFLICT,
}

_REFUSAL_DETAIL: dict[Refusal, str] = {
    Refusal.NOT_FOUND: "No existe una cita con ese id en esta clínica.",
    Refusal.NOT_ACTIVE: "Esa cita ya está cancelada, completada o marcada como no asistida.",
    Refusal.OUT_OF_ORDER: "Falta un paso anterior: registre primero la llegada del paciente.",
    Refusal.ALREADY_RECORDED: "Ese paso ya estaba registrado.",
}


class AttendanceOut(BaseModel):
    """What was recorded, so a screen can show it without re-reading."""

    appointment_id: uuid.UUID
    status: str
    checked_in_at: dt.datetime | None
    started_at: dt.datetime | None
    ended_at: dt.datetime | None


class RangeIn(BaseModel):
    """A date range and the optional narrowing, as query parameters.

    Validated as a pair: a range whose end precedes its start is a mistake
    worth refusing rather than silently returning nothing, because an empty
    report reads as "no appointments" instead of "you asked wrongly".
    """

    # Defaulted to the last 30 days rather than required. A manager opening
    # the panel should see this month without choosing dates first, and a
    # required range means every caller -- including the audit-coverage test --
    # has to know the shape before it can ask anything at all.
    start_date: dt.date = Field(
        default_factory=lambda: dt.datetime.now(tz=BOGOTA).date() - dt.timedelta(days=30)
    )
    end_date: dt.date = Field(default_factory=lambda: dt.datetime.now(tz=BOGOTA).date())
    doctor_id: uuid.UUID | None = None
    location_id: uuid.UUID | None = None

    @model_validator(mode="after")
    def _ordered(self) -> RangeIn:
        if self.end_date < self.start_date:
            raise ValueError("end_date must not precede start_date")
        return self


class FollowUpIn(RangeIn):
    """The follow-up list's filters.

    Extends `RangeIn` rather than sitting beside it as separate parameters:
    FastAPI stops expanding a flattened model's fields when a sibling query
    parameter is declared next to it, and then demands `filters` itself as one
    opaque argument.
    """

    response: str | None = None
    limit: int = Field(default=200, ge=1, le=500)


class FunnelOut(BaseModel):
    offered: int
    confirmed: int
    rescheduled: int
    cancelled: int
    no_show: int
    completed: int
    scheduled: int


class CancellationOut(BaseModel):
    reason: str
    actor: str
    count: int


class SummaryOut(BaseModel):
    """The summary panel.

    `formulas` travels with the numbers rather than living in the frontend,
    because Rev3 requires each formula printed on the report and a copy in the
    UI is a copy that can drift from the SQL.
    """

    funnel: FunnelOut
    percentages: dict[str, float | None]
    formulas: dict[str, str]
    cancellations: list[CancellationOut]


class ZoneOut(BaseModel):
    municipality_divipola: str | None
    neighborhood: str | None
    appointments: int
    no_shows: int
    no_show_pct: float | None
    suppressed: bool


class ZoneReportOut(BaseModel):
    """The geographic view, with its suppression stated.

    `suppressed_groups` and `minimum_group_size` are returned so the screen can
    say "three groups hidden because they were too small". A reader who cannot
    see that data was withheld reads the rest as complete.
    """

    rows: list[ZoneOut]
    suppressed_groups: int
    minimum_group_size: int


class PunctualityOut(BaseModel):
    doctor_id: uuid.UUID
    doctor_name: str
    total: int
    recorded_start: int
    recorded_end: int
    mean_start_delay_minutes: float | None
    mean_overrun_minutes: float | None
    start_coverage_pct: float | None
    end_coverage_pct: float | None


class FollowUpOut(BaseModel):
    """One appointment needing a decision, masked as /review/patients is.

    The client's request: a list staff can work through. `response` is the
    column that matters -- "no response" is the follow-up list, and it is only
    answerable because the appointment records whether a reminder was sent.
    """

    appointment_id: uuid.UUID
    patient_id: uuid.UUID
    patient_name: str
    starts_at: dt.datetime
    doctor_name: str
    response: str
    reminded: bool


async def _audit_report(
    session: SessionDep, *, clinic_id: uuid.UUID, view: str, filters: RangeIn
) -> None:
    """Record that a report was viewed.

    A report is patient data in aggregate: a no-show table by neighbourhood
    discloses something about the people in it even though no row is a patient
    record. The entry names the view and its filters rather than a patient,
    because there is no single patient to name.
    """
    await record_access(
        session,
        AccessAction.READ,
        Access(
            resource=f"reports/{view}",
            resource_id=f"{filters.start_date}..{filters.end_date}",
            patient_id=None,
        ),
    )


def _to_filter(clinic_id: uuid.UUID, body: RangeIn) -> ReportFilter:
    return ReportFilter(
        clinic_id=clinic_id,
        start_date=body.start_date,
        end_date=body.end_date,
        doctor_id=body.doctor_id,
        location_id=body.location_id,
    )


# ------------------------------------------------------------- attendance


@router.post(
    "/appointments/{appointment_id}/attendance/{step}",
    response_model=AttendanceOut,
    summary="Record check-in, start, end or no-show",
    responses={
        404: {"description": "No appointment with that id in this clinic."},
        409: {"description": "Already recorded, out of order, or no longer active."},
    },
)
async def record_attendance(
    session: SessionDep,
    scope: ClinicScopeDep,
    appointment_id: uuid.UUID,
    step: Step,
) -> AttendanceOut:
    """One button press.

    The time is the server's, not the caller's. A client-supplied timestamp
    would let a late entry be backdated, and the punctuality report exists
    precisely to measure lateness.
    """
    result = await attendance.record(
        session,
        clinic_id=scope.clinic_id,
        appointment_id=appointment_id,
        step=step,
        now=dt.datetime.now(tz=dt.UTC),
    )
    if not result.ok or result.appointment is None:
        refusal = result.refusal or Refusal.NOT_FOUND
        raise HTTPException(_REFUSAL_STATUS[refusal], _REFUSAL_DETAIL[refusal])

    row = result.appointment
    # Check-in, start, end and no-show all change what the follow-up list and
    # the funnel show, so an open dashboard is told.
    notify(scope.clinic_id, TOPIC_APPOINTMENTS)
    return AttendanceOut(
        appointment_id=row.id,
        status=row.status.value,
        checked_in_at=row.checked_in_at,
        started_at=row.started_at,
        ended_at=row.ended_at,
    )


# ---------------------------------------------------------------- reports


@router.get(
    "/reports/summary",
    response_model=SummaryOut,
    summary="Appointment funnel and rates for a date range",
)
async def report_summary(
    session: SessionDep,
    scope: ClinicScopeDep,
    filters: Annotated[RangeIn, Query()],
) -> SummaryOut:
    """The funnel, the four rates, and the formula behind each."""
    await _audit_report(session, clinic_id=scope.clinic_id, view="summary", filters=filters)
    report = await reporting.summary(session, _to_filter(scope.clinic_id, filters))

    return SummaryOut(
        funnel=FunnelOut(
            offered=report.funnel.offered,
            confirmed=report.funnel.confirmed,
            rescheduled=report.funnel.rescheduled,
            cancelled=report.funnel.cancelled,
            no_show=report.funnel.no_show,
            completed=report.funnel.completed,
            scheduled=report.funnel.scheduled,
        ),
        percentages=report.percentages,
        formulas=dict(FORMULAS),
        cancellations=[
            CancellationOut(reason=c.reason, actor=c.actor, count=c.count)
            for c in report.cancellations
        ],
    )


@router.get(
    "/reports/zones",
    response_model=ZoneReportOut,
    summary="No-show rate by municipality and neighbourhood",
)
async def report_zones(
    session: SessionDep,
    scope: ClinicScopeDep,
    filters: Annotated[RangeIn, Query()],
) -> ZoneReportOut:
    """The geographic view, with small groups suppressed.

    No map and no distance calculation, per Rev3. A map of patients by
    neighbourhood is a re-identification surface, and the clinic's question is
    "where are we losing people", which a table answers.
    """
    await _audit_report(session, clinic_id=scope.clinic_id, view="zones", filters=filters)
    rows = await reporting.zones(session, _to_filter(scope.clinic_id, filters))

    return ZoneReportOut(
        rows=[
            ZoneOut(
                municipality_divipola=row.municipality_divipola,
                neighborhood=row.neighborhood,
                appointments=row.appointments,
                no_shows=row.no_shows,
                no_show_pct=row.no_show_pct,
                suppressed=row.suppressed,
            )
            for row in rows
        ],
        suppressed_groups=sum(1 for row in rows if row.suppressed),
        minimum_group_size=MIN_GROUP_SIZE,
    )


@router.get(
    "/reports/punctuality",
    response_model=list[PunctualityOut],
    summary="Start delay and overrun per doctor, with recording coverage",
)
async def report_punctuality(
    session: SessionDep,
    scope: ClinicScopeDep,
    filters: Annotated[RangeIn, Query()],
) -> list[PunctualityOut]:
    """Delay and overrun, and how much of the data they rest on.

    The coverage is not decoration. A clinic that records start times for four
    appointments in ninety produces a mean that describes those four, and a
    manager shown the mean without the coverage will read it as the doctor.
    """
    await _audit_report(session, clinic_id=scope.clinic_id, view="punctuality", filters=filters)
    rows = await reporting.punctuality(session, _to_filter(scope.clinic_id, filters))

    return [
        PunctualityOut(
            doctor_id=row.doctor_id,
            doctor_name=row.doctor_name,
            total=row.total,
            recorded_start=row.recorded_start,
            recorded_end=row.recorded_end,
            mean_start_delay_minutes=row.mean_start_delay_minutes,
            mean_overrun_minutes=row.mean_overrun_minutes,
            start_coverage_pct=row.start_coverage_pct,
            end_coverage_pct=row.end_coverage_pct,
        )
        for row in rows
    ]


@router.get(
    "/reports/follow-up",
    response_model=list[FollowUpOut],
    summary="Who confirmed, cancelled, or has not answered",
)
async def report_follow_up(
    session: SessionDep,
    scope: ClinicScopeDep,
    role: RoleDep,
    filters: Annotated[FollowUpIn, Query()],
) -> list[FollowUpOut]:
    """The follow-up list.

    `no_response` is the one that fills slots, and it is only answerable
    because the appointment records whether a reminder went out: a `scheduled`
    appointment might mean "we asked and they went quiet" or "we have not asked
    yet", and those need different actions from a receptionist.

    Names are shown; identifiers are not. Staff ringing a patient need the name
    to know who they are calling, and the number comes from the patient's own
    record where unmasking it is audited as a disclosure (ADR-13).
    """
    await _audit_report(session, clinic_id=scope.clinic_id, view="follow-up", filters=filters)
    rows = await reporting.follow_up(
        session,
        _to_filter(scope.clinic_id, filters),
        response=filters.response,
        limit=filters.limit,
    )
    return [
        FollowUpOut(
            appointment_id=row.appointment_id,
            patient_id=row.patient_id,
            patient_name=row.patient_name,
            starts_at=row.starts_at,
            doctor_name=row.doctor_name,
            response=row.response,
            reminded=row.reminded,
        )
        for row in rows
    ]
