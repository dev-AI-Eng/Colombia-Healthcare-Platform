"""Synthetic Colombian clinic data for development and review (ADR-11).

Everything generated here is fictional. Names and emails come from Faker's
`es_CO` locale; document and phone numbers come from a `random.Random`. Both use
a fixed seed, so these values repeat for the same parameters. Birth dates are
relative to the current date, and availability and appointment dates to the
`today` argument (the current Bogotá date by default).

Generated data:
* one clinic with one location and one 20-minute appointment type;
* doctors cycling through eight specialties, each available Monday to Friday
  08:00-12:00;
* patients, each with a phone binding to their own number and an
  appointment-messaging consent;
* one handset shared by the first three patients, bound as `caregiver`, to
  exercise the shared-phone case (ADR-18);
* twelve consecutive 20-minute appointments per doctor, starting at 08:00
  Bogotá time on the next weekday after the seeding date, each assigned to a
  randomly chosen patient.

Every created patient, binding, consent and appointment is recorded in the
audit log as a `create` access.

Run through `python main.py` (automatic when the database is empty) or directly:

    python -m src.devdata.seed [--patients 200] [--doctors 8]
"""

from __future__ import annotations

import argparse
import asyncio
import random
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Final

from faker import Faker
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import Range
from sqlalchemy.ext.asyncio import AsyncSession

from src.audit.context import AccessAction, ActorKind, AuditContext, audit_context
from src.audit.service import Access, record_accesses
from src.core.config import get_settings
from src.core.crypto import blind_index
from src.core.db import configure_event_loop_policy, dispose_engine, get_sessionmaker
from src.core.tenancy import ClinicScope, apply_clinic_scope
from src.core.timezones import BOGOTA
from src.identity.models import (
    BindingRelationship,
    Consent,
    ConsentChannel,
    ConsentPurpose,
    EvidenceKind,
    PhoneBinding,
)
from src.registry.models import Clinic, Doctor, DocumentType, Location, Patient
from src.scheduling.models import (
    Appointment,
    AppointmentStatus,
    AppointmentType,
    AvailabilityRule,
)

RANDOM_SEED = 20260917
POLICY_VERSION = "synthetic-v1"
SPECIALTIES = (
    "Medicina General",
    "Pediatría",
    "Ginecología",
    "Ortopedia",
    "Dermatología",
    "Cardiología",
    "Oftalmología",
    "Urología",
)
# Weighted toward the cédula de ciudadanía, the most common adult document.
DOCUMENT_TYPES: tuple[DocumentType, ...] = (
    DocumentType.CC,
    DocumentType.TI,
    DocumentType.CE,
    DocumentType.PT,
    DocumentType.RC,
)
DOCUMENT_TYPE_WEIGHTS: tuple[int, ...] = (70, 12, 8, 6, 4)
APPOINTMENT_MINUTES = 20
APPOINTMENTS_PER_DOCTOR = 12
SHARED_HANDSET_PATIENTS = 3


@dataclass(frozen=True, slots=True)
class SeedReport:
    clinic_id: uuid.UUID
    doctors: int
    patients: int
    appointments: int


#: Colombian mobile prefixes that actually exist, from libphonenumber's own
#: metadata (`is_valid_number` accepts these and rejects the rest of 3xx).
#:
#: The generator used to be "+573 then nine random digits", which produced
#: numbers like +573954336730 — possible-looking and invalid, because the 39x
#: and 33x ranges are not assigned. Nothing noticed until the patient-facing
#: cancel route began validating the number it was given: a seeded patient
#: could not prove who they were, because their own number was not real. The
#: same reasoning as CLAUDE.md §11 — libphonenumber's metadata tracks the 2021
#: renumbering and a hand-rolled rule does not — applies to generating a
#: number as much as to checking one.
#: 324 and 333 are deliberately absent although libphonenumber accepts them:
#: only part of each range is assigned, so random digits inside them produce an
#: invalid number a few percent of the time. Twenty-two fully-assigned prefixes
#: is ample variety for synthetic data, and modelling sub-ranges here would be
#: re-implementing the metadata this list exists to respect.
_MOBILE_PREFIXES: Final = (
    300, 301, 302, 303, 304, 305,
    310, 311, 312, 313, 314, 315, 316, 317, 318, 319,
    320, 321, 322, 323,
    350, 351,
)  # fmt: skip


def _colombian_mobile(rng: random.Random) -> str:
    """A valid Colombian mobile number in E.164.

    Valid, not merely well-shaped: `phonenumbers.is_valid_number` accepts every
    number this returns, which is what the import path and the patient-facing
    routes both require.
    """
    prefix = rng.choice(_MOBILE_PREFIXES)
    return f"+57{prefix}" + "".join(str(rng.randrange(10)) for _ in range(7))


def _next_weekday(today: date) -> date:
    day = today + timedelta(days=1)
    while day.weekday() >= 5:  # Saturday or Sunday
        day += timedelta(days=1)
    return day


async def _create_patients(
    session: AsyncSession, clinic: Clinic, count: int, fake: Faker, rng: random.Random
) -> list[Patient]:
    patients: list[Patient] = []
    used_numbers: set[str] = set()
    for _ in range(count):
        document_number = str(rng.randrange(10_000_000, 2_000_000_000))
        while document_number in used_numbers:
            document_number = str(rng.randrange(10_000_000, 2_000_000_000))
        used_numbers.add(document_number)
        phone = _colombian_mobile(rng)
        patients.append(
            Patient(
                clinic_id=clinic.id,
                document_type=rng.choices(DOCUMENT_TYPES, DOCUMENT_TYPE_WEIGHTS)[0],
                document_number=document_number,
                document_number_bidx=blind_index(document_number),
                given_names=fake.first_name(),
                family_names=f"{fake.last_name()} {fake.last_name()}",
                birth_date=fake.date_of_birth(minimum_age=1, maximum_age=92),
                phone_e164=phone,
                phone_e164_bidx=blind_index(phone),
                email=fake.email(),
            )
        )
    session.add_all(patients)
    await session.flush()
    return patients


async def seed_synthetic_data(
    session: AsyncSession, *, patients: int = 200, doctors: int = 8, today: date | None = None
) -> SeedReport | None:
    """Insert synthetic data into an empty database. Returns None if a clinic exists.

    `today` anchors availability and appointment dates; it defaults to the current
    Bogotá date. Must run inside a bound `AuditContext`. The caller commits.
    """
    if await session.scalar(select(Clinic.id).limit(1)) is not None:
        return None

    fake = Faker("es_CO")
    fake.seed_instance(RANDOM_SEED)
    rng = random.Random(RANDOM_SEED)
    now = datetime.now(BOGOTA)
    anchor = today or now.date()

    # Foreign keys have no ORM relationships, so parents are flushed before
    # children to give the unit of work a valid insert order.
    clinic = Clinic(name="Clínica Demo Bogotá")
    session.add(clinic)
    await session.flush()

    # Every table below is under row-level security. Without a scope the clinic
    # policies reject each insert, so the scope is bound before the first child
    # row and lasts for the rest of this transaction.
    await apply_clinic_scope(session, ClinicScope(clinic_id=clinic.id))

    location = Location(
        clinic_id=clinic.id, name="Sede Principal", address="Calle 100 # 15-20, Bogotá"
    )
    appointment_type = AppointmentType(
        clinic_id=clinic.id, name="Consulta general", duration_minutes=APPOINTMENT_MINUTES
    )
    doctor_rows = [
        Doctor(
            clinic_id=clinic.id,
            full_name=f"Dr(a). {fake.first_name()} {fake.last_name()} {fake.last_name()}",
            specialty=SPECIALTIES[index % len(SPECIALTIES)],
        )
        for index in range(doctors)
    ]
    session.add_all([location, appointment_type, *doctor_rows])
    await session.flush()

    session.add_all(
        AvailabilityRule(
            clinic_id=clinic.id,
            doctor_id=doctor.id,
            location_id=location.id,
            weekday=weekday,
            start_time=time(8, 0),
            end_time=time(12, 0),
            valid_from=anchor,
        )
        for doctor in doctor_rows
        for weekday in range(5)
    )

    patient_rows = await _create_patients(session, clinic, patients, fake, rng)

    bindings = [
        PhoneBinding(
            clinic_id=clinic.id,
            phone_e164_bidx=patient.phone_e164_bidx,
            patient_id=patient.id,
            relationship_kind=BindingRelationship.SELF,
            verified_at=now,
        )
        for patient in patient_rows
        if patient.phone_e164_bidx is not None
    ]
    shared_number_index = blind_index(_colombian_mobile(rng))
    bindings += [
        PhoneBinding(
            clinic_id=clinic.id,
            phone_e164_bidx=shared_number_index,
            patient_id=patient.id,
            relationship_kind=BindingRelationship.CAREGIVER,
            verified_at=now,
        )
        for patient in patient_rows[:SHARED_HANDSET_PATIENTS]
    ]
    consents = [
        Consent(
            clinic_id=clinic.id,
            patient_id=patient.id,
            purpose=ConsentPurpose.APPOINTMENT_MESSAGING,
            channel=ConsentChannel.WHATSAPP,
            granted_at=now,
            evidence_kind=EvidenceKind.IMPORTED_DECLARATION,
            policy_version=POLICY_VERSION,
        )
        for patient in patient_rows
    ]

    first_start = datetime.combine(_next_weekday(anchor), time(8, 0), tzinfo=BOGOTA)
    appointments = [
        Appointment(
            clinic_id=clinic.id,
            patient_id=rng.choice(patient_rows).id,
            doctor_id=doctor.id,
            location_id=location.id,
            appointment_type_id=appointment_type.id,
            during=Range(start, start + timedelta(minutes=APPOINTMENT_MINUTES), bounds="[)"),
            status=AppointmentStatus.SCHEDULED,
        )
        for doctor in doctor_rows
        for start in (
            first_start + timedelta(minutes=APPOINTMENT_MINUTES * slot)
            for slot in range(APPOINTMENTS_PER_DOCTOR)
        )
    ]
    session.add_all([*bindings, *consents, *appointments])
    await session.flush()

    await record_accesses(
        session,
        AccessAction.CREATE,
        [Access(resource="patients", resource_id=str(p.id), patient_id=p.id) for p in patient_rows]
        + [
            Access(resource="phone_bindings", resource_id=str(b.id), patient_id=b.patient_id)
            for b in bindings
        ]
        + [
            Access(resource="consents", resource_id=str(c.id), patient_id=c.patient_id)
            for c in consents
        ]
        + [
            Access(resource="appointments", resource_id=str(a.id), patient_id=a.patient_id)
            for a in appointments
        ],
    )
    return SeedReport(
        clinic_id=clinic.id,
        doctors=len(doctor_rows),
        patients=len(patient_rows),
        appointments=len(appointments),
    )


async def run_seed(*, patients: int = 200, doctors: int = 8) -> SeedReport | None:
    """Seed in its own transaction as the runtime role.

    Allowed in any environment while ALLOW_REAL_PATIENT_DATA is false, and refused
    once it is true, so synthetic records never sit beside real ones.
    """
    settings = get_settings()
    if not settings.synthetic_seeding_allowed:
        raise RuntimeError(
            "Synthetic seeding is refused while ALLOW_REAL_PATIENT_DATA is true: "
            "synthetic records must never be mixed with real patient data."
        )
    context = AuditContext(actor_kind=ActorKind.SYSTEM, purpose="synthetic_seed", actor_id="seeder")
    try:
        with audit_context(context):
            async with get_sessionmaker()() as session, session.begin():
                return await seed_synthetic_data(session, patients=patients, doctors=doctors)
    finally:
        await dispose_engine()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed synthetic clinic data into an empty database"
    )
    parser.add_argument("--patients", type=int, default=200)
    parser.add_argument("--doctors", type=int, default=8)
    args = parser.parse_args()
    configure_event_loop_policy()
    report = asyncio.run(run_seed(patients=args.patients, doctors=args.doctors))
    if report is None:
        print("Database already contains a clinic; nothing seeded.")
    else:
        print(
            f"Seeded {report.doctors} doctors, {report.patients} patients and "
            f"{report.appointments} appointments. All data is synthetic."
        )


if __name__ == "__main__":
    main()
