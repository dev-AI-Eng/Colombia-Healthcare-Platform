"""Reading and writing an import, and applying it to the clinic's tables.

Two halves, with different rules.

The **staging** half writes to the `onboarding` schema. Those rows are a copy of
the clinic's file plus what our rules made of it, so they hold cédulas, names
and phone numbers. Reading them back is therefore a patient-data read and is
audited like any other: rule 8 is about the values, not about whether a person
has accepted them yet. The tables are clinic-scoped and behind row-level
security for the same reason.

Writing them is not audited. A staged row is this import's own working copy and
every value in it came from the file the clinic just uploaded; the upload itself
is the event worth recording, and an entry per staged cell would bury the
disclosures that matter in a log nobody can read.

The **apply** half writes to `app`, and every row it creates or updates is a
patient-data write, so it records an audit entry through the same helper every
other repository uses. It runs inside the caller's transaction: either the whole
import lands or none of it does.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import uuid
from dataclasses import dataclass
from typing import Any, Final

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.audit.context import AccessAction
from src.audit.service import Access, record_accesses
from src.core.crypto import blind_index
from src.core.timezones import BOGOTA
from src.identity.models import Consent, ConsentChannel, ConsentPurpose, EvidenceKind
from src.onboarding.canonical import Entity
from src.onboarding.matcher import normalize_header
from src.onboarding.models import ImportProfile, ImportSession, StagingRow, TransformLogEntry
from src.onboarding.service import RowResult
from src.registry.models import Doctor, DocumentType, Patient, Specialty


def header_fingerprint(headers: tuple[str, ...], entity: str) -> str:
    """Identify "a sheet of this kind, shaped like this one".

    Headers are normalized and sorted before hashing, so a clinic that reorders
    its columns, changes their capitalisation or adds an accent still matches
    the profile a person already confirmed. A file with genuinely different
    columns gets a different fingerprint and is reviewed afresh: an exact set is
    the point, because a near-miss key would apply a 12-column mapping to a
    13-column file and map the new column to nothing without saying so.

    `entity` is part of the key because headers alone do not identify a sheet.
    A workbook whose Doctores and Especialidades sheets both read
    ("Nombre", "Codigo") produced one fingerprint for both: the second overwrote
    the first on commit, and the next upload applied the doctor mapping *and the
    doctor entity* to the specialties sheet, pre-ticked "confirmed" with nothing
    missing, so specialties were staged and written as doctors with no refusal
    anywhere. Keying on the kind of sheet as well means one shape can carry one
    profile per entity, and a lookup for a sheet of kind X can never return a
    profile confirmed for kind Y.
    """
    normalized = sorted(normalize_header(h) for h in headers if h.strip())
    return hashlib.sha256("\u001f".join([entity, *normalized]).encode("utf-8")).hexdigest()


# ----------------------------------------------------------------- sessions
async def create_session(
    session: AsyncSession,
    *,
    clinic_id: uuid.UUID,
    filename: str,
    file_size: int,
    file_sha256: str,
    report: dict[str, Any],
    actor_id: str | None = None,
) -> ImportSession:
    record = ImportSession(
        clinic_id=clinic_id,
        filename=filename,
        file_size=file_size,
        file_sha256=file_sha256,
        status="analyzed",
        report=report,
        actor_id=actor_id,
    )
    session.add(record)
    await session.flush()
    return record


async def get_session(
    session: AsyncSession, *, clinic_id: uuid.UUID, session_id: uuid.UUID
) -> ImportSession | None:
    found: ImportSession | None = await session.scalar(
        select(ImportSession).where(
            ImportSession.id == session_id, ImportSession.clinic_id == clinic_id
        )
    )
    return found


async def list_sessions(
    session: AsyncSession, *, clinic_id: uuid.UUID, limit: int = 50
) -> list[ImportSession]:
    result = await session.scalars(
        select(ImportSession)
        .where(ImportSession.clinic_id == clinic_id)
        .order_by(ImportSession.started_at.desc())
        .limit(limit)
    )
    return list(result.all())


async def find_previous_import(
    session: AsyncSession, *, clinic_id: uuid.UUID, file_sha256: str
) -> ImportSession | None:
    """A byte-identical file this clinic already committed.

    Re-uploading the same export by accident is common; saying so is kinder than
    importing it twice and leaving someone to work out which rows duplicated.
    """
    found: ImportSession | None = await session.scalar(
        select(ImportSession)
        .where(
            ImportSession.clinic_id == clinic_id,
            ImportSession.file_sha256 == file_sha256,
            ImportSession.status == "committed",
        )
        .order_by(ImportSession.started_at.desc())
    )
    return found


# ------------------------------------------------------------------ staging
async def replace_staging(
    session: AsyncSession,
    *,
    clinic_id: uuid.UUID,
    session_id: uuid.UUID,
    sheet: str,
    entity: Entity,
    rows: list[RowResult],
) -> None:
    """Write this sheet's staged rows and transform log, replacing any earlier run.

    Validation may be run more than once — a reviewer corrects the mapping and
    tries again — so the previous attempt is cleared rather than added to.
    """
    # Both tables are cleared for this sheet only. The log used to be cleared
    # for the whole session, so validating a second sheet erased the first
    # sheet's evidence.
    for table in (StagingRow, TransformLogEntry):
        await session.execute(
            delete(table).where(
                table.session_id == session_id,
                table.clinic_id == clinic_id,
                table.sheet == sheet,
            )
        )

    for row in rows:
        session.add(
            StagingRow(
                clinic_id=clinic_id,
                session_id=session_id,
                entity=entity.value,
                sheet=sheet,
                row_number=row.row_number,
                raw=row.raw,
                normalized={k: _jsonable(v) for k, v in row.values.items()},
                status=row.status.value,
                errors={"errors": row.errors, "reviews": row.reviews}
                if (row.errors or row.reviews)
                else None,
            )
        )
        for cell in row.cells:
            session.add(
                TransformLogEntry(
                    clinic_id=clinic_id,
                    session_id=session_id,
                    sheet=sheet,
                    row_number=cell.row_number,
                    column_name=cell.column,
                    target_field=cell.target_field,
                    raw_value=cell.raw,
                    corrected_from_review=cell.corrected_from_review,
                    normalized_value=cell.normalized,
                    rule=cell.rule,
                    status=cell.status.value,
                    message=cell.message or None,
                )
            )
    await session.flush()


async def staged_rows(
    session: AsyncSession,
    *,
    clinic_id: uuid.UUID,
    session_id: uuid.UUID,
    sheet: str | None = None,
    status: str | None = None,
    limit: int = 200,
) -> list[StagingRow]:
    statement = select(StagingRow).where(
        StagingRow.clinic_id == clinic_id, StagingRow.session_id == session_id
    )
    if sheet:
        statement = statement.where(StagingRow.sheet == sheet)
    if status:
        statement = statement.where(StagingRow.status == status)
    result = await session.scalars(statement.order_by(StagingRow.row_number).limit(limit))
    rows = list(result.all())
    await _record_staging_reads(
        session,
        clinic_id=clinic_id,
        session_id=session_id,
        resource="staging_rows",
        count=len(rows),
    )
    return rows


async def _record_staging_reads(
    session: AsyncSession,
    *,
    clinic_id: uuid.UUID,
    session_id: uuid.UUID,
    resource: str,
    count: int,
) -> None:
    """Record that staged patient values were served.

    One entry per request rather than per row: the staging tables are keyed by
    import session, not by patient, so there is no patient id to attribute a
    row to until it is applied. What the log needs to answer is "who read this
    import's contents, and when", and the session id answers it.
    """
    if not count:
        return
    await record_accesses(
        session,
        AccessAction.READ,
        [
            Access(
                resource=resource,
                resource_id=str(session_id),
                # No patient id: a staged row is not yet attributed to a
                # patient record, and inventing one would be a wrong link.
                patient_id=None,
            )
        ],
    )


async def transform_log(
    session: AsyncSession,
    *,
    clinic_id: uuid.UUID,
    session_id: uuid.UUID,
    status: str | None = None,
    limit: int = 500,
) -> list[TransformLogEntry]:
    statement = select(TransformLogEntry).where(
        TransformLogEntry.clinic_id == clinic_id, TransformLogEntry.session_id == session_id
    )
    if status:
        statement = statement.where(TransformLogEntry.status == status)
    result = await session.scalars(
        statement.order_by(TransformLogEntry.row_number, TransformLogEntry.id).limit(limit)
    )
    entries = list(result.all())
    await _record_staging_reads(
        session,
        clinic_id=clinic_id,
        session_id=session_id,
        resource="transform_log",
        count=len(entries),
    )
    return entries


# ----------------------------------------------------------------- profiles
async def save_profile(
    session: AsyncSession,
    *,
    clinic_id: uuid.UUID,
    name: str,
    fingerprint: str,
    mapping: dict[str, Any],
) -> ImportProfile:
    """Remember a confirmed mapping so a file of this shape is not re-reviewed.

    The mapping only: never a rule derived from the values in the file that
    happened to be uploaded (ADR-08a).
    """
    existing = await session.scalar(
        select(ImportProfile).where(
            ImportProfile.clinic_id == clinic_id,
            ImportProfile.header_fingerprint == fingerprint,
        )
    )
    if existing is not None:
        existing.mapping = mapping
        existing.name = name
        existing.version += 1
        existing.updated_at = dt.datetime.now(dt.UTC)
        await session.flush()
        return existing

    profile = ImportProfile(
        clinic_id=clinic_id, name=name, header_fingerprint=fingerprint, mapping=mapping
    )
    session.add(profile)
    await session.flush()
    return profile


async def find_profiles_for_shape(
    session: AsyncSession, *, clinic_id: uuid.UUID, headers: tuple[str, ...]
) -> list[ImportProfile]:
    """Every profile this clinic has confirmed for a sheet shaped like this one.

    A list, not one row, because the entity is part of the key: a workbook whose
    Doctores and Especialidades sheets share headers has one profile each, and
    returning either of them on its own is how a sheet of specialties was
    relabelled and written as doctors.

    The caller decides what to do with more than one. It cannot be settled here:
    a profile holds the entity a person *confirmed*, which outranks the guess we
    arrive with, so the choice is only safe where the alternatives are visible.
    """
    candidates = [header_fingerprint(headers, entity.value) for entity in Entity]
    result = await session.scalars(
        select(ImportProfile).where(
            ImportProfile.clinic_id == clinic_id,
            ImportProfile.header_fingerprint.in_(candidates),
        )
    )
    return list(result.all())


async def list_profiles(session: AsyncSession, *, clinic_id: uuid.UUID) -> list[ImportProfile]:
    result = await session.scalars(
        select(ImportProfile)
        .where(ImportProfile.clinic_id == clinic_id)
        .order_by(ImportProfile.updated_at.desc())
    )
    return list(result.all())


# -------------------------------------------------------------------- apply
#: Recorded on a consent that arrived in a spreadsheet. A real policy version
#: names the text the patient actually agreed to; an import cannot know it, so it
#: says so rather than claiming a version we did not show them.
IMPORTED_POLICY_VERSION: Final = "imported-unknown"


@dataclass(frozen=True, slots=True)
class ApplyResult:
    created: int = 0
    updated: int = 0
    skipped: int = 0
    conflicts: tuple[str, ...] = ()


async def apply_rows(
    session: AsyncSession,
    *,
    clinic_id: uuid.UUID,
    entity: Entity,
    rows: list[RowResult],
) -> ApplyResult:
    """Write valid rows into the clinic's tables, in the caller's transaction.

    Only rows that converted cleanly are written. A row awaiting review is
    skipped rather than half-written, because a patient with a blank identifier
    is worse than a patient not yet imported.
    """
    match entity:
        case Entity.PATIENT:
            return await _apply_patients(session, clinic_id=clinic_id, rows=rows)
        case Entity.SPECIALTY:
            return await _apply_specialties(session, clinic_id=clinic_id, rows=rows)
        case Entity.DOCTOR:
            return await _apply_doctors(session, clinic_id=clinic_id, rows=rows)
        case Entity.APPOINTMENT:
            # An appointment written as a plain insert goes around the exclusion
            # constraint that prevents double-booking, so it needs the scheduling
            # engine's booking transaction (M2) rather than this path.
            return ApplyResult(
                skipped=len(rows),
                conflicts=(
                    "appointment rows are staged but not applied: they need the "
                    "scheduling engine's booking transaction (M2).",
                ),
            )
        case _:
            # Availability is a plain insert -- `availability_rules` has only
            # CHECK constraints, so no booking transaction is involved. What
            # blocks it is `location_id`, which is NOT NULL on the model and has
            # no canonical field feeding it: a rule has to say which sede it
            # applies to, and whether a clinic has one location or several is an
            # open question with the client. Staged and validated meanwhile, so
            # the rows and their refusals are visible.
            return ApplyResult(
                skipped=len(rows),
                conflicts=(
                    f"{entity.value} rows are staged and validated but not applied: "
                    "each rule needs the location it applies to, which this export "
                    "does not carry.",
                ),
            )


async def _apply_patients(
    session: AsyncSession, *, clinic_id: uuid.UUID, rows: list[RowResult]
) -> ApplyResult:
    created = updated = skipped = 0
    conflicts: list[str] = []
    accesses: list[Access] = []
    overwritten: list[Access] = []

    for row in rows:
        if row.status.value != "valid":
            skipped += 1
            continue
        values = row.values
        document_number = values.get("document_number")
        document_type = values.get("document_type")
        if not document_number or not document_type:
            skipped += 1
            continue

        given, family = _names(values)
        if given is None:
            skipped += 1
            continue

        index = blind_index(str(document_number))
        existing = await session.scalar(
            select(Patient).where(
                Patient.clinic_id == clinic_id,
                Patient.document_type == document_type,
                Patient.document_number_bidx == index,
            )
        )

        if existing is not None:
            if existing.deleted_at is not None:
                # Deletion is usually a privacy request. Reversing it because a
                # stale spreadsheet still lists the person needs a human.
                conflicts.append(
                    f"Row {row.row_number}: this patient was deleted on "
                    f"{existing.deleted_at:%Y-%m-%d}. Confirm before restoring them."
                )
                skipped += 1
                continue
            _assign_patient(existing, values, given, family)
            await _record_consent(
                session, clinic_id=clinic_id, patient_id=existing.id, values=values
            )
            updated += 1
            # An overwrite of a record the clinic already had is an update, not a
            # creation. The commit message tells the receptionist this happened;
            # the audit log has to agree with it.
            overwritten.append(
                Access(resource="patients", resource_id=str(existing.id), patient_id=existing.id)
            )
            continue

        patient = Patient(
            clinic_id=clinic_id,
            document_type=DocumentType(document_type),
            document_number=str(document_number),
            document_number_bidx=index,
            given_names=given,
            family_names=family,
        )
        _assign_patient(patient, values, given, family)
        session.add(patient)
        await session.flush()
        created += 1
        await _record_consent(session, clinic_id=clinic_id, patient_id=patient.id, values=values)
        accesses.append(
            Access(resource="patients", resource_id=str(patient.id), patient_id=patient.id)
        )

    if accesses:
        await record_accesses(session, AccessAction.CREATE, accesses)
    if overwritten:
        await record_accesses(session, AccessAction.UPDATE, overwritten)
    return ApplyResult(created, updated, skipped, tuple(conflicts))


async def _record_consent(
    session: AsyncSession,
    *,
    clinic_id: uuid.UUID,
    patient_id: uuid.UUID,
    values: dict[str, Any],
) -> None:
    """Store a consent the clinic's export carried, if it carried a whole one.

    All three parts are required together. A purpose with no date is not a
    consent record under Ley 1581 -- it is a claim that one exists somewhere --
    and writing it would let the dispatch gate in M4 treat an unproven consent
    as proven. Partial consent columns are therefore ignored here, while the
    per-cell rules have already flagged whichever part was unreadable.

    `channel` is `any` because an export says what the patient agreed to, not
    which app to use; the channel is settled by the verified phone binding.
    Re-importing the same consent does not duplicate it: one row per patient,
    purpose and grant date.
    """
    purpose = values.get("consent_purpose")
    granted_at = values.get("consent_granted_at")
    evidence = values.get("consent_evidence")
    if not purpose or granted_at is None or not evidence:
        return

    moment = (
        granted_at
        if isinstance(granted_at, dt.datetime)
        else dt.datetime.combine(granted_at, dt.time.min, tzinfo=BOGOTA)
    )

    existing = await session.scalar(
        select(Consent).where(
            Consent.clinic_id == clinic_id,
            Consent.patient_id == patient_id,
            Consent.purpose == purpose,
            Consent.granted_at == moment,
        )
    )
    if existing is not None:
        return

    consent = Consent(
        clinic_id=clinic_id,
        patient_id=patient_id,
        purpose=ConsentPurpose(str(purpose)),
        channel=ConsentChannel.ANY,
        granted_at=moment,
        evidence_kind=EvidenceKind(str(evidence)),
        # No evidence_ref: the document itself stayed with the clinic. The
        # evidence_kind says what to go and ask for.
        policy_version=IMPORTED_POLICY_VERSION,
    )
    session.add(consent)
    # Under its own resource, as every other consent write does. The enclosing
    # patient's entry is not a substitute: in M4 the dispatch gate asks when and
    # from where consent was recorded, and the import is the only route by which
    # this row exists. Flushed first so the entry can name the row's id.
    await session.flush()
    await record_accesses(
        session,
        AccessAction.CREATE,
        [Access(resource="consents", resource_id=str(consent.id), patient_id=patient_id)],
    )


def _names(values: dict[str, Any]) -> tuple[str | None, str]:
    """Given names and surnames, however the file supplied them."""
    if (split := values.get("full_name")) is not None and hasattr(split, "given_names"):
        return split.given_names, split.family_names
    given = values.get("given_names")
    family = values.get("family_names") or ""
    return (str(given) if given else None), str(family)


def _assign_patient(patient: Patient, values: dict[str, Any], given: str, family: str) -> None:
    patient.given_names = given
    patient.family_names = family
    for field, attribute in (
        ("birth_date", "birth_date"),
        ("email", "email"),
        ("eps", "eps"),
        ("telegram_chat_id", "telegram_chat_id"),
        ("secondary_contact_name", "secondary_contact_name"),
        ("external_ref", "external_ref"),
    ):
        if (value := values.get(field)) is not None:
            setattr(patient, attribute, value)

    if (phone := values.get("phone_e164") or values.get("phone_fixed")) is not None:
        patient.phone_e164 = str(phone)
        patient.phone_e164_bidx = blind_index(str(phone))
    if (contact := values.get("secondary_contact_phone")) is not None:
        patient.secondary_contact_phone = str(contact)


async def _apply_specialties(
    session: AsyncSession, *, clinic_id: uuid.UUID, rows: list[RowResult]
) -> ApplyResult:
    created = updated = skipped = 0
    for row in rows:
        if row.status.value != "valid":
            skipped += 1
            continue
        name = row.values.get("name")
        if not name:
            skipped += 1
            continue
        reference = row.values.get("external_ref")
        existing = (
            await session.scalar(
                select(Specialty).where(
                    Specialty.clinic_id == clinic_id, Specialty.external_ref == str(reference)
                )
            )
            if reference
            else None
        )
        if existing is not None:
            existing.name = str(name)
            updated += 1
            continue
        session.add(
            Specialty(
                clinic_id=clinic_id,
                name=str(name),
                external_ref=str(reference) if reference else None,
            )
        )
        created += 1
    await session.flush()
    return ApplyResult(created, updated, skipped)


async def doctor_reference_values(session: AsyncSession, *, clinic_id: uuid.UUID) -> set[str]:
    """Every way an appointment could name a doctor this clinic already has.

    Both the clinic's own code and the doctor's name, because a file may use
    either. Doctors are not patient data, so no audit entry is recorded: this
    returns no identifiers and discloses nothing about a patient.
    """
    rows = (
        await session.execute(
            select(Doctor.external_ref, Doctor.full_name).where(Doctor.clinic_id == clinic_id)
        )
    ).all()
    return {value for row in rows for value in row if value}


async def existing_patient_documents(
    session: AsyncSession, *, clinic_id: uuid.UUID, document_numbers: set[str]
) -> set[str]:
    """Which of these document numbers the clinic already holds a patient for.

    An appointments sheet usually names patients imported on an earlier run, so
    checking only the current file rejects every one of them. Soft-deleted
    patients are excluded: a deletion is normally a privacy request, and an
    appointment naming one has to reach a person rather than resolve quietly.

    Matching is by blind index, and only the numbers the caller already supplied
    are returned. Nothing is decrypted and no identifier the caller did not have
    is disclosed, so this records no audit entry.
    """
    if not document_numbers:
        return set()

    by_index = {blind_index(number): number for number in document_numbers}
    found = (
        await session.scalars(
            select(Patient.document_number_bidx).where(
                Patient.clinic_id == clinic_id,
                Patient.deleted_at.is_(None),
                Patient.document_number_bidx.in_(by_index.keys()),
            )
        )
    ).all()
    return {by_index[index] for index in found if index in by_index}


async def _apply_doctors(
    session: AsyncSession, *, clinic_id: uuid.UUID, rows: list[RowResult]
) -> ApplyResult:
    created = updated = skipped = 0
    specialties = {
        s.external_ref: s
        for s in (
            await session.scalars(select(Specialty).where(Specialty.clinic_id == clinic_id))
        ).all()
        if s.external_ref
    }

    for row in rows:
        if row.status.value != "valid":
            skipped += 1
            continue
        full_name = row.values.get("full_name")
        if full_name is not None and hasattr(full_name, "given_names"):
            name: Any = f"{full_name.given_names} {full_name.family_names}".strip()
        else:
            name = full_name
        if not name:
            skipped += 1
            continue

        reference = row.values.get("external_ref")
        raw_specialty = row.values.get("specialty")
        # The specialty column may hold a name or a code into the clinic's own
        # catalogue; both are kept, so neither reading is lost.
        catalogue = specialties.get(str(raw_specialty)) if raw_specialty else None

        existing = (
            await session.scalar(
                select(Doctor).where(
                    Doctor.clinic_id == clinic_id, Doctor.external_ref == str(reference)
                )
            )
            if reference
            else None
        )
        target = existing or Doctor(clinic_id=clinic_id, full_name=str(name), specialty="")
        target.full_name = str(name)
        target.specialty = catalogue.name if catalogue else str(raw_specialty or "")
        target.specialty_id = catalogue.id if catalogue else None
        if reference:
            target.external_ref = str(reference)
        if (office := row.values.get("office_number")) is not None:
            target.office_number = str(office)

        if existing is None:
            session.add(target)
            created += 1
        else:
            updated += 1

    await session.flush()
    return ApplyResult(created, updated, skipped)


def _jsonable(value: Any) -> Any:
    """Render a converted value for JSONB storage."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dt.date | dt.time | dt.datetime):
        return value.isoformat()
    if hasattr(value, "given_names"):
        return {"given_names": value.given_names, "family_names": value.family_names}
    return str(value)
