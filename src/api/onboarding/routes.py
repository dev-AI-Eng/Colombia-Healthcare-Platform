"""Upload a clinic file, confirm what its columns mean, then commit it.

Drive this from Swagger UI at /docs, or from the review screen at
`/onboarding/uploads/{id}/review`. The order is deliberate and enforced:

    POST /onboarding/uploads                  read the file, propose a mapping
    GET  /onboarding/uploads/{id}             what was proposed, and why
    PUT  /onboarding/uploads/{id}/mapping     the reviewer's corrections
    POST /onboarding/uploads/{id}/validate    run every rule over every row
    GET  /onboarding/uploads/{id}/rows        the staged rows and their errors
    POST /onboarding/uploads/{id}/commit      apply, only if nothing is invalid

The session, its staged rows and its transform log live in the `onboarding`
schema, so an import survives a restart and can be audited afterwards. The
*parsed file* is held in memory for the life of the process: the upload's bytes
are discarded once read, so a restart mid-review means uploading the file again.
That is a deliberate trade — keeping a clinic's file on disk for longer than the
review needs is a larger risk than asking for it twice.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Annotated, Any, Final

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from src.api.dependencies import ClinicScopeDep, SessionDep
from src.api.onboarding.schemas import (
    CellOut,
    CommitOut,
    CorrectionIn,
    MappingIn,
    ProfileOut,
    RowOut,
    SheetOut,
    StructureAnswerIn,
    StructureQuestionOut,
    UploadOut,
    ValidationOut,
)
from src.onboarding import repository, service
from src.onboarding.canonical import Entity
from src.onboarding.matcher import SHARED_FIELDS
from src.onboarding.models import ImportProfile
from src.onboarding.reader import ReadResult, UnreadableFile, read_isolated
from src.onboarding.repository import ApplyResult

router = APIRouter()

# Uploads are spooled to disk and parsed, so this bounds what a single request
# can cost. The archive guards in reader.py bound what it can expand to.
MAX_UPLOAD_BYTES: Final = 50 * 1024 * 1024


#: Parsed files, by session id. Not the state of the import — that lives in the
#: database — only the sheets already read, so the reviewer's next request does
#: not need the original upload again.
_PARSED: dict[uuid.UUID, ReadResult] = {}

#: The uploaded file on disk, by session id. Kept so a structure question can be
#: answered and the file re-read with that answer applied: the answer changes how
#: the bytes are interpreted, so it cannot be applied to an already-parsed table.
_SOURCE: dict[uuid.UUID, Path] = {}


async def _load(
    db: SessionDep, session_id: uuid.UUID, clinic_id: uuid.UUID
) -> tuple[Any, ReadResult]:
    """The stored session and its parsed file, or a 404 that reveals nothing."""
    record = await repository.get_session(db, clinic_id=clinic_id, session_id=session_id)
    if record is None:
        # Not 403 for another clinic's session: "exists, but not yours" would
        # leak one clinic's activity to another.
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such import session.")
    parsed = _PARSED.get(session_id)
    if parsed is None:
        raise HTTPException(
            status.HTTP_410_GONE,
            "The uploaded file is no longer held in memory, which happens after a "
            "restart. Upload it again to continue.",
        )
    return record, parsed


def _reports(record: Any) -> list[service.SheetReport]:
    return [service.report_from_dict(s) for s in (record.report or {}).get("sheets", [])]


def _store_reports(record: Any, reports: list[service.SheetReport], **extra: Any) -> None:
    """Replace the session's stored report. JSONB needs a new object to persist."""
    report = dict(record.report or {})
    report["sheets"] = [service.report_to_dict(r) for r in reports]
    report.update(extra)
    record.report = report


def _mapping_of(record: Any, sheet: str) -> dict[str, str | None]:
    return dict((record.report or {}).get("mappings", {}).get(sheet, {}))


def _decisions_of(record: Any, sheet: str) -> dict[str, str]:
    return dict((record.report or {}).get("decisions", {}).get(sheet, {}))


def _excluded_of(record: Any, sheet: str) -> set[int]:
    return set((record.report or {}).get("excluded", {}).get(sheet, []))


def _corrections_of(record: Any, sheet: str) -> dict[int, dict[str, str]]:
    """Cells a reviewer has answered, keyed by row number then column."""
    stored = (record.report or {}).get("corrections", {}).get(sheet, {})
    return {int(row): dict(cells) for row, cells in stored.items()}


#: Sent as a sheet's `entity` to leave it out of the import entirely.
SKIP_SHEET: Final = "skip"


def stored_skips(record: Any) -> list[str]:
    """Sheets the reviewer has said not to import."""
    return list((record.report or {}).get("skipped_sheets", []))


def _readable_size(limit: int) -> str:
    """A byte count as a person would say it, so a refusal names a real number."""
    megabytes = limit / (1024 * 1024)
    if megabytes >= 1:
        return f"{megabytes:.0f} MB"
    return f"{limit / 1024:.0f} KB"


def _stored_entity(profile: ImportProfile) -> Entity | None:
    """The sheet type a reviewer confirmed, or None if the profile predates it.

    Profiles written before the entity was stored hold only the mapping, so a
    missing or unrecognised value falls back to the heuristic rather than
    failing the upload.
    """
    stored = profile.mapping.get("entity")
    if not isinstance(stored, str):
        return None
    try:
        return Entity(stored)
    except ValueError:
        return None


def _default_mapping(report: service.SheetReport) -> dict[str, str | None]:
    """What the reviewer sees pre-ticked: confident proposals only."""
    return {c.column: (c.target_field if c.auto else None) for c in report.columns}


@router.post(
    "/uploads",
    response_model=UploadOut,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a clinic spreadsheet and get a proposed mapping",
)
async def upload(
    db: SessionDep,
    scope: ClinicScopeDep,
    file: Annotated[UploadFile, File(description="An .xlsx or .csv export from the clinic.")],
) -> UploadOut:
    """Read the file and propose what each column means. Nothing is converted yet.

    If this clinic already confirmed a mapping for a file of this shape, that
    profile is applied and there is nothing left to correct — which is what lets
    a repeat import run without a model call.
    """
    # The client's filename is never used as a path. Taking its last component
    # stopped "../x" escaping the temp directory, but the name still reached
    # `open()`, and a name of ".." or "a:b.csv" or 300 characters is a filesystem
    # error rather than a filename -- seven such names answered the upload with a
    # 500. The bytes go to a name we choose.
    #
    # The extension is kept because openpyxl refuses a file by extension before
    # looking at its contents: a neutral ".bin" made every workbook unreadable.
    # Only a short alphanumeric suffix is taken, so the extension cannot carry a
    # path or a filesystem-hostile character either.
    upload_dir = Path(tempfile.mkdtemp())
    suffix = Path((file.filename or "").replace("\\", "/")).suffix.lower()
    safe_suffix = suffix if suffix[1:].isalnum() and len(suffix) <= 6 else ""
    target = upload_dir / f"upload{safe_suffix}"
    # Kept for the session record and the screen, trimmed to what the column
    # holds, because that is the only place the clinic's own name belongs.
    stored_name = (file.filename or "upload")[:255]
    digest = hashlib.sha256()
    written = 0
    keep_upload = False
    try:
        with target.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        # Stated in whichever unit reads sensibly: integer MB
                        # rendered a sub-megabyte limit as "0 MB", which tells
                        # a receptionist nothing about the file they chose.
                        f"File is larger than the {_readable_size(MAX_UPLOAD_BYTES)} "
                        f"limit for an upload.",
                    )
                digest.update(chunk)
                handle.write(chunk)

        try:
            result = read_isolated(target)
        except UnreadableFile as error:
            # The reader's refusals are written for the person who uploaded the
            # file, so they are passed through rather than replaced.
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error

        reports = list(service.analyse(result))
        sha256 = digest.hexdigest()

        # A confirmed mapping for a file of this shape replaces the proposal.
        mappings: dict[str, dict[str, str | None]] = {}
        reused: list[str] = []
        for index, report in enumerate(reports):
            sheet = next(s for s in result.sheets if s.name == report.sheet)
            # Every profile confirmed for a sheet of this shape, not just one.
            # The entity is part of the key, so a workbook whose Doctores and
            # Especialidades sheets share headers has one profile each.
            candidates = await repository.find_profiles_for_shape(
                db, clinic_id=scope.clinic_id, headers=sheet.headers
            )
            # A profile confirmed for the kind of sheet we guessed is the plain
            # case, and it wins outright.
            matching = [p for p in candidates if _stored_entity(p) is report.entity]
            if not matching:
                # Nothing for our guess. A profile may still apply, but only if
                # it was confirmed for a sheet of this name: that is a reviewer
                # saying "this sheet, which you keep guessing wrong, is really a
                # patient sheet", and it is how AGENDA keeps its correction.
                #
                # Without the name it is a different sheet that merely shares
                # headers -- Especialidades borrowing the Doctores profile --
                # and applying it relabels the sheet and writes the rows as the
                # wrong kind with nothing blocking. So that one is left alone.
                matching = [
                    p
                    for p in candidates
                    if p.mapping.get("sheet") == report.sheet
                    and p.mapping.get("source") == stored_name
                ]
            profile = matching[0] if len(matching) == 1 else None
            if profile is None:
                mappings[report.sheet] = _default_mapping(report)
                continue
            stored = dict(profile.mapping.get("mapping", {}))
            mappings[report.sheet] = {c.column: stored.get(c.column) for c in report.columns}
            # The profile stores the sheet type the reviewer confirmed. Without
            # it the heuristic runs again on every upload, so a sheet named
            # AGENDA that actually holds patients comes back as `appointment`
            # and the stored mapping targets fields that entity does not have.
            reports[index] = service.apply_profile(
                report, mappings[report.sheet], _stored_entity(profile)
            )
            reused.append(report.sheet)

        previous = await repository.find_previous_import(
            db, clinic_id=scope.clinic_id, file_sha256=sha256
        )
        record = await repository.create_session(
            db,
            clinic_id=scope.clinic_id,
            filename=stored_name,
            file_size=written,
            file_sha256=sha256,
            report={},
        )
        _store_reports(
            record,
            reports,
            mappings=mappings,
            decisions={},
            reused_profiles=reused,
            duplicate_of=str(previous.id) if previous else None,
        )
        record.total_rows = sum(r.total_rows for r in reports)
        await db.flush()
        _PARSED[record.id] = result
        if any(sheet.questions for sheet in result.sheets):
            _SOURCE[record.id] = target
            keep_upload = True

        return UploadOut(
            session_id=record.id,
            filename=record.filename,
            file_sha256=sha256,
            encoding=result.encoding,
            delimiter=result.delimiter,
            sheets=[SheetOut.build(r) for r in reports],
            structure_questions=_structure_questions(result, {}),
            status=record.status,
            reused_profiles=reused,
            duplicate_of=previous.id if previous else None,
        )
    finally:
        # Deleted unless a structure question is outstanding, because answering
        # one re-reads the bytes. `keep_upload` is only ever set on the success
        # path, so a refused or oversized upload still deletes: leaving patient
        # data on disk for a file nobody can act on would be a slow leak.
        if not keep_upload:
            shutil.rmtree(upload_dir, ignore_errors=True)


@router.get(
    "/uploads/{session_id}",
    response_model=UploadOut,
    summary="What was proposed for this file, and why",
)
async def get_upload(db: SessionDep, session_id: uuid.UUID, scope: ClinicScopeDep) -> UploadOut:
    record, parsed = await _load(db, session_id, scope.clinic_id)
    stored = record.report or {}
    return UploadOut(
        session_id=record.id,
        filename=record.filename,
        file_sha256=record.file_sha256,
        encoding=parsed.encoding,
        delimiter=parsed.delimiter,
        sheets=[SheetOut.build(r) for r in _reports(record)],
        # Read from the parsed file rather than the stored report: a question is
        # a property of how the bytes read, and answering one can raise the next
        # one (declining the header row exposes the rows it would have fixed).
        structure_questions=_structure_questions(parsed, _structure_answers(record)),
        status=record.status,
        reused_profiles=list(stored.get("reused_profiles", [])),
        duplicate_of=uuid.UUID(stored["duplicate_of"]) if stored.get("duplicate_of") else None,
    )


async def _known_references(
    db: SessionDep,
    *,
    clinic_id: uuid.UUID,
    converted: list[tuple[service.SheetReport, list[service.RowResult]]],
) -> dict[str, set[str]]:
    """What a reference in this import is allowed to name.

    Two sources, because either one alone gives a wrong answer: the other sheets
    of the same workbook (a doctor defined in this import), and what the clinic
    already holds (a doctor imported last month). Checking only the file would
    reject every appointment on a repeat import; checking only the database would
    reject a workbook that defines its own doctors.
    """
    doctors: set[str] = set()
    patients: set[str] = set()

    for report, rows in converted:
        for row in rows:
            if report.entity is Entity.DOCTOR:
                for field_name in ("external_ref", "full_name"):
                    value = str(row.values.get(field_name, "")).strip()
                    if value:
                        doctors.add(value.casefold())
            elif report.entity is Entity.PATIENT:
                number = str(row.values.get("document_number", "")).strip()
                if number:
                    patients.add(number.casefold())

    for doctor in await repository.doctor_reference_values(db, clinic_id=clinic_id):
        doctors.add(doctor.casefold())

    # Patients the clinic already holds count too, and for the same reason as
    # doctors: an appointments sheet names the people imported last month, so
    # checking only this file rejects every one of them. The lookup matches by
    # blind index and returns nothing the caller did not already supply.
    wanted = {
        str(row.values.get("patient_document", "")).strip()
        for report, rows in converted
        if report.entity is Entity.APPOINTMENT
        for row in rows
        if str(row.values.get("patient_document", "")).strip()
    }
    for number in await repository.existing_patient_documents(
        db, clinic_id=clinic_id, document_numbers=wanted - patients
    ):
        patients.add(number.casefold())

    return {"doctor_ref": doctors, "patient_document": patients}


def _structure_questions(
    parsed: ReadResult, answered: dict[str, bool]
) -> list[StructureQuestionOut]:
    """Every structure question the current read raises, with its answer."""
    return [
        StructureQuestionOut(
            id=question.id,
            sheet=sheet.name,
            finding=question.finding,
            if_approved=question.applied_if_approved,
            if_declined=question.applied_if_declined,
            answered=answered.get(question.id),
        )
        for sheet in parsed.sheets
        for question in sheet.questions
    ]


def _structure_answers(record: Any) -> dict[str, bool]:
    return {str(k): bool(v) for k, v in (record.report or {}).get("structure", {}).items()}


@router.post(
    "/uploads/{session_id}/structure",
    response_model=UploadOut,
    summary="Approve or decline something about the file's shape",
)
async def answer_structure(
    db: SessionDep, session_id: uuid.UUID, scope: ClinicScopeDep, body: StructureAnswerIn
) -> UploadOut:
    """Answer one structure question, then re-read the file under that answer.

    These questions are about how the bytes are read — which row holds the
    headings, whether a row with too many fields is cut or left out — so an
    answer cannot be applied to a table that has already been parsed. The file
    is read again with every answer so far, and the mapping is re-proposed
    against whatever headings that produces.

    Declining is always the reading the importer would have used anyway, so a
    declined question discards that one finding and changes nothing else.
    """
    record, _ = await _load(db, session_id, scope.clinic_id)
    if record.status == "committed":
        raise HTTPException(status.HTTP_409_CONFLICT, "This import was already committed.")

    source = _SOURCE.get(session_id)
    if source is None or not source.exists():
        raise HTTPException(
            status.HTTP_410_GONE,
            "The uploaded file is no longer on disk, which happens after a restart. "
            "Upload it again to continue.",
        )

    known = {q.id for sheet in _PARSED[session_id].sheets for q in sheet.questions}
    if body.id not in known:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"{body.id!r} is not a question about this file. Open the file to see "
            f"which questions it raises.",
        )

    answers = _structure_answers(record)
    answers[body.id] = body.approved

    try:
        result = read_isolated(source, answers)
    except UnreadableFile as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error

    _PARSED[session_id] = result
    # A question stays listed once answered, so a reviewer can see what they
    # decided: "still raised" is not "still open". The upload is only needed
    # while an *unanswered* question could change how the bytes are read, and
    # holding a clinic's file on disk any longer than that is patient data left
    # lying around for no reason.
    if all(question.id in answers for sheet in result.sheets for question in sheet.questions):
        shutil.rmtree(source.parent, ignore_errors=True)
        _SOURCE.pop(session_id, None)

    # The headings may be different ones now, so the mapping is proposed afresh
    # rather than carried across from a table that no longer exists.
    reports = list(service.analyse(result))
    mappings = {r.sheet: _default_mapping(r) for r in reports}
    stored = dict(record.report or {})
    _store_reports(
        record,
        reports,
        mappings=mappings,
        decisions={},
        structure=answers,
        reused_profiles=stored.get("reused_profiles", []),
        duplicate_of=stored.get("duplicate_of"),
    )
    record.total_rows = sum(r.total_rows for r in reports)
    # Validation ran against the old reading, so its verdict no longer applies.
    record.status = "mapped"
    record.valid_rows = record.review_rows = record.invalid_rows = 0
    await db.flush()
    return await get_upload(db, session_id, scope)


@router.put(
    "/uploads/{session_id}/mapping",
    response_model=UploadOut,
    summary="Correct the proposed mapping for one sheet",
)
async def set_mapping(
    db: SessionDep, session_id: uuid.UUID, scope: ClinicScopeDep, body: MappingIn
) -> UploadOut:
    """Replace a sheet's mapping with what the reviewer confirmed.

    A field may be mapped from at most one column: two columns writing one field
    would mean one of them silently wins.
    """
    record, parsed = await _load(db, session_id, scope.clinic_id)
    reports = _reports(record)
    report = next((r for r in reports if r.sheet == body.sheet), None)
    if report is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No sheet named {body.sheet!r}.")
    index = reports.index(report)

    known = {c.column for c in report.columns}
    unknown = set(body.mapping) - known
    if unknown:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"These columns are not in the sheet: {sorted(unknown)}.",
        )

    # The name fields are exempt: RIPS exports a name in four columns, and
    # those are parts of one value rather than rival versions of it.
    assigned = [
        target for target in body.mapping.values() if target and target not in SHARED_FIELDS
    ]
    duplicated = {t for t in assigned if assigned.count(t) > 1}
    if duplicated:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"More than one column maps to {sorted(duplicated)}; one would overwrite the other.",
        )

    stored = dict(record.report or {})
    mappings = dict(stored.get("mappings", {}))
    decisions = dict(stored.get("decisions", {}))
    # Merged, not replaced. The screen posts every column, but a caller sending
    # one correction should not silently clear the rest of the sheet.
    merged = dict(mappings.get(body.sheet, {}))
    merged.update(body.mapping)
    mappings[body.sheet] = merged
    excluded = dict(stored.get("excluded", {}))
    if body.excluded_rows:
        excluded[body.sheet] = sorted(set(excluded.get(body.sheet, [])) | set(body.excluded_rows))
    decisions[body.sheet] = dict(body.decisions)

    if body.entity == SKIP_SHEET:
        # A workbook often carries a stale sheet nobody wants imported. Clearing
        # its columns one at a time is not a workflow, so the reviewer says so
        # once and validate leaves the sheet alone.
        skipped_sheets = sorted({*stored_skips(record), body.sheet})
        _store_reports(record, reports, skipped_sheets=skipped_sheets)
        record.status = "mapped"
        await db.flush()
        return await get_upload(db, session_id, scope)

    if body.entity and body.entity not in {e.value for e in Entity}:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"{body.entity!r} is not a kind of sheet. Choose one of "
            f"{[*sorted(e.value for e in Entity), SKIP_SHEET]}.",
        )

    if body.entity and body.entity != report.entity.value:
        # The sheet holds something other than we guessed, so its columns are
        # matched again against the right set of fields rather than kept.
        sheet = next(s for s in parsed.sheets if s.name == body.sheet)
        report = service.reanalyse(sheet, Entity(body.entity))
        reports[index] = report
        if not body.mapping:
            # No corrections came with the change, so start from the fresh proposal.
            merged = _default_mapping(report)
            mappings[body.sheet] = merged

    reports[index] = service.apply_profile(report, mappings[body.sheet])
    _store_reports(record, reports, mappings=mappings, decisions=decisions, excluded=excluded)
    record.status = "mapped"
    await db.flush()
    return await get_upload(db, session_id, scope)


@router.post(
    "/uploads/{session_id}/rows/correct",
    response_model=UploadOut,
    summary="Answer a row the file could not decide for itself",
)
async def correct_row(
    db: SessionDep, session_id: uuid.UUID, scope: ClinicScopeDep, body: CorrectionIn
) -> UploadOut:
    """Record a reviewer's answer for one row, then revalidate.

    Some rows cannot be decided from the file at all: a three-word Colombian
    name splits two ways and Ley 2129 de 2021 lets parents choose the order, so
    no positional rule settles it. Refusing such a row is right, but a refusal
    a reviewer cannot answer is a dead end — the row would simply never import.

    The answer supplies the cell's **text** and is stored against the session,
    not written into the staged row: `validate` applies it before conversion, so
    the corrected value runs through the same normalizer as every other cell on
    every revalidation. A correction that is itself invalid is caught exactly
    like an original value, and the transform log still shows the rule that
    produced the cell.
    """
    record, parsed = await _load(db, session_id, scope.clinic_id)
    if record.status == "committed":
        raise HTTPException(status.HTTP_409_CONFLICT, "This import was already committed.")

    sheet = next((s for s in parsed.sheets if s.name == body.sheet), None)
    if sheet is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No sheet named {body.sheet!r}.")

    unknown = set(body.cells) - set(sheet.headers)
    if unknown:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"These columns are not in the sheet: {sorted(unknown)}.",
        )

    last_row = sheet.header_row + len(sheet.rows)
    if not sheet.header_row < body.row_number <= last_row:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"Row {body.row_number} is not a data row; this sheet holds rows "
            f"{sheet.header_row + 1} to {last_row}.",
        )

    stored = dict(record.report or {})
    corrections = dict(stored.get("corrections", {}))
    for_sheet = dict(corrections.get(body.sheet, {}))
    # Merged per row, so answering a second column does not undo the first.
    row_cells = dict(for_sheet.get(str(body.row_number), {}))
    row_cells.update(body.cells)
    for_sheet[str(body.row_number)] = row_cells
    corrections[body.sheet] = for_sheet
    _store_reports(record, _reports(record), corrections=corrections)
    await db.flush()

    # Revalidated here rather than left to the caller: a correction that does
    # not change the row's status is a correction the reviewer needs to see.
    await validate(db, session_id, scope)
    return await get_upload(db, session_id, scope)


@router.post(
    "/uploads/{session_id}/validate",
    response_model=ValidationOut,
    summary="Run every rule over every row and stage the results",
)
async def validate(db: SessionDep, session_id: uuid.UUID, scope: ClinicScopeDep) -> ValidationOut:
    """Convert all rows, stage them, and say whether they may be committed.

    Runs over 100% of rows, never a sample (ADR-08a): a sample that looks fine
    is exactly how a bad transform reaches production.
    """
    record, parsed = await _load(db, session_id, scope.clinic_id)
    reports = _reports(record)

    skipped_sheets = stored_skips(record)
    summaries: list[service.SheetReport] = []
    blocking: list[str] = []
    # Failures small enough to import around, reported rather than hidden.
    tolerated: list[str] = []
    totals = {"valid": 0, "review": 0, "invalid": 0}

    # Converted rows per sheet, kept so the cross-row checks can run once every
    # sheet has been converted: a reference may point at a doctor defined on
    # another sheet of the same workbook, so neither check can be done per sheet.
    converted: list[tuple[service.SheetReport, list[service.RowResult]]] = []

    for report in reports:
        sheet = next(s for s in parsed.sheets if s.name == report.sheet)
        if report.sheet in skipped_sheets:
            continue  # the reviewer said this sheet is not part of the import
        mapping = _mapping_of(record, report.sheet)
        if not any(mapping.values()):
            continue  # nothing confirmed for this sheet, so nothing to import

        rows, columns = service.validate(
            sheet,
            report.entity,
            mapping,
            decisions=_decisions_of(record, report.sheet),
            excluded_rows=_excluded_of(record, report.sheet),
            corrections=_corrections_of(record, report.sheet),
        )
        await repository.replace_staging(
            db,
            clinic_id=scope.clinic_id,
            session_id=record.id,
            sheet=report.sheet,
            entity=report.entity,
            rows=rows,
        )
        converted.append((report, rows))
        summary = service.summarise(rows, report, columns)
        summaries.append(summary)
        totals["valid"] += summary.valid_rows
        totals["review"] += summary.review_rows
        totals["invalid"] += summary.invalid_rows

        if summary.invalid_rows:
            # A few unconvertible rows no longer refuse the whole file. Above
            # the allowance the export itself is wrong and a partial import
            # would be worse than none: staff trust a schedule that looks
            # populated. Below it, the valid rows import and every failed row is
            # still listed with its reason, so nothing is silently discarded --
            # which is what every other importer does here.
            allowed = service.tolerable_invalid_rows(summary.total_rows)
            if summary.invalid_rows > allowed:
                blocking.append(
                    f"{summary.sheet}: {summary.invalid_rows} of {summary.total_rows} row(s) "
                    f"could not be converted, which is more than the {allowed} this file's "
                    f"size allows. The export itself looks wrong. Each row is listed below "
                    f"with the reason, so the source file can be corrected."
                )
            else:
                tolerated.append(
                    f"{summary.sheet}: {summary.invalid_rows} of {summary.total_rows} row(s) "
                    f"could not be converted and will not be imported. Each one is listed "
                    f"with its reason."
                )
        if summary.missing_required:
            blocking.append(
                f"{summary.sheet}: no column was mapped to {list(summary.missing_required)}."
            )
        if summary.questions:
            blocking.append(f"{summary.sheet}: {len(summary.questions)} question(s) unanswered.")

    # ---- duplicates, and references that point at nothing (scope: validation)
    # Both are properties of the file as a whole, so they run once every sheet
    # has been converted rather than inside the per-sheet loop.
    known = await _known_references(db, clinic_id=scope.clinic_id, converted=converted)
    for report, rows in converted:
        duplicates = service.find_duplicates(rows, report.entity)
        dangling = service.find_dangling_references(rows, report.entity, known=known)

        for row_number, reason in sorted(duplicates.items()):
            blocking.append(f"{report.sheet} row {row_number}: {reason}.")
        for row_number, reason in sorted(dangling.items()):
            blocking.append(f"{report.sheet} row {row_number}: {reason}.")

    # A question about the file's shape decides which rows and headings exist at
    # all, so leaving one unanswered would commit a reading nobody confirmed.
    answered = _structure_answers(record)
    unanswered = [
        question.id
        for sheet in parsed.sheets
        for question in sheet.questions
        if question.id not in answered
    ]
    for question_id in unanswered:
        blocking.append(
            f"{question_id}: approve or decline how this file is read before importing it."
        )

    if not summaries:
        blocking.append("No sheet has a confirmed mapping, so there is nothing to import.")

    record.status = "validated" if not blocking else "needs_review"
    record.valid_rows = totals["valid"]
    record.review_rows = totals["review"]
    record.invalid_rows = totals["invalid"]
    _store_reports(
        record,
        summaries or reports,
        blocking=blocking,
        tolerated=tolerated,
        skipped_sheets=skipped_sheets,
    )
    await db.flush()

    return ValidationOut(
        session_id=record.id,
        status=record.status,
        sheets=[SheetOut.build(s) for s in summaries],
        can_commit=not blocking,
        blocking=blocking,
        tolerated=tolerated,
    )


@router.get(
    "/uploads/{session_id}/rows",
    response_model=list[RowOut],
    summary="The staged rows, so a reviewer can see what would be written",
)
async def get_rows(
    db: SessionDep,
    session_id: uuid.UUID,
    scope: ClinicScopeDep,
    sheet: Annotated[str | None, Query(description="Which sheet's rows to show.")] = None,
    row_status: Annotated[
        str | None, Query(alias="status", description="valid, review or invalid.")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> list[RowOut]:
    await _load(db, session_id, scope.clinic_id)
    if not await repository.staged_rows(
        db, clinic_id=scope.clinic_id, session_id=session_id, limit=1
    ):
        raise HTTPException(
            status.HTTP_409_CONFLICT, "Validate the import before reading its rows."
        )
    rows = await repository.staged_rows(
        db,
        clinic_id=scope.clinic_id,
        session_id=session_id,
        sheet=sheet,
        status=row_status,
        limit=limit,
    )
    return [
        RowOut(
            row_number=r.row_number,
            status=r.status,
            values={k: str(v) for k, v in (r.normalized or {}).items()},
            errors=list((r.errors or {}).get("errors", [])),
            reviews=list((r.errors or {}).get("reviews", [])),
        )
        for r in rows
    ]


@router.get(
    "/uploads/{session_id}/transform-log",
    response_model=list[CellOut],
    summary="What every rule did to every cell (ADR-08a)",
)
async def transform_log(
    db: SessionDep,
    session_id: uuid.UUID,
    scope: ClinicScopeDep,
    cell_status: Annotated[
        str | None, Query(alias="status", description="valid, review or invalid.")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=2000)] = 200,
) -> list[CellOut]:
    """The evidence that no transform was inferred from the data.

    Every cell carries the named rule that produced its value, so a reviewer can
    ask why a value became what it became without re-running the import.
    """
    await _load(db, session_id, scope.clinic_id)
    entries = await repository.transform_log(
        db, clinic_id=scope.clinic_id, session_id=session_id, status=cell_status, limit=limit
    )
    if not entries and not await repository.transform_log(
        db, clinic_id=scope.clinic_id, session_id=session_id, limit=1
    ):
        raise HTTPException(
            status.HTTP_409_CONFLICT, "Validate the import before reading its transform log."
        )
    return [
        CellOut(
            row_number=e.row_number,
            column=e.column_name,
            target_field=e.target_field,
            raw=e.raw_value or "",
            corrected_from_review=e.corrected_from_review,
            normalized=e.normalized_value,
            rule=e.rule,
            status=e.status,
            message=e.message or "",
        )
        for e in entries
    ]


@router.post(
    "/uploads/{session_id}/commit",
    response_model=CommitOut,
    summary="Apply the import, if and only if nothing is invalid",
)
async def commit(
    db: SessionDep,
    session_id: uuid.UUID,
    scope: ClinicScopeDep,
    save_profile: Annotated[
        bool, Query(description="Remember this mapping for the next file of the same shape.")
    ] = True,
) -> CommitOut:
    """Write the valid rows into the clinic's tables, in one transaction, or refuse.

    Refuses on exactly what `validate` reported. Recomputing a narrower check
    here is how the two drift apart: an earlier version counted invalid rows
    only, so a sheet blocked on an unanswered date question committed anyway.

    Rows awaiting review are not written. A patient imported without a usable
    identifier is worse than a patient not yet imported.
    """
    record, parsed = await _load(db, session_id, scope.clinic_id)
    stored = record.report or {}

    if record.status == "committed":
        raise HTTPException(status.HTTP_409_CONFLICT, "This import was already committed.")
    # What blocked validation is reported before "validate it first", so the
    # reviewer is told what is actually wrong rather than to repeat a step they
    # have already done.
    if blocking := list(stored.get("blocking", [])):
        raise HTTPException(status.HTTP_409_CONFLICT, "Refusing to commit. " + " ".join(blocking))
    if record.status != "validated":
        raise HTTPException(status.HTTP_409_CONFLICT, "Validate the import before committing it.")

    committed: dict[str, int] = {}
    created: dict[str, int] = {}
    updated: dict[str, int] = {}
    skipped: dict[str, int] = {}
    conflicts: list[str] = []

    # Specialties before doctors, so a doctor can reference the catalogue its
    # own file defines.
    order = {Entity.SPECIALTY: 0, Entity.PATIENT: 1, Entity.DOCTOR: 2}
    skipped_sheets = stored_skips(record)
    for report in sorted(_reports(record), key=lambda r: order.get(r.entity, 9)):
        if report.sheet in skipped_sheets:
            continue
        mapping = _mapping_of(record, report.sheet)
        if not any(mapping.values()):
            continue
        sheet = next(s for s in parsed.sheets if s.name == report.sheet)
        rows, _ = service.validate(
            sheet,
            report.entity,
            mapping,
            decisions=_decisions_of(record, report.sheet),
            excluded_rows=_excluded_of(record, report.sheet),
            corrections=_corrections_of(record, report.sheet),
        )
        # A sheet of visits converts to a patient AND an appointment per line,
        # so the rows are grouped by what they are rather than by the sheet's
        # own entity. Passing an appointment row to `_apply_patients` happened
        # to be harmless -- it has no document number, so it was skipped -- but
        # that is luck, not a guarantee, and the counts would be wrong.
        #
        # Patients before the appointments that name them, which is the parent
        # before the child: the same ordering `order` applies across sheets.
        applied = ApplyResult()
        for row_entity in sorted({r.entity for r in rows}, key=lambda e: order.get(e, 9)):
            of_entity = [r for r in rows if r.entity is row_entity]
            result = await repository.apply_rows(
                db, clinic_id=scope.clinic_id, entity=row_entity, rows=of_entity
            )
            applied = ApplyResult(
                created=applied.created + result.created,
                updated=applied.updated + result.updated,
                skipped=applied.skipped + result.skipped,
                conflicts=applied.conflicts + result.conflicts,
            )
        committed[report.sheet] = applied.created + applied.updated
        # Kept apart on purpose. An updated row replaced a record the clinic
        # already had, and anything a receptionist edited by hand since the last
        # import is gone. One combined number reads as "8 patients imported"
        # whether that is 8 new or 1 new and 7 overwritten.
        created[report.sheet] = applied.created
        updated[report.sheet] = applied.updated
        skipped[report.sheet] = applied.skipped
        conflicts.extend(applied.conflicts)

        if save_profile:
            await repository.save_profile(
                db,
                clinic_id=scope.clinic_id,
                name=f"{record.filename} - {report.sheet}",
                fingerprint=repository.header_fingerprint(sheet.headers, report.entity.value),
                mapping={
                    "mapping": mapping,
                    "entity": report.entity.value,
                    # Which sheet of which file this was confirmed for. The
                    # filename matters because every uploaded CSV is parsed from
                    # a temporary file called `upload`, so its sheet name alone
                    # cannot tell two different CSVs apart.
                    "sheet": report.sheet,
                    "source": record.filename,
                },
            )

    record.status = "committed"
    # Kept so the review screen can say what the import actually did. Without it
    # the page says only "committed", and the counts live in an API response the
    # receptionist never sees.
    record.report = {
        **(record.report or {}),
        "outcome": {"created": created, "updated": updated, "skipped": skipped},
    }
    await db.flush()

    total_created = sum(created.values())
    total_updated = sum(updated.values())
    saved = " The mapping was saved for the next file of this shape." if save_profile else ""
    if total_created or total_updated:
        # Overwrites are named, because the clinic cannot see them any other way.
        written = f"Imported {total_created} new record(s)"
        if total_updated:
            written += (
                f" and replaced {total_updated} existing one(s). Anything edited by "
                "hand in those records since the last import has been overwritten"
            )
        message = f"{written}. Rows awaiting review were not written.{saved}"
    else:
        # "Imported." for a commit that wrote nothing tells a receptionist the
        # opposite of what happened.
        message = f"Nothing was written: every row is awaiting review or was excluded.{saved}"

    return CommitOut(
        session_id=record.id,
        status=record.status,
        committed=committed,
        created=created,
        updated=updated,
        skipped=skipped,
        conflicts=conflicts,
        message=message,
    )


@router.get(
    "/profiles",
    response_model=list[ProfileOut],
    summary="Mapping profiles this clinic has confirmed",
)
async def profiles(db: SessionDep, scope: ClinicScopeDep) -> list[ProfileOut]:
    """A profile here is why a repeat import needs no review and no model call."""
    return [
        ProfileOut(
            id=p.id,
            name=p.name,
            header_fingerprint=p.header_fingerprint,
            entity=str(p.mapping.get("entity", "")),
            mapping={k: v for k, v in p.mapping.get("mapping", {}).items() if v},
            version=p.version,
            updated_at=p.updated_at,
        )
        for p in await repository.list_profiles(db, clinic_id=scope.clinic_id)
    ]


@router.get(
    "/canonical-fields",
    summary="The fields a column can be mapped to",
)
async def canonical_fields() -> dict[str, list[dict[str, object]]]:
    """What the confirmation screen offers in its dropdowns."""
    from src.onboarding.canonical import FIELDS_BY_ENTITY

    return {
        entity.value: [
            {
                "name": f.name,
                "requirement": f.requirement.value,
                "description": f.description,
                "examples": list(f.examples),
                "aliases": list(f.aliases[:8]),
            }
            for f in fields
        ]
        for entity, fields in FIELDS_BY_ENTITY.items()
    }


__all__ = ["Entity", "router"]
