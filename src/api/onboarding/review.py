"""The confirmation screen: one page, no build step, no JavaScript framework.

The PDF asks for a mapping confirmation screen. This is it — a server-rendered
page a person opens after uploading, showing what each column was taken to mean
and how much of it converted, with a dropdown to correct anything wrong.

It is deliberately plain HTML. A staff interface belongs to M11 and will be
designed properly; anything more here would be work the milestone does not own,
and would have to be thrown away. What matters now is that a human can actually
see and change the mapping before anything is written, because that is the step
the whole milestone turns on.

Two decisions carried from ADR-08a are visible on the page:

- Each column shows **percent valid over every row**, not a preview of the first
  few. A preview looks reassuring while row 100 is broken.
- Anything the file cannot decide is a **question**, shown before the columns
  and blocking the commit until answered.
"""

from __future__ import annotations

import contextlib
import html
import re
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse, RedirectResponse

from src.api.dependencies import ClinicScopeDep, SessionDep
from src.api.onboarding.routes import SKIP_SHEET
from src.onboarding import repository, service
from src.onboarding.canonical import FIELDS_BY_ENTITY, Entity

router = APIRouter()

_STYLE = """
:root { color-scheme: light dark; --line:#d6dae2; --muted:#5b667a; --ok:#1b7f79;
        --warn:#b5651d; --bad:#b3261e; --bg:#fff; --fg:#1d2433; }
@media (prefers-color-scheme: dark) {
  :root { --line:#333a46; --muted:#9aa4b5; --bg:#15181d; --fg:#e8ecf3; } }
* { box-sizing:border-box; }
body { font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif; margin:0;
       background:var(--bg); color:var(--fg); }
main { max-width:1100px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:22px; margin:0 0 4px; }
h2 { font-size:17px; margin:28px 0 8px; }
.sub { color:var(--muted); margin:0 0 20px; }
table { border-collapse:collapse; width:100%; margin:8px 0 4px; font-size:14px; }
th,td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--line);
        vertical-align:top; }
th { font-weight:600; font-size:12px; text-transform:uppercase;
     letter-spacing:.04em; color:var(--muted); }
select { width:100%; padding:5px; font:inherit; background:var(--bg);
         color:var(--fg); border:1px solid var(--line); border-radius:5px; }
.bar { display:inline-block; min-width:52px; }
.ok { color:var(--ok); } .warn { color:var(--warn); } .bad { color:var(--bad); }
.note { color:var(--muted); font-size:13px; }
.card { border:1px solid var(--line); border-radius:8px; padding:12px 16px;
        margin:12px 0; }
.card.warn { border-color:var(--warn); }
.card.bad { border-color:var(--bad); }
button { font:inherit; padding:9px 18px; border-radius:6px; border:1px solid var(--ok);
         background:var(--ok); color:#fff; cursor:pointer; }
button.secondary { background:transparent; color:var(--fg); border-color:var(--line); }
ul { margin:6px 0; padding-left:20px; }
code { font-family:ui-monospace,Consolas,monospace; font-size:13px; }
"""


def _answer_hint(reason: str) -> str:
    """What to type back, for the refusals a reviewer actually meets.

    An empty box with no hint is the reason a receptionist stalls on a name: the
    importer wants "Surnames, Given names", and nothing on screen said so.
    """
    text = reason.casefold()
    if "surname" in text or "split" in text:
        return "Surnames, Given names"
    if "reached" in text or "phone" in text:
        return "A reachable number, e.g. 3101234567"
    if "year" in text or "date" in text:
        return "yyyy-mm-dd"
    return ""


def _slug(value: str) -> str:
    """An id for a sheet anchor. Sheet names carry spaces, accents and brackets."""
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or "sheet"


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _percent(column: service.ColumnReport) -> str:
    """Percent valid, coloured by how much attention it needs."""
    if not column.total:
        return '<span class="note">not yet validated</span>'
    share = column.percent_valid
    css = "ok" if share == 100 else ("warn" if share >= 80 else "bad")
    detail = []
    if column.review:
        detail.append(f"{column.review} to review")
    if column.invalid:
        detail.append(f"{column.invalid} rejected")
    suffix = f' <span class="note">({", ".join(detail)})</span>' if detail else ""
    return f'<span class="bar {css}">{share:.0f}%</span>{suffix}'


def _options(entity: Entity, selected: str | None) -> str:
    chosen = "" if selected else " selected"
    parts = [f'<option value=""{chosen}>— do not import —</option>']
    for field in FIELDS_BY_ENTITY[entity]:
        mark = " selected" if field.name == selected else ""
        required = " *" if field.requirement.value == "required" else ""
        parts.append(
            f'<option value="{_escape(field.name)}"{mark}>'
            f"{_escape(field.name)}{required} — {_escape(field.description[:70])}</option>"
        )
    return "".join(parts)


def _sheet_section(report: service.SheetReport, session_id: uuid.UUID, clinic_id: uuid.UUID) -> str:
    rows = "".join(
        f"<tr><td><code>{_escape(c.column)}</code></td>"
        f'<td><select name="col::{_escape(c.column)}">{_options(report.entity, c.target_field)}'
        "</select></td>"
        f"<td>{_percent(c)}</td>"
        f'<td class="note">{_escape(c.reason)}</td></tr>'
        for c in report.columns
    )

    blocks = []
    if report.questions:
        items = "".join(f"<li>{_escape(q)}</li>" for q in report.questions)
        # Each question is a column the file cannot decide for itself. The answer
        # applies to the whole column, never row by row.
        inputs = "".join(
            f'<label class="note">{_escape(q.split(":")[0])}: '
            f'<select name="ask::{_escape(q.split(":")[0])}">'
            '<option value="">— choose —</option>'
            '<option value="day_first">day / month / year</option>'
            '<option value="month_first">month / day / year</option>'
            "</select></label><br>"
            for q in report.questions
        )
        blocks.append(
            f'<div class="card warn"><strong>Needs your decision</strong>'
            f"<ul>{items}</ul>{inputs}</div>"
        )
    if report.missing_required:
        missing = ", ".join(_escape(m) for m in report.missing_required)
        blocks.append(
            f'<div class="card bad"><strong>Required fields not mapped:</strong> {missing}.'
            " Choose a column for each, or export the file with those columns.</div>"
        )
    if report.warnings:
        items = "".join(f"<li>{_escape(w)}</li>" for w in report.warnings)
        blocks.append(f'<div class="card"><strong>About this file</strong><ul>{items}</ul></div>')

    counts = (
        f"{report.total_rows} rows · {report.valid_rows} ready · "
        f"{report.review_rows} to review · {report.invalid_rows} rejected"
        if report.valid_rows or report.review_rows or report.invalid_rows
        else f"{report.total_rows} rows"
    )

    kinds = "".join(
        f'<option value="{e.value}"{" selected" if e is report.entity else ""}>{e.value}</option>'
        for e in Entity
    )

    return f"""
    <h2 id="sheet-{_slug(report.sheet)}">{_escape(report.sheet)}</h2>
    <p class="sub">Read as <strong>{_escape(report.entity.value)}</strong> —
       {_escape(report.entity_reason)}<br>{counts}</p>
    {"".join(blocks)}
    <form method="post"
          action="/onboarding/uploads/{session_id}/review/{_escape(report.sheet)}?clinic_id={clinic_id}">
      <p class="note">
        This sheet holds
        <select name="entity" style="width:auto">{kinds}
          <option value="{SKIP_SHEET}">— do not import this sheet —</option>
        </select>
        · rows to leave out (numbers as the file shows them, e.g. a total line):
        <input name="exclude" placeholder="17, 18, 19" style="width:12em;padding:5px">
      </p>
      <table>
        <tr><th>Column in the file</th><th>Import as</th><th>Valid</th><th>Why</th></tr>
        {rows}
      </table>
      <p><button type="submit">Save this sheet</button>
         <span class="note">Saves this sheet only. Changes typed into another
         sheet below are not saved until you press its own button.</span></p>
    </form>
    """


def _review_rows(rows: list[Any], session_id: uuid.UUID, clinic_id: uuid.UUID) -> str:
    """The rows the file could not decide, each with the cell to answer.

    Refusing these rows is correct, but a refusal nobody can answer means the
    patient never imports. The reviewer types the cell's value; it is converted
    by the same normalizer as every other cell.
    """
    if not rows:
        return ""

    blocks = []
    for row in rows:
        # Stored as JSON on the row, the same shape the rows endpoint returns.
        review_reasons = list((row.errors or {}).get("reviews", []))
        reasons = "".join(f"<li>{_escape(r)}</li>" for r in review_reasons)
        # The column each reason names, so the reviewer answers that cell.
        columns = [r.split(":")[0] for r in review_reasons if ":" in r]
        # The reason names the column; the reason's text says what is wrong with
        # it, which is what tells the reviewer the shape to type back.
        hint_for = {r.split(":")[0]: _answer_hint(r) for r in review_reasons if ":" in r}
        inputs = "".join(
            f'<label class="note">{_escape(column)}: '
            f'<input name="cell::{_escape(column)}" style="width:18em;padding:5px"'
            f' placeholder="{_escape(hint_for.get(column, ""))}">'
            "</label><br>"
            for column in dict.fromkeys(columns)
        )
        blocks.append(
            f'<div class="card warn">'
            f"<strong>Row {row.row_number}</strong><ul>{reasons}</ul>"
            f'<form method="post" action="/onboarding/uploads/{session_id}'
            f'/review/rows/{row.row_number}?clinic_id={clinic_id}">'
            f'<input type="hidden" name="sheet" value="{_escape(row.sheet)}">'
            f"{inputs}"
            '<p><button class="secondary" type="submit">Save this row</button></p>'
            "</form></div>"
        )
    return (
        '<h2>Rows needing an answer</h2><p class="sub">These rows will not be '
        "imported until they are answered. What you type is converted by the same "
        "rules as the rest of the file.</p>" + "".join(blocks)
    )


@router.post("/uploads/{session_id}/review/rows/{row_number}", include_in_schema=False)
async def review_correct_row(
    db: SessionDep,
    session_id: uuid.UUID,
    row_number: int,
    scope: ClinicScopeDep,
    request: Request,
) -> RedirectResponse:
    from src.api.onboarding.routes import correct_row
    from src.api.onboarding.schemas import CorrectionIn

    form = await request.form()
    cells = {
        key[6:]: str(value).strip()
        for key, value in form.multi_items()
        if key.startswith("cell::") and str(value).strip()
    }
    if cells:
        await correct_row(
            db,
            session_id,
            scope,
            CorrectionIn(sheet=str(form.get("sheet") or ""), row_number=row_number, cells=cells),
        )
    return RedirectResponse(
        f"/onboarding/uploads/{session_id}/review?clinic_id={scope.clinic_id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get(
    "/uploads/{session_id}/review",
    response_class=HTMLResponse,
    include_in_schema=False,
    summary="The mapping confirmation screen",
)
async def review_screen(
    db: SessionDep, session_id: uuid.UUID, scope: ClinicScopeDep, request: Request
) -> HTMLResponse:
    record = await repository.get_session(db, clinic_id=scope.clinic_id, session_id=session_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such import session.")

    stored = record.report or {}
    reports = [service.report_from_dict(s) for s in stored.get("sheets", [])]
    sections = "".join(_sheet_section(r, session_id, scope.clinic_id) for r in reports)

    from src.api.onboarding.routes import _PARSED, _structure_answers

    answered = _structure_answers(record)
    parsed = _PARSED.get(session_id)
    structure = "".join(
        f'<div class="card {"warn" if question.id not in answered else ""}">'
        f"<strong>How this file is read</strong>"
        f"<p>{_escape(question.finding)}</p>"
        f'<form method="post" action="/onboarding/uploads/{session_id}'
        f'/review/structure?clinic_id={scope.clinic_id}">'
        f'<input type="hidden" name="id" value="{_escape(question.id)}">'
        f'<button name="approved" value="yes">{_escape(question.applied_if_approved)}</button> '
        f'<button class="secondary" name="approved" value="no">'
        f"{_escape(question.applied_if_declined)}</button>"
        f"</form>"
        + (
            f'<p class="note">Answered: {"applied" if answered[question.id] else "declined"}.</p>'
            if question.id in answered
            else ""
        )
        + "</div>"
        for sheet in (parsed.sheets if parsed else ())
        for question in sheet.questions
    )
    needing_answers = _review_rows(
        await repository.staged_rows(
            db,
            clinic_id=scope.clinic_id,
            session_id=session_id,
            status="review",
            limit=50,
        ),
        session_id,
        scope.clinic_id,
    )

    # Failures the file was allowed to carry. Shown whatever else the screen
    # says, because "imported" without them reads as "all of it imported".
    tolerated = stored.get("tolerated", [])
    tolerated_note = (
        '<div class="card warn"><strong>Some rows will not be imported.</strong>'
        + "".join(f"<p>{_escape(t)}</p>" for t in tolerated)
        + '<p class="note">They are listed below with the reason for each, so the '
        "source file can be corrected.</p></div>"
        if tolerated
        else ""
    )

    blocking = stored.get("blocking", [])
    if blocking:
        items = "".join(f"<li>{_escape(b)}</li>" for b in blocking)
        banner = (
            f'<div class="card bad"><strong>This import cannot be committed yet.</strong>'
            f"<ul>{items}</ul></div>"
        )
    elif record.status == "committed":
        outcome = stored.get("outcome", {})
        made = sum((outcome.get("created") or {}).values())
        replaced = sum((outcome.get("updated") or {}).values())
        held = sum((outcome.get("skipped") or {}).values())
        detail = f"{made} new record(s) written"
        if replaced:
            # Named, because it is the one outcome a clinic cannot see otherwise.
            detail += (
                f", {replaced} existing one(s) replaced -- anything edited by hand "
                "in those since the last import has been overwritten"
            )
        if held:
            detail += f", {held} row(s) left for review"
        banner = (
            f'<div class="card"><strong>This import has been committed.</strong> '
            f"{_escape(detail)}.</div>"
        )
    elif record.status == "validated":
        banner = (
            '<div class="card"><strong>Ready to import.</strong> Rows awaiting review '
            "will not be written.</div>"
        )
    else:
        banner = (
            '<div class="card">Check the mapping below, then <strong>Validate</strong> to '
            "convert every row.</div>"
        )

    reused = stored.get("reused_profiles", [])
    reused_note = (
        f'<p class="note">Mapping reused from a profile you confirmed earlier for: '
        f"{_escape(', '.join(reused))}.</p>"
        if reused
        else ""
    )
    duplicate = (
        '<div class="card warn">This exact file has already been imported for this '
        "clinic. Importing it again would repeat work someone has already done.</div>"
        if stored.get("duplicate_of")
        else ""
    )

    query = f"?clinic_id={scope.clinic_id}"
    return HTMLResponse(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Confirm import — {_escape(record.filename)}</title>
<style>{_STYLE}</style></head>
<body><main>
  <h1>Confirm this import</h1>
  <p class="sub"><code>{_escape(record.filename)}</code> · status
     <strong>{_escape(record.status)}</strong></p>
  {duplicate}{banner}{tolerated_note}{reused_note}{structure}
  {sections}
  {needing_answers}
  <h2>When the mapping is right</h2>
  <form method="post" action="/onboarding/uploads/{session_id}/review/validate{query}"
        style="display:inline">
    <button class="secondary" type="submit">Validate every row</button>
  </form>
  <form method="post" action="/onboarding/uploads/{session_id}/review/commit{query}"
        style="display:inline;margin-left:8px">
    <button type="submit">Import</button>
  </form>
  <p class="note" style="margin-top:20px">
    Percentages count every row in the file, not a sample. Rows awaiting review are
    never written. Nothing is imported until you press Import.
  </p>
</main></body></html>""")


@router.get(
    "/start",
    response_class=HTMLResponse,
    include_in_schema=False,
    summary="Pick a clinic, then import into it",
)
async def start_screen(db: SessionDep) -> HTMLResponse:
    """The only page that needs no clinic, because it is where one is chosen.

    Every other route requires `clinic_id`, because row-level security compares
    it against the transaction and without it nothing is visible. That is the
    right default and is not relaxed here: this page just saves pasting a UUID
    into the address bar for each file being tested.
    """
    from src.registry.repository import list_clinics

    clinics = await list_clinics(db)
    if not clinics:
        return HTMLResponse(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>No clinics yet</title>
<style>{_STYLE}</style></head><body><main>
  <h1>No clinics yet</h1>
  <div class="card bad">This database has no clinics, so there is nothing to
    import into. Run <code>python main.py</code> once to provision and seed it.</div>
</main></body></html>""")

    rows = "".join(
        f"<tr><td><strong>{_escape(c.name)}</strong></td>"
        f'<td class="note"><code>{c.id}</code></td>'
        f'<td><a href="/onboarding/?clinic_id={c.id}">import a file</a></td></tr>'
        for c in clinics
    )
    return HTMLResponse(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Import a spreadsheet</title><style>{_STYLE}</style></head>
<body><main>
  <h1>Import a spreadsheet</h1>
  <p class="sub">Pick the clinic to import into. Synthetic data only.</p>
  <table>
    <tr><th>Clinic</th><th>Id</th><th></th></tr>
    {rows}
  </table>
  <p class="note" style="margin-top:20px">
    Every other screen needs the clinic in its address, because the database
    hides rows belonging to a clinic the request did not name. This page exists
    so you do not have to type it.
  </p>
</main></body></html>""")


@router.get(
    "/",
    response_class=HTMLResponse,
    include_in_schema=False,
    summary="Choose a spreadsheet to import",
)
async def upload_screen(db: SessionDep, scope: ClinicScopeDep, request: Request) -> HTMLResponse:
    """Pick a file, and see the imports already started for this clinic.

    Deliberately plain: a file input, and a list of recent sessions so a half
    finished import can be picked up rather than started again. The staff
    interface proper belongs to M11; this is the "(basic)" the scope asks for.
    """
    recent = await repository.list_sessions(db, clinic_id=scope.clinic_id, limit=10)
    query = f"?clinic_id={scope.clinic_id}"

    rows = "".join(
        f"<tr><td><code>{_escape(session.filename)}</code></td>"
        f"<td>{_escape(session.status)}</td>"
        f'<td class="note">{_escape(session.started_at.strftime("%Y-%m-%d %H:%M"))}</td>'
        f'<td><a href="/onboarding/uploads/{session.id}/review{query}">open</a></td></tr>'
        for session in recent
    )
    table = (
        f"<h2>Imports already started</h2><table>"
        f"<tr><th>File</th><th>Status</th><th>Started</th><th></th></tr>{rows}</table>"
        if recent
        else '<p class="note">No imports yet for this clinic.</p>'
    )

    return HTMLResponse(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Import a spreadsheet</title>
<style>{_STYLE}</style></head>
<body><main>
  <h1>Import a spreadsheet</h1>
  <p class="sub">Excel (.xlsx) or CSV. Nothing is written to the clinic until you
     confirm the mapping on the next screen.</p>
  <div class="card">
    <form method="post" action="/onboarding/upload{query}"
          enctype="multipart/form-data">
      <p><input type="file" name="file" accept=".xlsx,.csv,.txt,.tsv" required></p>
      <p><button type="submit">Read this file</button></p>
    </form>
  </div>
  {table}
  <p class="note" style="margin-top:20px">
    Synthetic data only. This screen is served on loopback and is not available
    once real patient data is enabled.
  </p>
</main></body></html>""")


@router.post("/upload", include_in_schema=False)
async def upload_from_screen(
    db: SessionDep,
    scope: ClinicScopeDep,
    file: Annotated[UploadFile, File(description="The clinic's spreadsheet.")],
) -> RedirectResponse:
    """Hand the file to the ingestion endpoint, then go to the mapping screen.

    The API route does the work, so the screen cannot drift from it: a file
    uploaded here and one uploaded through Swagger take the same path.
    """
    from src.api.onboarding.routes import upload

    body = await upload(db, scope, file)
    return RedirectResponse(
        f"/onboarding/uploads/{body.session_id}/review?clinic_id={scope.clinic_id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.post("/uploads/{session_id}/review/structure", include_in_schema=False)
async def review_structure(
    db: SessionDep, session_id: uuid.UUID, scope: ClinicScopeDep, request: Request
) -> RedirectResponse:
    """Approve or decline one structure question from the screen."""
    from src.api.onboarding.routes import answer_structure
    from src.api.onboarding.schemas import StructureAnswerIn

    form = await request.form()
    question_id = str(form.get("id") or "")
    approved = str(form.get("approved") or "") == "yes"
    if question_id:
        await answer_structure(
            db, session_id, scope, StructureAnswerIn(id=question_id, approved=approved)
        )
    return RedirectResponse(
        f"/onboarding/uploads/{session_id}/review?clinic_id={scope.clinic_id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.post("/uploads/{session_id}/review/validate", include_in_schema=False)
async def review_validate(
    db: SessionDep, session_id: uuid.UUID, scope: ClinicScopeDep
) -> RedirectResponse:
    from src.api.onboarding.routes import validate

    await validate(db, session_id, scope)
    return RedirectResponse(
        f"/onboarding/uploads/{session_id}/review?clinic_id={scope.clinic_id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.post("/uploads/{session_id}/review/commit", include_in_schema=False)
async def review_commit(
    db: SessionDep, session_id: uuid.UUID, scope: ClinicScopeDep
) -> RedirectResponse:
    """Import, or show the reason it was refused on the screen itself."""
    from src.api.onboarding.routes import commit

    # A refusal is not an error here: the screen already renders why, from the
    # blocking reasons validate stored, which is more use than a raw 409.
    with contextlib.suppress(HTTPException):
        await commit(db, session_id, scope, save_profile=True)
    return RedirectResponse(
        f"/onboarding/uploads/{session_id}/review?clinic_id={scope.clinic_id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.post("/uploads/{session_id}/review/{sheet}", include_in_schema=False)
async def review_save_mapping(
    db: SessionDep,
    session_id: uuid.UUID,
    sheet: str,
    scope: ClinicScopeDep,
    request: Request,
) -> RedirectResponse:
    """Apply what the reviewer chose on the screen.

    Form fields are prefixed so one submission carries both kinds of answer:
    `col::<column>` is a mapping choice, `ask::<column>` answers a question.
    """
    from src.api.onboarding.routes import set_mapping
    from src.api.onboarding.schemas import MappingIn

    form = await request.form()
    mapping: dict[str, str | None] = {}
    decisions: dict[str, str] = {}
    for key, value in form.multi_items():
        text = str(value).strip()
        if key.startswith("col::"):
            mapping[key[5:]] = text or None
        elif key.startswith("ask::") and text:
            decisions[key[5:]] = text

    entity = str(form.get("entity") or "").strip() or None
    # Typed by a person, so anything that is not a row number is dropped rather
    # than rejected: "17, 18 y 19" should not lose the whole submission.
    excluded = [int(p) for p in re.split(r"[^0-9]+", str(form.get("exclude") or "")) if p]

    await set_mapping(
        db,
        session_id,
        scope,
        MappingIn(
            sheet=sheet,
            mapping=mapping,
            decisions=decisions,
            entity=entity,
            excluded_rows=excluded,
        ),
    )
    # Back to the sheet just saved. Without the anchor every action returns the
    # reviewer to the top of a long page, and on a multi-sheet workbook they
    # scroll back to where they were after each save.
    return RedirectResponse(
        f"/onboarding/uploads/{session_id}/review?clinic_id={scope.clinic_id}#sheet-{_slug(sheet)}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


__all__ = ["router"]
