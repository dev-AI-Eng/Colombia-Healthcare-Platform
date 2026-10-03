"""Validation that is a property of the whole file, not of one cell.

The scope's validation layer is "types, duplicates, referential integrity".
Types are per cell and live in `test_onboarding_normalizers.py`. These two
cannot be: whether a row duplicates another, or points at a doctor that exists,
is only answerable once every row of every sheet has been converted.

Both are guards against an import that looks successful and is wrong — a patient
written twice under one cédula, or an appointment with nobody attending it.
"""

from __future__ import annotations

from src.onboarding import service
from src.onboarding.canonical import Entity


def _patient(number: int, **values: object) -> service.RowResult:
    return service.RowResult(row_number=number, entity=Entity.PATIENT, raw={}, values=values)


def _appointment(number: int, **values: object) -> service.RowResult:
    return service.RowResult(row_number=number, entity=Entity.APPOINTMENT, raw={}, values=values)


# ------------------------------------------------------------------ duplicates
def test_a_repeated_document_is_reported_against_the_row_it_repeats() -> None:
    """Naming the earlier row is what makes the message actionable.

    "3 duplicates" sends a receptionist hunting; "the same cédula as row 2" is
    something they can look at. The first occurrence is the record, not a
    duplicate.
    """
    rows = [
        _patient(2, document_type="CC", document_number="1020304050"),
        _patient(3, document_type="CC", document_number="1020304051"),
        _patient(4, document_type="CC", document_number="1020304050"),
    ]
    found = service.find_duplicates(rows, Entity.PATIENT)

    assert set(found) == {4}
    assert "row 2" in found[4]


def test_the_same_number_under_a_different_document_type_is_not_a_duplicate() -> None:
    """A cédula and a tarjeta de identidad may share digits legitimately.

    Identity is the pair, which is what `_apply_patients` matches on. Keying on
    the number alone would refuse two real, different people.
    """
    rows = [
        _patient(2, document_type="CC", document_number="1020304050"),
        _patient(3, document_type="TI", document_number="1020304050"),
    ]
    assert service.find_duplicates(rows, Entity.PATIENT) == {}


def test_duplicate_detection_ignores_case_and_surrounding_space() -> None:
    """Excel leaves trailing spaces, and clinics write both cc and CC."""
    rows = [
        _patient(2, document_type="CC", document_number="1020304050"),
        _patient(3, document_type="cc", document_number=" 1020304050 "),
    ]
    assert set(service.find_duplicates(rows, Entity.PATIENT)) == {3}


def test_a_row_missing_its_identity_is_not_called_a_duplicate() -> None:
    """An empty cédula is already a per-cell refusal.

    Reporting it again as a duplicate would make the screen noisier without
    saying anything new, and two rows with no cédula are not evidence that they
    are the same person.
    """
    rows = [
        _patient(2, document_type="CC", document_number=""),
        _patient(3, document_type="CC", document_number=""),
    ]
    assert service.find_duplicates(rows, Entity.PATIENT) == {}


# ------------------------------------------------------- referential integrity
def test_an_appointment_naming_an_unknown_doctor_is_reported() -> None:
    """Otherwise it imports as an appointment with nobody attending it."""
    rows = [_appointment(2, doctor_ref="D01"), _appointment(3, doctor_ref="D99")]
    known = {"doctor_ref": {"d01"}, "patient_document": set[str]()}
    found = service.find_dangling_references(rows, Entity.APPOINTMENT, known=known)

    assert set(found) == {3}
    assert "D99" in found[3]


def test_a_reference_satisfied_by_another_sheet_is_accepted() -> None:
    """A workbook that defines its own doctors must not reject its own rows."""
    rows = [_appointment(2, doctor_ref="Dra. Ana Perez")]
    known = {"doctor_ref": {"dra. ana perez"}, "patient_document": set[str]()}
    assert service.find_dangling_references(rows, Entity.APPOINTMENT, known=known) == {}


def test_an_absent_reference_is_left_to_the_per_cell_rules() -> None:
    """A missing required field is already refused by the normalizers.

    This stage answers "does it point at something real", not "is it there".
    """
    rows = [_appointment(2, doctor_ref="")]
    known = {"doctor_ref": set[str](), "patient_document": set[str]()}
    assert service.find_dangling_references(rows, Entity.APPOINTMENT, known=known) == {}


def test_an_entity_with_no_references_is_not_checked() -> None:
    rows = [_patient(2, document_type="CC", document_number="1020304050")]
    known = {"doctor_ref": set[str](), "patient_document": set[str]()}
    assert service.find_dangling_references(rows, Entity.PATIENT, known=known) == {}


def test_two_columns_for_one_name_field_join_in_the_files_order() -> None:
    """The mapping arrives from JSONB, which does not preserve insertion order.

    Iterating the mapping joined "primerApellido" and "segundoApellido" in
    whatever order the database handed back, so a patient whose file says
    "Perez Gomez" was stored as "Gomez Perez" -- the surnames reversed, marked
    valid. The file's own column order is the only order that means anything.
    """
    from src.onboarding.reader import Sheet

    sheet = Sheet(
        name="Pacientes",
        headers=("TIPO DOC", "IDENTIFICACION", "NOMBRES", "PRIMER APELLIDO", "SEGUNDO APELLIDO"),
        rows=(("CC", "1020304050", "Ana", "Perez", "Gomez"),),
        header_row=1,
    )
    # Deliberately scrambled, as a JSONB round trip would return it.
    mapping = {
        "IDENTIFICACION": "document_number",
        "TIPO DOC": "document_type",
        "SEGUNDO APELLIDO": "family_names",
        "NOMBRES": "given_names",
        "PRIMER APELLIDO": "family_names",
    }

    rows, _ = service.validate(sheet, Entity.PATIENT, mapping, decisions={})
    assert rows[0].values["family_names"] == "Perez Gomez"


def test_a_repeated_heading_names_the_first_column_once() -> None:
    """Declining the duplicate question promises "the first of each".

    `{header: position}` keeps the LAST, so the second column was converted --
    twice, because the heading appears twice in `sheet.headers`, which joined a
    shared name field to itself and stored "Ana Ana".
    """
    from src.onboarding.reader import Sheet

    sheet = Sheet(
        name="Pacientes",
        headers=("TIPO DOC", "IDENTIFICACION", "NOMBRES", "NOMBRES", "APELLIDOS"),
        rows=(("CC", "1020304050", "Ana", "Maria", "Perez Gomez"),),
        header_row=1,
    )
    mapping = {
        "TIPO DOC": "document_type",
        "IDENTIFICACION": "document_number",
        "NOMBRES": "given_names",
        "APELLIDOS": "family_names",
    }

    rows, _ = service.validate(sheet, Entity.PATIENT, mapping, decisions={})
    assert rows[0].values["given_names"] == "Ana"


def test_four_name_columns_still_join_after_the_dedupe() -> None:
    """Deduping headings must not break the RIPS four-column join.

    Those are four DIFFERENT headings mapping to two fields, which is the case
    `SHARED_FIELDS` exists for; a repeated heading is one column named twice.
    """
    from src.onboarding.reader import Sheet

    sheet = Sheet(
        name="Pacientes",
        headers=("PRIMER NOMBRE", "SEGUNDO NOMBRE", "PRIMER APELLIDO", "SEGUNDO APELLIDO"),
        rows=(("Carlos", "Andres", "Perez", "Gomez"),),
        header_row=1,
    )
    mapping = {
        "PRIMER NOMBRE": "given_names",
        "SEGUNDO NOMBRE": "given_names",
        "PRIMER APELLIDO": "family_names",
        "SEGUNDO APELLIDO": "family_names",
    }

    rows, _ = service.validate(sheet, Entity.PATIENT, mapping, decisions={})
    assert rows[0].values["given_names"] == "Carlos Andres"
    assert rows[0].values["family_names"] == "Perez Gomez"


# ------------------------------------------------- an empty cell is not a value
# `str(None)` is "None", which is not empty and survives `.strip()`. Both checks
# below treated that as a real value: one told a receptionist a doctor named
# 'None' was missing, the other called two rows duplicates of each other because
# both were missing the same identifier.


def test_an_empty_reference_is_not_reported_as_a_doctor_called_none() -> None:
    rows = [_appointment(2, doctor_ref=None, patient_document="1020304050")]
    dangling = service.find_dangling_references(
        rows,
        Entity.APPOINTMENT,
        known={"doctor_ref": set(), "patient_document": {"1020304050"}},
    )
    assert dangling == {}, dangling


def test_two_rows_missing_the_same_identifier_are_not_duplicates() -> None:
    rows = [
        _patient(2, document_type="CC", document_number=None),
        _patient(3, document_type="CC", document_number=None),
    ]
    assert service.find_duplicates(rows, Entity.PATIENT) == {}


# ------------------------------- a profile belongs to a kind of sheet, not a shape
# The fingerprint keyed on headers alone, so a workbook whose Doctores and
# Especialidades sheets both read ("Nombre", "Codigo") produced one profile for
# both. The next upload then applied the doctor mapping AND the doctor entity to
# the specialties sheet, pre-ticked "confirmed" with nothing missing, so
# specialties were written into the clinic's doctors with no refusal anywhere.


def test_two_sheet_kinds_sharing_headers_do_not_share_a_profile() -> None:
    from src.onboarding.repository import header_fingerprint

    headers = ("Nombre", "Codigo")
    assert header_fingerprint(headers, "doctor") != header_fingerprint(headers, "specialty")


def test_a_profile_still_matches_the_same_sheet_reordered_or_recased() -> None:
    """The reuse the exit criterion depends on must survive the stricter key."""
    from src.onboarding.repository import header_fingerprint

    original = header_fingerprint(("Nombre", "Codigo"), "doctor")
    assert header_fingerprint(("Codigo", "Nombre"), "doctor") == original
    assert header_fingerprint(("NOMBRE", "codigo"), "doctor") == original
    # A genuinely different shape is still reviewed afresh.
    assert header_fingerprint(("Nombre", "Codigo", "Email"), "doctor") != original


def test_a_blank_row_does_not_shift_every_row_number_after_it(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Row numbers must mean the line the reviewer sees in their own file.

    Blank rows are dropped while reading, so counting from the header put every
    later row one out. The hidden-row flag then landed on the row below the
    hidden one: the hidden row imported as valid and a visible row was sent to
    review in its place. `excluded_rows` and every refusal that names a line to
    fix drift the same way.
    """
    from openpyxl import Workbook

    from src.onboarding.reader import read

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.append(["Nombre", "Codigo"])  # row 1
    worksheet.append(["Ana", "A1"])  # row 2
    worksheet.append([None, None])  # row 3, blank and dropped
    worksheet.append(["Luis", "L1"])  # row 4, hidden
    worksheet.append(["Marta", "M1"])  # row 5
    worksheet.row_dimensions[4].hidden = True
    path = tmp_path / "blank.xlsx"
    workbook.save(path)

    result = read(path)
    sheet = result.sheets[0]
    assert sheet.hidden_row_numbers == (4,)
    assert sheet.row_numbers == (2, 4, 5), "rows did not keep their own line numbers"

    report = service.analyse(result)[0]
    rows, _ = service.validate(
        sheet, report.entity, {c.column: c.target_field for c in report.columns}
    )
    # By name, not by number. Under the old counting "Marta" was numbered 4 and
    # was flagged, so a test asserting only the number passed while the wrong
    # patient was the one held back.
    flagged = {row.raw["Nombre"] for row in rows if row.reviews}
    assert flagged == {"Luis"}, f"the hidden row is Luis; the flag landed on {flagged}"
    assert {row.row_number for row in rows} == {2, 4, 5}, (
        "rows are not numbered as the sheet shows them"
    )


# ------------------------------------- a sheet is named after who it holds
# The sheet-name hints knew `medico`, `doctor`, `profesional` and `especialista`
# but not `dentista`, so a dental suite's sheet of practitioners was guessed as
# patients. A doctors sheet read as patients maps almost nothing, because a
# patient has no specialty and no consulting room.


def test_a_sheet_named_after_a_speciality_is_still_a_sheet_of_doctors() -> None:
    from src.onboarding.matcher import guess_entity

    # Headings that say nothing about who the sheet holds, so only the name can
    # decide. With "Nombre Dentista" in the headers the header matcher reaches
    # `doctor` by itself and the test passes whether the hint is there or not.
    headers = ("Codigo", "Nombre", "Area", "Sala")
    for name in ("Dentistas", "Odontologos", "Prestadores", "Terapeutas"):
        entity, reason = guess_entity(name, headers)
        assert entity is Entity.DOCTOR, f"{name!r} was guessed as {entity.value}: {reason}"
        assert name in reason, f"{name!r} resolved by headers rather than by its name"


def test_the_vocabulary_of_a_dental_suite_maps_without_a_model() -> None:
    """Aliases before models, per CLAUDE.md section 8.

    These are the headings a dental suite exports. Every one must resolve from
    the dictionary alone: a column left unmapped here is a reviewer's manual
    decision on every import, forever.
    """
    from src.onboarding.canonical import field_for
    from src.onboarding.matcher import match_sheet

    pairs = (
        (Entity.DOCTOR, ("ID Dentista", "Nombre Dentista", "Especialidad", "Box")),
        (Entity.PATIENT, ("Identificación", "Nombres", "Apellidos", "Celular", "Previsión")),
        (Entity.APPOINTMENT, ("Identificación", "ID Dentista", "Fecha", "Hora Inicio")),
    )
    for entity, headers in pairs:
        mapping = match_sheet(headers, entity)
        unmapped = [p.column for p in mapping.proposals if p.field is None]
        assert not unmapped, f"{entity.value}: {unmapped} needed a model or a person"
        for proposal in mapping.proposals:
            assert proposal.field is not None
            assert field_for(entity, proposal.field.name) is not None


# ---------------------------------------- one row describing more than one thing
# A clinic's sheet is one row per visit as often as it is one row per patient:
# the patient's columns and the appointment's sit side by side, and the patient
# repeats on every row they appear in. The importer assigned one entity per
# sheet, so the columns belonging to the other one were silently dropped --
# not refused, not questioned, just never seen. HubSpot calls the shape "one
# file, multiple objects" and assigns each column an object as well as a field.


def _visit_sheet() -> object:
    from src.onboarding.reader import Sheet

    return Sheet(
        name="Control",
        headers=("T.D.", "CEDULA", "NOMBRE COMPLETO", "CELULAR", "FECHA CITA", "ESTADO"),
        rows=(
            ("CC", "1020304050", "Ana Perez", "3101234567", "2027-03-15", "atendida"),
            ("CC", "1020304051", "Luis Gomez", "3109876543", "2027-03-16", "cancelada"),
        ),
        header_row=1,
    )


def test_a_sheet_of_visits_yields_both_the_patient_and_the_appointment() -> None:
    sheet = _visit_sheet()
    report = service.analyse_sheet_for_test(sheet)  # type: ignore[arg-type]

    targets = {c.column: c.target_field for c in report.columns}
    assert all(targets.values()), f"columns were dropped: {targets}"
    # The sheet's own entity is unqualified; the other one is named.
    assert targets["FECHA CITA"] == "appointment.appointment_date"
    assert targets["ESTADO"] == "appointment.status"

    rows, _ = service.validate(sheet, report.entity, targets)  # type: ignore[arg-type]
    produced = {(r.row_number, r.entity) for r in rows}
    assert len(produced) == 4, f"two source rows should yield four records: {produced}"

    patients = [r for r in rows if r.entity is Entity.PATIENT]
    appointments = [r for r in rows if r.entity is Entity.APPOINTMENT]
    assert len(patients) == len(appointments) == 2
    # Each half carries only its own fields, keyed by the plain field name.
    assert patients[0].values["document_number"] == "1020304050"
    assert "appointment_date" in appointments[0].values
    assert "document_number" not in appointments[0].values


def test_an_unqualified_target_still_means_the_sheets_own_entity() -> None:
    """Every stored profile and hand-written mapping predates the qualified form."""
    assert service.split_target("full_name", Entity.PATIENT) == (Entity.PATIENT, "full_name")
    assert service.split_target("appointment.status", Entity.PATIENT) == (
        Entity.APPOINTMENT,
        "status",
    )
    # An unknown prefix is not an entity, and dropping the column would be worse
    # than treating the name as the sheet's own.
    assert service.split_target("nonsense.thing", Entity.PATIENT) == (
        Entity.PATIENT,
        "nonsense.thing",
    )
