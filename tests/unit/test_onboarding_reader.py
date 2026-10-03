"""The reader turns a clinic file into rows of strings, or refuses it.

These tests run against the generated fixtures (tests/fixtures/onboarding), so
they exercise real workbooks and a real Spanish-Excel CSV rather than mocks.
Each one guards a failure that loses or corrupts patient data silently, which is
the only kind of failure that matters here: a loud refusal costs a re-export,
while a quiet mis-read writes a wrong patient record nothing downstream detects.
"""

from __future__ import annotations

import codecs
import csv
import pickle
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from src.onboarding.reader import (
    MAX_UNCOMPRESSED_BYTES,
    UnreadableFile,
    decode,
    detect_kind,
    inspect_archive,
    read,
    read_isolated,
    sniff_delimiter,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "onboarding"
#: The project root, so a probe subprocess can resolve `src` by `-m`.
_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", autouse=True)
def fixtures_exist() -> None:
    """Generate the fixtures if they are missing; they are not in version control."""
    if not (FIXTURES / "1_clean_ips.xlsx").exists():
        subprocess.run(
            [sys.executable, "-m", "tests.fixtures.onboarding.generate"],
            check=True,
            capture_output=True,
        )


# --------------------------------------------------------------- file identity
def test_an_executable_renamed_csv_is_refused() -> None:
    """The extension is a claim by the uploader; the bytes are the evidence."""
    with pytest.raises(UnreadableFile, match="executable"):
        read(FIXTURES / "5_not_really_csv.csv")


def test_kind_comes_from_the_bytes(tmp_path: Path) -> None:
    xlsx = tmp_path / "claims_to_be.csv"
    xlsx.write_bytes(b"PK\x03\x04rest-of-an-archive")
    assert detect_kind(xlsx) == "xlsx"


# ------------------------------------------------------------------- archives
def test_a_zip_bomb_is_refused_before_it_is_parsed() -> None:
    with pytest.raises(UnreadableFile, match=r"uncompressed|expands"):
        inspect_archive(FIXTURES / "5_zip_bomb.xlsx")


def test_an_xxe_payload_is_refused() -> None:
    """defusedxml blocks the entity; the reader turns that into a clean refusal."""
    with pytest.raises(UnreadableFile):
        read(FIXTURES / "5_xxe.xlsx")


# ------------------------------------------------------------------- encoding
def test_the_encoding_ladder_prefers_utf8_and_falls_back_to_cp1252() -> None:
    # utf-8-sig leads the ladder and decodes plain UTF-8 too, so the encoding
    # label matters less than the text being right and the BOM being gone.
    text, encoding = decode("Pérez".encode())
    assert text == "Pérez"
    assert encoding in {"utf-8", "utf-8-sig"}

    # A BOM must be consumed, or the first header becomes "﻿Tipo" and the
    # column mapping misses it with no error anywhere.
    assert decode("Tipo".encode("utf-8-sig")) == ("Tipo", "utf-8-sig")

    # Windows-1252 is not valid UTF-8, so the ladder must fall through to it
    # rather than mangling every accented name in a Spanish export.
    assert decode("Pérez".encode("cp1252")) == ("Pérez", "cp1252")


def test_spanish_excel_csv_is_read_with_the_right_encoding_and_delimiter() -> None:
    result = read(FIXTURES / "3_excel_csv_es.csv")
    assert result.encoding == "cp1252"
    assert result.delimiter == ";"

    sheet = result.sheets[0]
    assert "NOMBRE COMPLETO" in sheet.headers
    # 7 columns, not 1: a comma-assuming reader collapses this file into a
    # single column and every row becomes one unusable string.
    assert len(sheet.headers) == 7
    # Accented names survived the decoding.
    assert any("Pérez" in cell for row in sheet.rows for cell in row)


def test_a_semicolon_file_with_comma_decimals_is_not_split_on_the_comma() -> None:
    """The comma is the decimal separator in Colombia, not the delimiter.

    Every line here splits plausibly on BOTH characters, and the comma is the
    more frequent one, so a scorer that counts raw frequency the way
    `csv.Sniffer` does picks it. That silently cuts `1.250,50` into two fields
    and shifts every later column by one, which still reads as valid data.
    Consistency is the discriminator: the semicolon gives every line the same
    width, the comma does not.
    """
    sample = (
        "NOMBRE;CELULAR;VALOR;EPS\n"
        "Carlos Pérez;3001234567;1.250,50;Sura\n"
        "María López;3119876543;15.300,75;Nueva EPS\n"
        "Luis Castro;3204567890;900;Sanitas\n"
    )
    assert sniff_delimiter(sample) == ";"

    assert sniff_delimiter("A,B\n1,2\n") == ","  # a real comma file still works


def test_the_delimiter_is_chosen_by_consistency_not_by_field_count() -> None:
    """Addresses contain commas, so the wrong delimiter can look just as good.

    Here both characters yield a most-common width of 3, so field count alone
    cannot choose between them: only the semicolon gives *every* line that
    width. Picking the comma would split "Calle 10 # 5-20, Apto 301" across
    columns and shift the EPS into the address field, which still looks like
    data and passes every later shape check.
    """
    sample = (
        "NOMBRE COMPLETO;DIRECCION;EPS\n"
        "Carlos Pérez Gómez;Calle 10 # 5-20, Apto 301, Torre B;Sura\n"
        "María López Torres;Cra 7 # 12-34, Of. 502;Nueva EPS\n"
        "Luis Castro Vargas;Av 68 # 24-15, Casa 2, Barrio Nuevo;Sanitas\n"
    )
    assert sniff_delimiter(sample) == ";"


def test_a_wider_split_does_not_beat_a_consistent_one() -> None:
    """The wrong delimiter can produce more columns, and still be wrong.

    Commas inside every address split these lines into 4 fields while the real
    delimiter gives 2, so ranking on field count picks the comma and destroys
    the address column. Only the header row disagrees with it, and that single
    disagreement is the whole signal: the semicolon splits every line the same
    way, the comma does not.
    """
    sample = (
        "NOMBRE;DIRECCION\n"
        "Carlos Pérez;Calle 10, Apto 301, Torre B, Bogotá\n"
        "María López;Cra 7, Of. 502, Chapinero, Bogotá\n"
        "Luis Castro;Av 68, Casa 2, Barrio Nuevo, Cali\n"
    )
    assert sniff_delimiter(sample) == ";"


def test_an_embedded_newline_stays_inside_one_field() -> None:
    result = read(FIXTURES / "3_excel_csv_es.csv")
    addresses = [row[-1] for row in result.sheets[0].rows]
    assert any("\n" in address for address in addresses)


def test_a_ragged_row_is_padded_and_reported_not_dropped() -> None:
    """A row missing its trailing fields is usually still a real patient."""
    result = read(FIXTURES / "3_excel_csv_es.csv")
    sheet = result.sheets[0]
    assert all(len(row) == len(sheet.headers) for row in sheet.rows)
    assert any("fields, expected" in warning for warning in sheet.warnings)


# ------------------------------------------------------------------ structure
def test_a_clean_export_reads_every_sheet() -> None:
    result = read(FIXTURES / "1_clean_ips.xlsx")
    assert {sheet.name for sheet in result.sheets} == {"Pacientes", "Medicos", "Citas"}
    patients = next(sheet for sheet in result.sheets if sheet.name == "Pacientes")
    assert patients.header_row == 1
    assert len(patients.rows) == 10


def test_a_header_below_title_rows_is_found() -> None:
    """Hand-made workbooks put a clinic name and a print date above the table."""
    sheet = read(FIXTURES / "2_receptionist.xlsx").sheets[0]
    assert sheet.header_row == 5
    assert sheet.headers[:3] == ("TIPO DOC", "IDENTIFICACION", "NOMBRE COMPLETO")


def test_the_first_table_wins_not_the_tidiest_one() -> None:
    """A second table pasted below scores better precisely because it is smaller.

    Taking the best-scoring row anywhere on the sheet skips the real data: this
    fixture's waiting-list table would be read instead of the 12-patient agenda,
    losing every patient without an error.
    """
    sheet = read(FIXTURES / "2_receptionist.xlsx").sheets[0]
    assert len(sheet.rows) >= 12
    assert "NOMBRE COMPLETO" in sheet.headers  # the agenda's header, not "NOMBRE"


def test_hidden_rows_are_imported_and_flagged() -> None:
    """Hiding a row is not deleting it; dropping it silently loses a patient."""
    sheet = read(FIXTURES / "2_receptionist.xlsx").sheets[0]
    assert sheet.hidden_row_numbers == (15, 16)
    assert any("hidden row" in warning for warning in sheet.warnings)


def test_hidden_columns_and_hidden_sheets_are_flagged() -> None:
    result = read(FIXTURES / "2_receptionist.xlsx")
    agenda = result.sheets[0]
    assert "J" in agenda.hidden_columns
    assert "COD_EPS_INTERNO" in agenda.headers
    assert any("hidden" in warning.lower() for warning in result.warnings)


def test_a_lying_dimension_does_not_truncate_the_import() -> None:
    """The declared dimension is a claim by the file, and this one is false.

    openpyxl's read-only mode trusts `<dimension ref="A1:A1"/>` and yields one
    cell for a sheet holding 5,000 rows. Without `reset_dimensions()` the import
    would quietly drop 4,999 patients.
    """
    sheet = read(FIXTURES / "5_lying_dimension.xlsx").sheets[0]
    assert len(sheet.rows) == 5000


def test_an_uncalculated_formula_is_kept_not_read_as_empty() -> None:
    """`data_only=True` returns None for a formula the file never calculated.

    A clinic workbook built on formulas would otherwise import as blank columns
    with no error at all. Keeping the formula text means validation can reject
    the cell, and the export layer can neutralise an injected payload.
    """
    sheet = read(FIXTURES / "5_formula_injection.xlsx").sheets[0]
    assert sheet.rows[0][1].startswith("=")
    assert any("formula" in warning for warning in sheet.warnings)


# ------------------------------------------------------------------- no typing
def test_values_are_returned_as_text_without_inference() -> None:
    """Conversion belongs to the confirmed mapping, never to the reader.

    The damage in this fixture must still be visible downstream: a document
    number already reduced to scientific notation has to reach validation as
    such so the row can be rejected, not be rounded into a plausible number.
    """
    sheet = read(FIXTURES / "4_corrupted.xlsx").sheets[0]
    assert all(isinstance(cell, str) for row in sheet.rows for cell in row)

    columns = {name: index for index, name in enumerate(sheet.headers)}
    first = sheet.rows[0]
    # A cédula whose leading zero Excel already destroyed stays as stored.
    assert first[columns["IDENTIFICACION"]] == "123456"
    # The Spanish 12-hour form keeps its non-breaking space for the normalizer.
    assert " " in first[columns["HORA"]]


# ----------------------------------------------------------- refusal paths
# Coverage showed these branches untested. They are the ones that decide whether
# a malformed upload stops with a message a receptionist can act on, or crashes
# with a stack trace the API cannot report.
def test_an_empty_file_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(UnreadableFile, match="empty"):
        read(empty)


def test_a_workbook_with_no_rows_is_refused(tmp_path: Path) -> None:
    from openpyxl import Workbook

    path = tmp_path / "blank.xlsx"
    Workbook().save(path)
    with pytest.raises(UnreadableFile, match="no readable sheet"):
        read(path)


def test_a_truncated_archive_is_refused_not_crashed(tmp_path: Path) -> None:
    """The magic bytes claim an archive; the contents are not one.

    Without this the reader raises BadZipFile, which the upload endpoint cannot
    turn into an explanation for the person who uploaded the file.
    """
    path = tmp_path / "truncated.xlsx"
    path.write_bytes(b"PK\x03\x04" + b"nonsense" * 20)
    with pytest.raises(UnreadableFile, match="not a valid archive"):
        read(path)


def test_a_single_column_file_is_read_not_refused(tmp_path: Path) -> None:
    """A list of document numbers is a legitimate import with no delimiter."""
    path = tmp_path / "one.csv"
    path.write_text("DOCUMENTO\n1045678901\n1023456789\n", encoding="utf-8")
    sheet = read(path).sheets[0]
    assert sheet.headers == ("DOCUMENTO",)
    assert len(sheet.rows) == 2


def test_legacy_xls_is_refused_with_advice(tmp_path: Path) -> None:
    """The OLE2 format needs a different parser; say so rather than failing oddly."""
    path = tmp_path / "old.xls"
    path.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64)
    with pytest.raises(UnreadableFile, match="xlsx"):
        read(path)


def test_a_header_with_an_unlabelled_column_keeps_every_row(tmp_path: Path) -> None:
    """A blank header cell made the header row look narrower than the data.

    The first patient was then promoted to column names and lost, and every
    column was mislabelled, with no error anywhere.
    """
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.append(["NOMBRE", "", None, "EPS"])
    sheet.append(["Ana Gómez", "x", "y", "Sura"])
    sheet.append(["Luis Castro", "p", "q", "Nueva EPS"])
    path = tmp_path / "gap.xlsx"
    workbook.save(path)

    result = read(path).sheets[0]
    assert result.header_row == 1
    assert result.headers[0] == "NOMBRE"
    assert len(result.rows) == 2


# --------------------------------------------------- broad CSV recognition
# Every case here is a real clinic export shape. Two of them used to raise
# `_csv.Error` straight out of the stdlib, and one used to be read as mojibake.


def _csv(tmp_path: Path, raw: bytes, name: str = "export.csv") -> Path:
    path = tmp_path / name
    path.write_bytes(raw)
    return path


@pytest.mark.parametrize(
    ("label", "raw", "delimiter"),
    [
        ("semicolon", b"A;B;C\n1;2;3\n", ";"),
        ("comma", b"A,B,C\n1,2,3\n", ","),
        ("tab", b"A\tB\tC\n1\t2\t3\n", "\t"),
        ("pipe", b"A|B|C\n1|2|3\n", "|"),
    ],
)
def test_every_delimiter_excel_writes_is_recognised(
    tmp_path: Path, label: str, raw: bytes, delimiter: str
) -> None:
    result = read(_csv(tmp_path, raw))
    assert result.delimiter == delimiter
    assert result.sheets[0].headers == ("A", "B", "C")
    assert result.sheets[0].rows == (("1", "2", "3"),)


@pytest.mark.parametrize(
    ("label", "ending"),
    [("unix", b"\n"), ("windows", b"\r\n"), ("classic_mac", b"\r")],
)
def test_every_line_ending_is_read(tmp_path: Path, label: str, ending: bytes) -> None:
    """A bare CR is what classic Mac Excel writes, and `csv` refuses it outright.

    It reached the caller as an opaque `_csv.Error` about newlines in unquoted
    fields. A CR outside CRLF is unambiguously a line ending, so it is
    normalised rather than reported.
    """
    raw = ending.join([b"A;B", b"1;2", b""])
    sheet = read(_csv(tmp_path, raw)).sheets[0]
    assert sheet.headers == ("A", "B")
    assert sheet.rows == (("1", "2"),)


@pytest.mark.parametrize("encoding", ["utf-16", "utf-16-le", "utf-16-be", "utf-32"])
def test_utf16_is_decoded_not_mangled(tmp_path: Path, encoding: str) -> None:
    """Excel writes UTF-16LE for "Unicode Text", and cp1252 "succeeds" on it.

    The encoding ladder cannot fail its way past a single-byte codec, so a
    UTF-16 file was decoded into text full of NULs and imported as garbage: the
    headers came out as 'T\x00I\x00P\x00O'. A byte-order mark is now trusted
    over the ladder.
    """
    text = "NOMBRE;CIUDAD\nJosé Muñoz;Medellín\n"
    raw = text.encode(encoding)
    if not raw.startswith(codecs.BOM_UTF16) and not raw.startswith(codecs.BOM_UTF32_LE):
        raw = ("\ufeff" + text).encode(encoding)

    sheet = read(_csv(tmp_path, raw)).sheets[0]
    assert sheet.headers == ("NOMBRE", "CIUDAD")
    assert sheet.rows == (("José Muñoz", "Medellín"),)


def test_a_field_larger_than_the_stdlib_limit_is_read(tmp_path: Path) -> None:
    """A pasted clinical note exceeds csv's 128 KiB field limit.

    It raised `_csv.Error: field larger than field limit`, which reached the
    caller as a 500 rather than a message about the file.
    """
    note = "x" * 200_000
    sheet = read(_csv(tmp_path, f"A;NOTA\n1;{note}\n".encode())).sheets[0]
    assert sheet.rows[0][1] == note


def test_the_stdlib_field_limit_is_restored_afterwards(tmp_path: Path) -> None:
    """It is process-global, so leaving it raised would weaken every other caller."""
    before = csv.field_size_limit()
    read(_csv(tmp_path, b"A;B\n1;2\n"))
    assert csv.field_size_limit() == before


def test_a_spanish_decimal_column_does_not_split_on_the_comma(tmp_path: Path) -> None:
    """`csv.Sniffer` reads `1.250,50` as a comma-separated pair."""
    raw = b"CONCEPTO;VALOR\nConsulta;1.250,50\nControl;980,00\n"
    sheet = read(_csv(tmp_path, raw)).sheets[0]
    assert sheet.headers == ("CONCEPTO", "VALOR")
    assert sheet.rows[0] == ("Consulta", "1.250,50")


def test_delimiters_inside_quotes_do_not_split(tmp_path: Path) -> None:
    raw = b'NOMBRE;NOTA\n"Perez, Ana";"a;b;c"\n'
    sheet = read(_csv(tmp_path, raw)).sheets[0]
    assert sheet.rows[0] == ("Perez, Ana", "a;b;c")


def test_a_newline_inside_a_quoted_field_stays_in_the_field(tmp_path: Path) -> None:
    raw = b'NOMBRE;DIRECCION\n"Ana";"Calle 5\nApto 4"\n'
    sheet = read(_csv(tmp_path, raw)).sheets[0]
    assert len(sheet.rows) == 1
    assert sheet.rows[0][1] == "Calle 5\nApto 4"


def test_an_empty_file_is_refused_clearly(tmp_path: Path) -> None:
    with pytest.raises(UnreadableFile, match="empty"):
        read(_csv(tmp_path, b"   \n\n"))


# ------------------------------------------- what the file cannot decide alone


def test_a_title_above_the_table_is_asked_about_not_guessed(tmp_path: Path) -> None:
    """Reading the title as the header loses every patient in the file.

    Row 1 is a clinic name, so taking it on faith yields a one-column table and
    every real row looks ragged. Which row holds the headings is a judgement, so
    it is asked. Declining reads the file exactly as written.
    """
    raw = (
        b"LISTADO DE PACIENTES CLINICA X\n\n"
        b"TIPO DOC;IDENTIFICACION;NOMBRES\n"
        b"CC;1020304050;Ana\nCC;1020304051;Luis\n"
    )
    path = _csv(tmp_path, raw)

    asked = read(path)
    assert [q.id for q in asked.questions] == ["csv.header_row"]

    # Unanswered: the file as written, which is what declining also gives.
    assert asked.sheets[0].headers == ("LISTADO DE PACIENTES CLINICA X",)

    approved = read(path, {"csv.header_row": True})
    assert approved.sheets[0].headers == ("TIPO DOC", "IDENTIFICACION", "NOMBRES")
    assert len(approved.sheets[0].rows) == 2
    assert approved.sheets[0].header_row == 3

    declined = read(path, {"csv.header_row": False})
    assert declined.sheets[0].headers == ("LISTADO DE PACIENTES CLINICA X",)


def test_only_one_question_is_asked_for_one_cause(tmp_path: Path) -> None:
    """A title line makes every row look overlong, which is the same problem.

    Asking about both would show a second question blaming "an unquoted
    delimiter" for something the first question explains, and a reviewer who
    answers the misleading one first gets a worse file.
    """
    raw = b"CLINICA X\n\nA;B;C\n1;2;3\n"
    assert [q.id for q in read(_csv(tmp_path, raw)).questions] == ["csv.header_row"]


def test_a_row_with_too_many_fields_is_asked_about(tmp_path: Path) -> None:
    """Truncating drops real cells silently; declining leaves the row out.

    Neither is obviously right: the extra field is usually an unquoted delimiter
    inside an address, so cutting it loses part of the address, while dropping
    the row loses the patient. The reviewer decides which.
    """
    path = _csv(tmp_path, b"A;B;C\n1;2;3\n4;5;6;7\n8;9;10\n")

    asked = read(path)
    assert [q.id for q in asked.questions] == ["csv.overlong_rows"]
    # Until answered the row is held back rather than quietly cut.
    assert [r[0] for r in asked.sheets[0].rows] == ["1", "8"]

    approved = read(path, {"csv.overlong_rows": True})
    assert [r[0] for r in approved.sheets[0].rows] == ["1", "4", "8"]
    assert approved.sheets[0].rows[1] == ("4", "5", "6")  # the 7 is dropped

    declined = read(path, {"csv.overlong_rows": False})
    assert [r[0] for r in declined.sheets[0].rows] == ["1", "8"]


def test_two_columns_with_one_name_are_asked_about(tmp_path: Path) -> None:
    """A field may be mapped from one column only, so the second would be lost.

    Renaming is not done silently because a mapping the clinic confirmed earlier
    refers to the original spelling.
    """
    path = _csv(tmp_path, b"NOMBRE;NOMBRE;CELULAR\nAna;Perez;3101234567\n")

    asked = read(path)
    assert [q.id for q in asked.questions] == ["csv.duplicate_headers"]

    approved = read(path, {"csv.duplicate_headers": True})
    assert approved.sheets[0].headers == ("NOMBRE", "NOMBRE (2)", "CELULAR")

    declined = read(path, {"csv.duplicate_headers": False})
    assert declined.sheets[0].headers == ("NOMBRE", "NOMBRE", "CELULAR")


def test_declining_never_loses_more_than_approving(tmp_path: Path) -> None:
    """The safety property behind the whole design.

    A reviewer who does not understand a question must be able to decline it
    without destroying data. Declining is always the reading the importer would
    have used with no question asked, so it can lose rows it cannot place, but it
    never silently alters a value.
    """
    path = _csv(tmp_path, b"A;B\n1;2\n3;4;5\n")
    declined = read(path, {"csv.overlong_rows": False})
    approved = read(path, {"csv.overlong_rows": True})

    # Approving is the one that discards data (the trailing 5), and it says so.
    assert len(approved.sheets[0].rows) > len(declined.sheets[0].rows)
    assert any("expected 2" in w for w in approved.sheets[0].warnings)


def test_a_row_number_means_the_line_in_the_reviewer_s_file(tmp_path: Path) -> None:
    """Blank lines are skipped, but numbering must not be.

    A row number is how a refusal points a receptionist at the line to fix, and
    it is what `excluded_rows` names on the confirmation screen. Counting
    compacted rows instead of file lines makes both point at the wrong line.
    """
    raw = b"CLINICA X\n\nA;B\n\n1;2\n\n\n3;4\n"
    sheet = read(_csv(tmp_path, raw), {"csv.header_row": True}).sheets[0]

    assert sheet.header_row == 3  # the third line of the file
    assert sheet.headers == ("A", "B")
    assert sheet.rows == (("1", "2"), ("3", "4"))


def test_a_warning_names_the_real_line_number(tmp_path: Path) -> None:
    raw = b"A;B\n\n1;2;3\n"
    sheet = read(_csv(tmp_path, raw), {"csv.overlong_rows": True}).sheets[0]
    # The ragged row is line 3, not line 2.
    assert any("Row 3" in warning for warning in sheet.warnings)


# ------------------------------------------- parsing in a throwaway process
# The parsing libraries are the importer's largest attack surface: the file
# arrives from outside, and zipfile, the XML parser and openpyxl are C or
# C-adjacent code reading attacker-shaped input. The guards above refuse the
# attacks we know about; isolation bounds the ones we do not.


def test_an_isolated_read_returns_the_same_result_as_an_inline_one() -> None:
    """Isolation must not change what is read, only where it runs."""
    path = FIXTURES / "1_clean_ips.xlsx"
    assert read_isolated(path) == read(path)


def test_an_isolated_read_carries_structure_questions_back(tmp_path: Path) -> None:
    """The questions cross the process boundary, or a CSV cannot be confirmed."""
    path = tmp_path / "preamble.csv"
    path.write_bytes(b"CLINICA X\n\nA;B;C\n1;2;3\n")

    asked = read_isolated(path)
    assert [q.id for q in asked.questions] == ["csv.header_row"]

    approved = read_isolated(path, {"csv.header_row": True})
    assert approved.sheets[0].headers == ("A", "B", "C")
    assert approved.sheets[0].rows == (("1", "2", "3"),)


def test_a_refusal_from_the_child_reaches_the_caller_verbatim() -> None:
    """A refusal is a decision, not a crash, and its wording is for a person.

    The worker exits with its own code for this so the parent can tell a file it
    declined to read from a parse that died.
    """
    with pytest.raises(UnreadableFile, match=r"uncompressed|expands"):
        read_isolated(FIXTURES / "5_zip_bomb.xlsx")


def test_the_worker_exit_code_matches_the_one_the_parent_expects() -> None:
    """The two constants are duplicated to avoid an import cycle, so pin them.

    The worker imports the reader, so the reader cannot import the worker. If
    they ever disagree, a refusal would be reported as a crash.
    """
    from src.onboarding import parse_worker, reader

    assert parse_worker.EXIT_UNREADABLE == reader.EXIT_UNREADABLE


def test_a_child_that_dies_is_reported_as_an_unreadable_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A segfault or a kill must not reach the reviewer as a 500.

    This is the case isolation exists for: a library failing in a way no guard
    of ours can catch. The child is replaced with one that dies, because making
    openpyxl actually segfault is not something a test can rely on.
    """
    from src.onboarding import reader as reader_module

    path = tmp_path / "x.csv"
    path.write_bytes(b"A;B\n1;2\n")

    # The dead child also emits a usable pickle. A reader that only checked
    # whether stdout parses would accept this; only the exit code says the parse
    # died, and half-written output from a killed process is not a result.
    plausible = pickle.dumps(read(path))

    def _dies(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            args=[], returncode=-11, stdout=plausible, stderr=b"Segmentation fault"
        )

    monkeypatch.setattr(reader_module.subprocess, "run", _dies)
    with pytest.raises(UnreadableFile, match="could not be read"):
        read_isolated(path)


def test_a_parse_that_never_finishes_is_stopped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A parse that hangs must not hold an API worker forever."""
    from src.onboarding import reader as reader_module

    path = tmp_path / "x.csv"
    path.write_bytes(b"A;B\n1;2\n")

    seen: dict[str, object] = {}

    def _hangs(*args: object, **kwargs: object) -> None:
        # Recorded, so this asserts that a timeout was requested rather than
        # only that we handle the exception when something else raises it.
        seen.update(kwargs)
        raise subprocess.TimeoutExpired(cmd="parse", timeout=1)

    monkeypatch.setattr(reader_module.subprocess, "run", _hangs)
    with pytest.raises(UnreadableFile, match="longer than"):
        read_isolated(path)

    assert seen.get("timeout") == reader_module.PARSE_TIMEOUT_SECONDS


def test_a_truncated_stream_from_the_child_is_not_trusted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A library printing to stdout would corrupt the pickle; that is not a result."""
    from src.onboarding import reader as reader_module

    path = tmp_path / "x.csv"
    path.write_bytes(b"A;B\n1;2\n")

    def _garbage(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout=b"not a pickle", stderr=b""
        )

    monkeypatch.setattr(reader_module.subprocess, "run", _garbage)
    with pytest.raises(UnreadableFile, match="could not be read"):
        read_isolated(path)


def test_the_child_does_not_import_openpyxl_for_a_csv(tmp_path: Path) -> None:
    """openpyxl pulls in numpy, about half a second of startup per process.

    Every parse now runs in its own process, so a CSV upload paying for a
    library it never touches would be a cost on every import. This asserts the
    laziness that avoids it rather than trusting a comment about it.
    """
    path = tmp_path / "x.csv"
    path.write_bytes(b"A;B\n1;2\n")
    probe = (
        f"import sys; sys.argv = ['w', {str(path)!r}];"
        "from src.onboarding import parse_worker;"
        "parse_worker.main(sys.argv[1:]);"
        "sys.stderr.write('OPENPYXL' if 'openpyxl' in sys.modules else 'CLEAN')"
    )
    done = subprocess.run(  # noqa: S603  (fixed argv, no shell)
        [sys.executable, "-c", probe], capture_output=True, check=True, cwd=_ROOT
    )
    assert done.stderr.decode().endswith("CLEAN")


# ------------------------------------- found by adversarial probing
# Both of these were silent: no crash, no refusal, no warning, wrong data.
# That is the only failure class that reaches a clinic unnoticed.


@pytest.mark.parametrize("encoding", ["utf-16-le", "utf-16-be"])
def test_utf16_without_a_bom_is_decoded_not_mangled(tmp_path: Path, encoding: str) -> None:
    """Excel's UTF-16 carries a BOM; a programmatic export may not.

    Every single-byte codec "succeeds" on UTF-16 bytes, so the encoding ladder
    cannot fail its way past cp1252: the file was read as text full of NULs and
    imported as garbage. Byte order comes from where the NULs sit, because a
    little-endian reading of big-endian bytes yields CJK characters rather than
    raising -- "it decoded" is not evidence of the right order.
    """
    text = "TIPO DOC;IDENTIFICACION;NOMBRES\nCC;1020304050;José Muñoz\n"
    sheet = read(_csv(tmp_path, text.encode(encoding))).sheets[0]

    assert sheet.headers == ("TIPO DOC", "IDENTIFICACION", "NOMBRES")
    assert sheet.rows == (("CC", "1020304050", "José Muñoz"),)


def test_a_utf16_file_with_a_bom_keeps_no_bom_in_its_first_header(
    tmp_path: Path,
) -> None:
    """The BOM must be consumed by a codec that strips it.

    `utf-16-le` consumes no BOM, so probing for byte order before the BOM check
    left U+FEFF glued to the first heading, which then matched no alias. The
    probe runs after the BOM loop for that reason.
    """
    text = "TIPO DOC;IDENTIFICACION\nCC;1020304050\n"
    sheet = read(_csv(tmp_path, text.encode("utf-16"))).sheets[0]

    assert sheet.headers == ("TIPO DOC", "IDENTIFICACION")
    assert not sheet.headers[0].startswith("\ufeff")


def test_a_file_with_no_heading_row_is_asked_about(tmp_path: Path) -> None:
    """Otherwise the first patient is silently consumed as column names.

    Nothing in the file settles it: a clinic exporting without headings and one
    whose headings happen to be words look identical. Approving names the
    columns by position so every row stays data; declining reads the first row
    as headings, which is what the importer would have done unasked.
    """
    path = _csv(tmp_path, b"CC;1020304050;Ana Perez\nCC;1020304051;Luis Gomez\n")

    asked = read(path)
    assert [q.id for q in asked.questions] == ["csv.no_header_row"]
    # Unanswered, the first row is still taken as headings -- so the question is
    # what stands between that reading and a committed import.
    assert len(asked.sheets[0].rows) == 1

    approved = read(path, {"csv.no_header_row": True})
    assert approved.sheets[0].headers == ("column 1", "column 2", "column 3")
    assert len(approved.sheets[0].rows) == 2
    assert approved.sheets[0].rows[0] == ("CC", "1020304050", "Ana Perez")

    declined = read(path, {"csv.no_header_row": False})
    assert declined.sheets[0].headers == ("CC", "1020304050", "Ana Perez")


def test_a_blank_heading_row_does_not_eat_the_first_patient(tmp_path: Path) -> None:
    """A present-but-empty heading row was the original symptom.

    Cells are stripped before blank rows are dropped, so "  ;  ;  " vanishes and
    row 2 became the headings with no question asked at all.
    """
    path = _csv(tmp_path, b"  ;  ;  \n1;2;3\n4;5;6\n")

    assert [q.id for q in read(path).questions] == ["csv.no_header_row"]
    approved = read(path, {"csv.no_header_row": True})
    assert [list(r) for r in approved.sheets[0].rows] == [["1", "2", "3"], ["4", "5", "6"]]


@pytest.mark.parametrize(
    ("label", "raw"),
    [
        ("spanish", b"TIPO DOC;IDENTIFICACION;NOMBRES\nCC;1020304050;Ana\n"),
        ("rips", b"tipoDocumentoIdentificacion;numDocumentoIdentificacion\nCC;102030\n"),
        ("two name columns", b"NOMBRES;APELLIDOS\nAna;Perez Gomez\n"),
        ("single column", b"IDENTIFICACION\n1020304050\n"),
        ("short labels", b"A;B;C\n1;2;3\n"),
    ],
)
def test_a_real_heading_row_is_never_questioned(tmp_path: Path, label: str, raw: bytes) -> None:
    """A question nobody needs teaches reviewers to click past the one that matters.

    A one-column file is exempt outright: `_header_score` returns 0 below two
    filled cells, and there is nothing useful to ask about a single column a
    reviewer can see.
    """
    assert read(_csv(tmp_path, raw)).questions == ()


def test_a_repeated_heading_is_not_mistaken_for_a_missing_one(tmp_path: Path) -> None:
    """One cause, one question.

    A repeated heading costs distinctness, which dropped the row below the
    header floor and raised "this file has no headings" alongside the duplicate
    question. A reviewer answering the wrong one first gets a worse file, so the
    floor is scored on distinct values: repetition is a naming problem, not
    evidence about the shape.
    """
    path = _csv(tmp_path, b"NOMBRE;NOMBRE;CELULAR\nAna;Perez;3101234567\n")
    assert [q.id for q in read(path).questions] == ["csv.duplicate_headers"]


# ------------------------------------- one archive guard per test
# The zip-bomb fixture cannot reach these guards: it has no workbook part, so it
# is refused as unreadable first. Each archive below trips exactly one guard.


def _archive(tmp_path: Path, members: dict[str, bytes], name: str = "a.xlsx") -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member, payload in members.items():
            archive.writestr(member, payload)
    return path


def test_a_declared_uncompressed_size_over_the_limit_is_refused(tmp_path: Path) -> None:
    """The cheap check: the central directory says how big it unpacks to."""
    oversized = b"\0" * (MAX_UNCOMPRESSED_BYTES + 1024)
    path = _archive(tmp_path, {"xl/workbook.xml": oversized})

    with pytest.raises(UnreadableFile, match="uncompressed"):
        inspect_archive(path)


def test_a_compression_ratio_over_the_limit_is_refused(tmp_path: Path) -> None:
    """A small file that unpacks enormously, under the absolute size cap.

    Sized to stay below MAX_UNCOMPRESSED_BYTES so only the ratio can refuse it;
    zeros compress far past the ratio limit.
    """
    payload = b"\0" * (MAX_UNCOMPRESSED_BYTES // 2)
    path = _archive(tmp_path, {"xl/workbook.xml": payload})

    with pytest.raises(UnreadableFile, match=r"expands|ratio"):
        inspect_archive(path)


def test_too_many_members_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A workbook has tens of parts, not thousands; thousands is a bomb.

    The cap is lowered for the test rather than the archive built past the real
    one. Building 2,001 members works, but if the cap is ever raised the test
    would build enough tiny files to trip the *ratio* guard instead and pass for
    the wrong reason -- which is exactly what it did on the first attempt.
    """
    monkeypatch.setattr("src.onboarding.reader.MAX_ARCHIVE_MEMBERS", 10)
    members = {f"xl/worksheets/sheet{i}.xml": b"<x/>" for i in range(11)}
    path = _archive(tmp_path, members)

    with pytest.raises(UnreadableFile, match=r"parts|members"):
        inspect_archive(path)


def test_an_ordinary_workbook_passes_every_archive_guard(tmp_path: Path) -> None:
    """The guards must not refuse a real file; this is what makes them usable."""
    inspect_archive(FIXTURES / "1_clean_ips.xlsx")


def test_a_workbook_with_no_heading_row_is_asked_about(tmp_path: Path) -> None:
    """The CSV-only fix left the Excel half of the same defect open.

    A workbook whose row 1 is data had that patient read as the column names:
    lost from the data, and -- because `_ask_model` keys off `sheet.questions` --
    still sent to the mapping model as if it were a heading. Same question id as
    the CSV path, because it is the same question about the same thing.
    """
    from openpyxl import Workbook

    path = tmp_path / "headerless.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = "Pacientes"
    sheet.append(["CC", "1020304050", "Ana Perez Gomez", "3101234567"])
    sheet.append(["CC", "1020304051", "Luis Gomez Diaz", "3151112233"])
    workbook.save(path)

    asked = read(path)
    assert [q.id for q in asked.questions] == ["csv.no_header_row"]
    # Unanswered, row 1 is still taken as headings, which is what the question
    # exists to stand between and a committed import.
    assert len(asked.sheets[0].rows) == 1

    approved = read(path, {"csv.no_header_row": True})
    assert approved.sheets[0].headers == ("column 1", "column 2", "column 3", "column 4")
    assert len(approved.sheets[0].rows) == 2
    assert approved.sheets[0].rows[0][2] == "Ana Perez Gomez"

    declined = read(path, {"csv.no_header_row": False})
    assert declined.sheets[0].headers[2] == "Ana Perez Gomez"


def test_a_workbook_with_real_headings_is_not_questioned(tmp_path: Path) -> None:
    """A question nobody needs teaches reviewers to click past the real one."""
    from openpyxl import Workbook

    path = tmp_path / "headed.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = "Pacientes"
    sheet.append(["TIPO DOC", "IDENTIFICACION", "NOMBRES", "APELLIDOS"])
    sheet.append(["CC", "1020304050", "Ana", "Perez Gomez"])
    workbook.save(path)

    assert read(path).questions == ()


# ------------------- what makes a heading safe to send to a model
# Gating the model stage on "any question at all" closed the leak but silenced
# an in-scope feature for files whose headings are genuine. Only a question
# about WHICH ROW holds the headings means a heading might be a patient.


def test_a_duplicate_heading_question_does_not_unsettle_the_headings(
    tmp_path: Path,
) -> None:
    """Duplicate headings are a naming problem; the headings are still headings.

    Four columns, so this trips only the duplicate question: a three-column
    file with a repeated heading also scores low enough to raise
    `csv.header_row`, which legitimately does unsettle them.
    """
    path = _csv(
        tmp_path,
        b"TIPO DOC;IDENTIFICACION;NOMBRE;NOMBRE\nCC;1020304050;Ana;Perez\n",
    )
    result = read(path)

    assert [q.id for q in result.questions] == ["csv.duplicate_headers"]
    assert result.sheets[0].headings_are_settled


def test_an_open_header_question_unsettles_the_headings(tmp_path: Path) -> None:
    path = _csv(tmp_path, b"CC;1020304050;Ana Perez\nCC;1020304051;Luis Gomez\n")
    assert read(path).sheets[0].headings_are_settled is False


def test_declining_a_header_question_does_not_settle_the_headings(
    tmp_path: Path,
) -> None:
    """Declining is the reviewer asserting what the file could not show.

    They may be right, but the cost of being wrong is a patient's cédula leaving
    the machine, and the dictionary and fuzzy stages run on those columns either
    way. Approving does settle them, because the headings are then positional
    names we generated.
    """
    path = _csv(tmp_path, b"CC;1020304050;Ana Perez\n")
    assert read(path, {"csv.no_header_row": False}).sheets[0].headings_are_settled is False
    assert read(path, {"csv.no_header_row": True}).sheets[0].headings_are_settled is True


def test_a_question_records_whether_it_was_answered(tmp_path: Path) -> None:
    """ "Still listed" is not "still open": a reviewer sees what they decided."""
    path = _csv(tmp_path, b"CC;1020304050;Ana Perez\n")

    assert read(path).questions[0].answered is None
    assert read(path, {"csv.no_header_row": True}).questions[0].answered is True
    assert read(path, {"csv.no_header_row": False}).questions[0].answered is False


def test_a_positional_export_is_offered_the_headerless_question(tmp_path: Path) -> None:
    """A file with no headings anywhere must be recognised as having none.

    The national RIPS archivo is positional: every row is a patient and no row
    is a label. `_find_csv_header` returned the least-bad row regardless of
    score, so a file where nothing clears the floor was reported as having five
    title lines above a header on row 6. The headerless question was then never
    asked -- it is gated on "no preamble found" -- and the only readings on offer
    discarded five patients or one.
    """
    raw = (
        b"CC,1045678901,EPS001,1,Perez,Gomez,Carlos,Andres\n"
        b"CC,1023456789,EPS002,1,Lopez,Torres,Maria,Fernanda\n"
        b"TI,1102345678,EPS005,2,Ortiz,Mejia,Luisa,Fernanda\n"
        b"CC,71234567,EPS008,1,Castro,Vargas,Luis,Ernesto\n"
        b"RC,1140567890,EPS010,2,Suarez,Velez,Andres,Felipe\n"
        b"CE,E0456789,EPS001,1,Dubois,Martin,Jean,Pierre\n"
    )
    path = _csv(tmp_path, raw)

    result = read(path)
    assert "csv.no_header_row" in [q.id for q in result.questions], (
        "a file with no headings was not offered the headerless question"
    )
    assert result.sheets[0].headings_are_settled is False

    # Approving keeps every row, which is the whole point.
    approved = read(path, {"csv.no_header_row": True})
    assert len(approved.sheets[0].rows) == 6, "approving the headerless reading lost rows"
    assert approved.sheets[0].headings_are_settled is True


def test_a_real_heading_row_still_needs_no_question(tmp_path: Path) -> None:
    """The margin must not make every ordinary file ask.

    A genuine heading beats its own data comfortably, because data carries
    numbers, dates and repeated values and a heading does not.
    """
    path = _csv(
        tmp_path,
        b"TIPO DOC;IDENTIFICACION;NOMBRES;CELULAR\n"
        b"CC;1020304050;Ana Perez;3101234567\n"
        b"CC;1020304051;Luis Gomez;3109876543\n",
    )
    result = read(path)
    assert [q.id for q in result.questions] == []
    assert result.sheets[0].headers[0] == "TIPO DOC"
