"""Four exports in the shapes real clinic systems produce.

    python -m tests.fixtures.onboarding.generate_platform_exports

Every value here is invented. No file in this directory contains real patient
data, and none may: see CLAUDE.md rule 7. The cédulas are sequential, the
mobiles are textbook 310/320 numbers, and the names are common enough to belong
to nobody in particular.

These complement `generate.py`, which escalates from a tidy file to one built to
break the parser. These four are the ordinary middle: files a clinic would
actually hand over, each modelled on a different system's way of thinking.

    6_rips_us.csv           the national reporting format, field for field
    7_agenda_citas.xlsx     a scheduling export: one row per appointment
    8_odontologia.xlsx      a dental-suite style workbook, three sheets
    9_hoja_recepcion.xlsx   a spreadsheet the front desk maintains by hand

WHAT IS SOURCED AND WHAT IS NOT
-------------------------------
Sourced, and reproduced faithfully:

  * The legacy RIPS US field list and order (Resolución 3374 de 2000): document
    type, number, entity code, user type, the four separate name fields, age
    plus its unit, sex, department, municipality, zone. Fixture 6 follows it.
  * `dd/mm/aaaa` as the legacy RIPS date pattern, and `AAAA-MM-DD` / `HH:MM`
    from Dentalink's published API.
  * The appointment status words Medifolios documents: agendada, confirmada,
    en espera, atendida, cancelada, no asistió. Fixture 7 uses that vocabulary.
  * Colombian document codes from the national IHCE code system: CC, TI, CE, RC,
    PA, CD, SC, PE, PT, AS, MS, plus NUIP and NIT, which clinics also type.
  * Ten-digit numbering since Resolución CRC 5826 de 2019: mobiles are 10
    digits, fixed lines are 60 + area code + 7 digits.
  * EPS names and codes as the Ministry publishes them (NUEVA EPS EPS001,
    SALUD TOTAL EPS002, SANITAS EPS005, COMPENSAR EPS008, SURA EPS010).

NOT sourced, and deliberately so: no vendor publishes the literal column
headings of its Excel export. The headings below are plausible Spanish, built
from each product's own vocabulary where that vocabulary is public (Dentalink's
API uses `id_paciente`, `nombre_dentista`, `estado_cita`), and invented where it
is not. They are a test of whether the importer copes with headings it has never
seen -- which is the real requirement -- not a claim about any product.

The spreadsheet pathologies are Excel's, not any vendor's: merged cells keep
their value only in the top-left cell, dates become serial numbers, numbers
beyond 15 significant digits lose precision, and a long number displays as
scientific notation. Those are documented behaviours of the format.
"""

from __future__ import annotations

import csv
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

HERE = Path(__file__).parent


def _autosize(worksheet: Worksheet) -> None:
    for column in worksheet.columns:
        width = max((len(str(cell.value)) for cell in column if cell.value), default=8)
        worksheet.column_dimensions[get_column_letter(column[0].column)].width = min(width + 2, 42)


# --------------------------------------------------------------- 6. RIPS US
#: The US (usuarios) record of Resolución 3374 de 2000, in its published order.
#: Names arrive already split into four fields, which is the shape that costs a
#: reviewer nothing -- the measurement behind the export-format question.
_RIPS_USERS = (
    # tipo, numero, eps, tipo_usuario, apellido1, apellido2, nombre1, nombre2,
    # edad, unidad, sexo, departamento, municipio, zona
    (
        "CC",
        "1045678901",
        "EPS001",
        "1",
        "Pérez",
        "Gómez",
        "Carlos",
        "Andrés",
        "38",
        "1",
        "M",
        "11",
        "001",
        "U",
    ),
    (
        "CC",
        "1023456789",
        "EPS002",
        "1",
        "López",
        "Torres",
        "María",
        "Fernanda",
        "45",
        "1",
        "F",
        "05",
        "001",
        "U",
    ),
    (
        "TI",
        "1102345678",
        "EPS005",
        "2",
        "Ortiz",
        "Mejía",
        "Luisa",
        "Fernanda",
        "16",
        "1",
        "F",
        "76",
        "001",
        "U",
    ),
    (
        "CC",
        "71234567",
        "EPS008",
        "1",
        "Castro",
        "Vargas",
        "Luis",
        "Ernesto",
        "52",
        "1",
        "M",
        "08",
        "001",
        "U",
    ),
    (
        "RC",
        "1140567890",
        "EPS010",
        "2",
        "Suárez",
        "Vélez",
        "Andrés",
        "Felipe",
        "7",
        "2",
        "M",
        "11",
        "001",
        "R",
    ),
    (
        "CE",
        "E0456789",
        "EPS001",
        "1",
        "Dubois",
        "Martín",
        "Jean",
        "Pierre",
        "41",
        "1",
        "M",
        "11",
        "001",
        "U",
    ),
    (
        "CC",
        "43567890",
        "EPS002",
        "1",
        "Muñoz",
        "Restrepo",
        "Diana",
        "Carolina",
        "34",
        "1",
        "F",
        "05",
        "001",
        "U",
    ),
    (
        "PT",
        "7845120",
        "EPS005",
        "3",
        "Rodríguez",
        "Blanco",
        "Yoselin",
        "",
        "29",
        "1",
        "F",
        "11",
        "001",
        "U",
    ),
)


def rips_us() -> Path:
    """The national format, field for field. The control for this set.

    No headings at all -- the legacy RIPS archivo is positional, which is
    exactly the case `csv.no_header_row` exists for. A reviewer confirms that
    the first line is data, and every row survives.
    """
    path = HERE / "6_rips_us.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=",")
        writer.writerows(_RIPS_USERS)
    return path


# ----------------------------------------------------------- 7. agenda export
#: Medifolios publishes exactly these six status words. `en espera` is the one
#: the importer does not yet recognise, which is the point of including it.
_STATUSES = ("atendida", "no asistió", "cancelada", "confirmada", "agendada", "en espera")

_AGENDA = (
    (
        "CIT-00412",
        "CC",
        "1045678901",
        "Carlos Andrés Pérez Gómez",
        "MED-01",
        "Dr. Ricardo Soto",
        "Medicina General",
        "2026-11-03",
        "08:00",
        "atendida",
        "+57 310 123 4567",
        "primera vez",
    ),
    (
        "CIT-00413",
        "CC",
        "1023456789",
        "María Fernanda López Torres",
        "MED-01",
        "Dr. Ricardo Soto",
        "Medicina General",
        "2026-11-03",
        "08:30",
        "no asistió",
        "3209876543",
        "control",
    ),
    (
        "CIT-00414",
        "TI",
        "1102345678",
        "Luisa Fernanda Ortiz Mejía",
        "MED-02",
        "Dra. Ana Lucía Rojas",
        "Pediatría",
        "2026-11-03",
        "09:00",
        "atendida",
        "(601) 234 5678",
        "control",
    ),
    (
        "CIT-00415",
        "CC",
        "71234567",
        "Luis Ernesto Castro Vargas",
        "MED-02",
        "Dra. Ana Lucía Rojas",
        "Pediatría",
        "2026-11-04",
        "10:15",
        "cancelada",
        "3105554433",
        "primera vez",
    ),
    (
        "CIT-00416",
        "CC",
        "43567890",
        "Diana Carolina Muñoz Restrepo",
        "MED-03",
        "Dr. Jorge Iván Mejía",
        "Odontología",
        "2026-11-04",
        "11:00",
        "confirmada",
        "3112223344",
        "control",
    ),
    (
        "CIT-00417",
        "RC",
        "1140567890",
        "Andrés Felipe Suárez Vélez",
        "MED-02",
        "Dra. Ana Lucía Rojas",
        "Pediatría",
        "2026-11-05",
        "07:45",
        "agendada",
        "3156667788",
        "primera vez",
    ),
    (
        "CIT-00418",
        "CE",
        "E0456789",
        "Jean Pierre Dubois Martín",
        "MED-01",
        "Dr. Ricardo Soto",
        "Medicina General",
        "2026-11-05",
        "14:30",
        "en espera",
        "3124445566",
        "control",
    ),
    (
        "CIT-00419",
        "CC",
        "1054321987",
        "Sandra Milena Zuluaga Franco",
        "MED-03",
        "Dr. Jorge Iván Mejía",
        "Odontología",
        "2026-11-06",
        "15:00",
        "atendida",
        "601 2345678 ext 104",
        "control",
    ),
)


def agenda_citas() -> Path:
    """One row per appointment, with the patient repeated on every row.

    This is the commonest shape a scheduling system produces, and it exercises
    the reference check: every appointment names a doctor by code, and those
    doctors are defined on the second sheet rather than in the appointment rows.

    The quirks here are ordinary rather than hostile: a heading row styled bold
    and frozen, an appointment whose phone carries an extension, and one status
    word the dictionary does not yet know.
    """
    path = HERE / "7_agenda_citas.xlsx"
    workbook = Workbook()

    citas = workbook.active
    citas.title = "Citas"
    headings = (
        "No. Cita",
        "Tipo Doc",
        "Documento",
        "Nombre del paciente",
        "Cod. Profesional",
        "Profesional",
        "Especialidad",
        "Fecha de la cita",
        "Hora",
        "Estado",
        "Teléfono de contacto",
        "Tipo de consulta",
    )
    citas.append(headings)
    for cell in citas[1]:
        cell.font = Font(bold=True)
    citas.freeze_panes = "A2"
    for row in _AGENDA:
        citas.append(row)
    _autosize(citas)

    profesionales = workbook.create_sheet("Profesionales")
    profesionales.append(("Cod. Profesional", "Nombre", "Especialidad", "Consultorio"))
    for cell in profesionales[1]:
        cell.font = Font(bold=True)
    for row in (
        ("MED-01", "Dr. Ricardo Soto", "Medicina General", "201"),
        ("MED-02", "Dra. Ana Lucía Rojas", "Pediatría", "202"),
        ("MED-03", "Dr. Jorge Iván Mejía", "Odontología", "105"),
    ):
        profesionales.append(row)
    _autosize(profesionales)

    workbook.save(path)
    return path


# ------------------------------------------------------- 8. dental-suite style
def odontologia() -> Path:
    """Three sheets, in a dental suite's vocabulary.

    Dentalink's published API names its fields `id_paciente`, `nombre`,
    `apellidos`, `celular`, `nombre_dentista`, `estado_cita`, and gives dates as
    AAAA-MM-DD with times as HH:MM. The headings here are that vocabulary
    written the way a Spanish-language export would present it.

    Two things make it interesting. Names arrive in two columns, which is the
    shape that imports with nothing to confirm. And the agenda sheet references
    both a patient and a dentist, so the whole workbook has to resolve together.
    """
    path = HERE / "8_odontologia.xlsx"
    workbook = Workbook()

    pacientes = workbook.active
    pacientes.title = "Pacientes"
    pacientes.append(
        (
            "ID Paciente",
            "Tipo Identificación",
            "Identificación",
            "Nombres",
            "Apellidos",
            "Fecha Nacimiento",
            "Celular",
            "Teléfono",
            "Correo",
            "Previsión",
        )
    )
    for cell in pacientes[1]:
        cell.font = Font(bold=True)
    for row in (
        (
            "P-1001",
            "CC",
            "1045678901",
            "Carlos Andrés",
            "Pérez Gómez",
            "1988-03-15",
            "3101234567",
            "6012345678",
            "carlos.perez@example.invalid",
            "NUEVA EPS",
        ),
        (
            "P-1002",
            "CC",
            "1023456789",
            "María Fernanda",
            "López Torres",
            "1981-07-22",
            "3209876543",
            "",
            "m.lopez@example.invalid",
            "SALUD TOTAL",
        ),
        (
            "P-1003",
            "TI",
            "1102345678",
            "Luisa Fernanda",
            "Ortiz Mejía",
            "2010-01-30",
            "3156667788",
            "",
            "",
            "EPS SANITAS",
        ),
        (
            "P-1004",
            "CC",
            "71234567",
            "Luis Ernesto",
            "Castro Vargas",
            "1974-11-08",
            "3105554433",
            "6044567890",
            "l.castro@example.invalid",
            "COMPENSAR",
        ),
        (
            "P-1005",
            "CC",
            "43567890",
            "Diana Carolina",
            "Muñoz Restrepo",
            "1992-05-19",
            "3112223344",
            "",
            "d.munoz@example.invalid",
            "EPS SURA",
        ),
        (
            "P-1006",
            "CE",
            "E0456789",
            "Jean Pierre",
            "Dubois Martín",
            "1985-09-02",
            "3124445566",
            "",
            "",
            "NUEVA EPS",
        ),
    ):
        pacientes.append(row)
    _autosize(pacientes)

    dentistas = workbook.create_sheet("Dentistas")
    dentistas.append(("ID Dentista", "Nombre Dentista", "Especialidad", "Box"))
    for cell in dentistas[1]:
        cell.font = Font(bold=True)
    for row in (
        ("D-01", "Dr. Jorge Iván Mejía", "Odontología General", "105"),
        ("D-02", "Dra. Paula Andrea Gil", "Ortodoncia", "106"),
    ):
        dentistas.append(row)
    _autosize(dentistas)

    agenda = workbook.create_sheet("Agenda")
    agenda.append(
        (
            "ID Cita",
            "Identificación",
            "ID Dentista",
            "Fecha",
            "Hora Inicio",
            "Hora Fin",
            "Estado Cita",
        )
    )
    for cell in agenda[1]:
        cell.font = Font(bold=True)
    for row in (
        ("A-5001", "1045678901", "D-01", "2026-11-10", "09:00", "09:30", "confirmada"),
        ("A-5002", "1023456789", "D-01", "2026-11-10", "09:30", "10:00", "atendida"),
        ("A-5003", "1102345678", "D-02", "2026-11-11", "15:00", "16:00", "agendada"),
        ("A-5004", "71234567", "D-02", "2026-11-12", "08:00", "08:45", "cancelada"),
    ):
        agenda.append(row)
    _autosize(agenda)

    workbook.save(path)
    return path


# ------------------------------------------------------- 9. front-desk sheet
def hoja_recepcion() -> Path:
    """A spreadsheet kept by hand, with everything that implies.

    Not hostile -- 2_receptionist.xlsx already covers deliberate sabotage. This
    is the ordinary decay of a file three people have edited for two years:

      * a merged title across the top, so row 1 is not the headings
      * a blank spacer row between the title and the table
      * a merged "DATOS DEL PACIENTE" band above the real headings
      * names in one column, three words wide, which cannot be split by program
      * dates typed four different ways, including a Spanish month abbreviation
        and a two-digit year
      * a cédula Excel has stored as a number, losing its leading zero
      * a phone with an extension in the same cell, and one with two numbers
      * `cancelado` in the masculine, which the dictionary does not know
      * a blank row in the middle where somebody deleted a patient
      * a totals line at the bottom that is not a patient

    Every one of these is either a question the reviewer answers or a refusal
    naming the row. None of them may import silently as something else.
    """
    path = HERE / "9_hoja_recepcion.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Control Pacientes"

    sheet["A1"] = "CLÍNICA SAN RAFAEL — CONTROL DE PACIENTES 2026"
    sheet.merge_cells("A1:H1")
    sheet["A1"].font = Font(bold=True, size=14)
    sheet["A1"].alignment = Alignment(horizontal="center")
    sheet.append(())  # a spacer row the file has always had

    sheet["A3"] = "DATOS DEL PACIENTE"
    sheet.merge_cells("A3:D3")
    sheet["E3"] = "CITA"
    sheet.merge_cells("E3:H3")
    for reference in ("A3", "E3"):
        sheet[reference].font = Font(bold=True)
        sheet[reference].alignment = Alignment(horizontal="center")

    sheet.append(
        (
            "T.D.",
            "CÉDULA",
            "NOMBRE COMPLETO",
            "CELULAR",
            "FEC NAC",
            "FECHA CITA",
            "MÉDICO",
            "ESTADO",
        )
    )
    for cell in sheet[4]:
        cell.font = Font(bold=True)

    rows: tuple[tuple[object, ...], ...] = (
        (
            "CC",
            "1045678901",
            "Carlos Andrés Pérez Gómez",
            "3101234567",
            "15/03/1988",
            "03/11/2026",
            "Dr. Soto",
            "atendida",
        ),
        (
            "CC",
            "1023456789",
            "María Fernanda López Torres",
            "320 987 6543",
            "22-jul-1981",
            "03/11/2026",
            "Dr. Soto",
            "no asistió",
        ),
        # Excel stored this cédula as a number, so its leading zero is gone.
        (
            "CC",
            98765432,
            "Luis Ernesto Castro Vargas",
            "3105554433",
            "08/11/74",
            "04/11/2026",
            "Dra. Rojas",
            "cancelado",
        ),
        (
            "TI",
            "1102345678",
            "Luisa Fernanda Ortiz Mejía",
            "601 2345678 ext 104",
            "30/01/2010",
            "04/11/2026",
            "Dra. Rojas",
            "atendida",
        ),
        (),  # somebody deleted a patient and left the row
        (
            "CC",
            "43567890",
            "Diana Carolina Muñoz Restrepo",
            "311 222 3344 / 310 999 8877",
            "19/05/1992",
            "05/11/2026",
            "Dr. Mejía",
            "confirmada",
        ),
        # Three words: one given name and two surnames, or two and one. Undecidable.
        (
            "CC",
            "1067891234",
            "Juan Carlos Pérez",
            "3156667788",
            "02/09/1985",
            "05/11/2026",
            "Dr. Mejía",
            "agendada",
        ),
        (
            "CE",
            "E0456789",
            "Jean Pierre Dubois Martín",
            "3124445566",
            "1985-09-02",
            "06/11/2026",
            "Dr. Soto",
            "en espera",
        ),
        (),
        ("TOTAL", 7, "", "", "", "", "", ""),
    )
    for row in rows:
        sheet.append(row)
    _autosize(sheet)

    workbook.save(path)
    return path


def main() -> None:
    produced = [rips_us(), agenda_citas(), odontologia(), hoja_recepcion()]
    for path in produced:
        print(f"  {path.name:28} {path.stat().st_size:>9,} bytes")
    print(f"\n{len(produced)} platform-style exports written to {HERE}")
    print(
        "\nEach is a different system's way of thinking:\n"
        "  6_rips_us.csv          positional, no headings at all\n"
        "  7_agenda_citas.xlsx    one row per appointment, doctors on a second sheet\n"
        "  8_odontologia.xlsx     three sheets that must resolve together\n"
        "  9_hoja_recepcion.xlsx  a hand-kept sheet: merged bands, mixed dates, a totals line"
    )


if __name__ == "__main__":
    main()
