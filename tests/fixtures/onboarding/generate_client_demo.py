"""Three exports shaped like the client's own, for the walkthrough video.

    python -m tests.fixtures.onboarding.generate_client_demo

Every value here is invented. No file in this directory contains real patient
data, and none may: see CLAUDE.md rule 7. The cédulas are sequential, the
mobiles are textbook 300/310/320 numbers, and the names are common enough to
belong to nobody in particular.

`generate.py` escalates from a tidy file to one built to break the parser, and
`generate_platform_exports.py` covers the ordinary middle. These three are
narrower and exist for one purpose: each isolates **one** of the defects the
client's own export uncovered, so a demonstration can show cause and effect
without four things happening at once.

    A_mobiles_in_a_phone_column.xlsx   a column headed "phone" holding mobiles
    B_blank_optional_columns.xlsx      optional columns most clinics leave empty
    C_english_headings.xlsx            an English export, dates and times included

Each file is small enough to read on screen in full, and each is *correct* --
these are files a clinic would legitimately hand over. The importer's job is to
notice what is ambiguous about them, not to reject them.

WHY THESE THREE
---------------
All three were silent failures: the import succeeded and the data was wrong, or
rows vanished, with nothing on screen to say so.

  A. "phone" and "telefono" were landline-only aliases, so a column of mobiles
     mapped to `phone_fixed`. The phone validator checks a number against
     Colombia but not against the field it landed in, so every row converted as
     *valid* -- and no appointment reminder would ever have been sent.

  B. Every normalizer reports an empty cell as `review`, which is right for a
     required field and wrong for an optional one. A blank emergency contact --
     which most clinics leave blank for most patients -- held the entire row
     back. Six of ten patients in the client's export were skipped, five of them
     for this alone.

  C. The bare Spanish `fecha` and `hora` were aliases; the bare English `date`
     and `time` were not. An appointments sheet exported in English dropped both
     and then reported them missing.

WHAT IS SOURCED AND WHAT IS NOT
-------------------------------
Sourced: ten-digit numbering since Resolución CRC 5826 de 2019 (mobiles are 10
digits beginning 3, fixed lines are 60 + area code + 7 digits); Colombian
document codes from the IHCE code system; EPS names as the Ministry publishes
them. The unassigned prefixes used below (306, 309) are unassigned in
libphonenumber's current CO metadata, which is what `normalizers.phone` checks
against.

Not sourced: the column headings. No vendor publishes the literal headings of
its Excel export. These follow the client's own file, which uses English
snake_case, because matching his shape is the point.
"""

from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook

HERE = Path(__file__).resolve().parent


def _sheet(wb: Workbook, title: str, headings: list[str], rows: list[list[str]]) -> None:
    """One sheet of plain strings.

    Everything is written as text, including dates and document numbers. Excel
    would otherwise store a cédula as a float and a date as a serial, which is
    the corruption `reader.py` exists to survive -- but these files are meant to
    demonstrate the importer's ordinary path, not its recovery path.
    """
    ws = wb.create_sheet(title) if wb.sheetnames != ["Sheet"] else wb.active
    ws.title = title
    ws.append(headings)
    for row in rows:
        ws.append(row)


# ---------------------------------------------------------------- A: mobiles
def _mobiles_in_a_phone_column() -> None:
    """A column headed `phone` that holds mobile numbers.

    The heading alone cannot settle which kind it is, and the two are not
    interchangeable: reminders go to the mobile. Row 6 carries an unassigned
    number so the file also shows a refusal that is *not* ambiguity -- that one
    is flagged and never approximated, because the nearest valid number belongs
    to somebody else.
    """
    wb = Workbook()
    _sheet(
        wb,
        "Patients",
        ["id_type", "document_number", "full_name", "phone", "email", "eps"],
        [
            ["CC", "1045678901", "Carlos Pérez", "3001234567", "carlos@example.co", "Sura"],
            ["CC", "1023456789", "María López", "3119876543", "maria@example.co", "Nueva EPS"],
            ["CC", "71234567", "Luis Castro", "3204567890", "luis@example.co", "Sanitas"],
            ["CC", "1098765432", "Ana Gómez", "3157778899", "ana@example.co", "Compensar"],
            # Unassigned prefix: refused, never corrected.
            ["CC", "1076543210", "Jorge Ramírez", "3067891234", "jorge@example.co", "Sura"],
        ],
    )
    wb.save(HERE / "A_mobiles_in_a_phone_column.xlsx")


# ------------------------------------------------------- B: blank optionals
def _blank_optional_columns() -> None:
    """Optional columns that most clinics leave blank for most patients.

    Only one patient here has an emergency contact, which is realistic. None of
    these blanks is a question for anybody: the clinic did not record the value.
    Row 5's second contact number is unassigned, so the file still shows that an
    optional field carrying a *wrong* value is refused -- the exemption is for
    absence, not for skipping the check.
    """
    wb = Workbook()
    _sheet(
        wb,
        "Patients",
        [
            "id_type",
            "document_number",
            "full_name",
            # "celular", not "phone": this file is about blank optional columns,
            # and a bare "phone" heading would raise the mobile-or-landline
            # question from file A on top of it. One file, one thing to see.
            "celular",
            "email",
            "secondary_contact_name",
            "secondary_contact_phone",
        ],
        [
            ["CC", "1011111111", "Diana Torres", "3101111111", "", "", ""],
            ["CC", "1022222222", "Felipe Moreno", "3102222222", "", "", ""],
            ["CC", "1033333333", "Sandra Ruiz", "3103333333", "sandra@example.co", "", ""],
            [
                "CC",
                "1044444444",
                "Andrés Vargas",
                "3104444444",
                "",
                "Marta Vargas",
                "3205556677",
            ],
            # A value that is present and wrong: still refused.
            ["CC", "1055555555", "Paula Jiménez", "3105555555", "", "Luis Jiménez", "3098765432"],
        ],
    )
    wb.save(HERE / "B_blank_optional_columns.xlsx")


# ------------------------------------------------------- C: English headings
def _english_headings() -> None:
    """An export written in English, with bare `date` and `time` columns.

    Three sheets, because the appointments sheet is where the defect showed: its
    date and time were dropped and then reported missing. The doctors and
    patients sheets are here so every reference an appointment makes resolves
    *inside the file*. Without them the import blocks on four missing patients,
    which is the referential-integrity check working correctly but is not what
    this file is meant to demonstrate -- and a blocked screen would make the
    headings fix impossible to see.
    """
    wb = Workbook()
    _sheet(
        wb,
        "Patients",
        ["id_type", "document_number", "full_name", "celular", "eps"],
        [
            ["CC", "1045678901", "Carlos Pérez", "3001234567", "Sura"],
            ["CC", "1023456789", "María López", "3119876543", "Nueva EPS"],
            ["CC", "71234567", "Luis Castro", "3204567890", "Sanitas"],
            ["CC", "1098765432", "Ana Gómez", "3157778899", "Compensar"],
        ],
    )
    _sheet(
        wb,
        "Doctors",
        ["doctor_id", "doctor_name", "specialty", "office_number"],
        [
            ["M01", "Patricia Gómez", "Ortopedia", "12"],
            ["M02", "Juan Ramírez", "Ortopedia", "14"],
            ["M03", "Carlos Vega", "Cirugía General", "08"],
        ],
    )
    _sheet(
        wb,
        "Appointments",
        ["appointment_id", "document_number", "doctor_id", "date", "time", "status"],
        [
            ["C0001", "1045678901", "M01", "2026-11-03", "08:00", "Confirmada"],
            ["C0002", "1023456789", "M03", "2026-11-03", "09:30", "Pendiente"],
            ["C0003", "71234567", "M02", "2026-11-04", "14:00", "Atendida"],
            ["C0004", "1098765432", "M01", "2026-11-04", "15:30", "Cancelada"],
        ],
    )
    wb.save(HERE / "C_english_headings.xlsx")


def main() -> None:
    _mobiles_in_a_phone_column()
    _blank_optional_columns()
    _english_headings()
    for name in (
        "A_mobiles_in_a_phone_column.xlsx",
        "B_blank_optional_columns.xlsx",
        "C_english_headings.xlsx",
    ):
        print(f"wrote {HERE / name}")


if __name__ == "__main__":
    main()
