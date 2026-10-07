"""Build the four name-export shapes a Colombian clinic can plausibly send.

These are for trying the importer by hand, not for the test suite: the suite
asserts these shapes directly in `test_onboarding_api.py`. They exist because
the name column is the single biggest driver of how much review an import
costs, and seeing the four files side by side makes that concrete.

The same eight patients appear in every file, so the only thing that differs is
how the name was exported.

    python -m tests.fixtures.onboarding.generate_name_shapes

Written to this directory, which is not in version control.
"""

from __future__ import annotations

import csv
from pathlib import Path

from openpyxl import Workbook

HERE = Path(__file__).resolve().parent

#: (document, given names, surnames, birth date, mobile). Invented, not sampled:
#: every document number is sequential and every mobile uses a prefix
#: libphonenumber accepts, so nothing here resembles a real person.
PEOPLE: tuple[tuple[str, str, str, str, str], ...] = (
    ("1000000001", "Carlos Andres", "Perez Gomez", "1987-04-12", "3101234567"),
    ("1000000002", "Maria Jose", "Rojas Sanin", "1992-11-03", "3151112233"),
    # One given name and two surnames: the shape nothing can split.
    ("1000000003", "Luis", "Rojas Sanin", "1975-02-28", "3001234567"),
    ("1000000004", "Ana", "Munoz Castro", "1990-07-19", "3024445566"),
    # A compound given name the dictionary knows, which does resolve.
    ("1000000005", "Juan Carlos", "Diaz", "1968-09-30", "3112223344"),
    # A particle surname.
    ("1000000006", "Sofia", "de la Cruz Moreno", "2001-01-15", "3186667788"),
    # Two given names, one surname.
    ("1000000007", "Diego Alejandro", "Ramirez", "1983-06-07", "3209998877"),
    ("1000000008", "Valentina", "Lopez Ruiz", "1996-12-21", "3134445566"),
)


def _one_column() -> Path:
    """Everything in one cell. The worst case, and what hand-kept files do."""
    path = HERE / "name_1_one_column.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(
            ["TIPO DOC", "IDENTIFICACION", "NOMBRE COMPLETO", "FECHA NACIMIENTO", "CELULAR"]
        )
        for document, given, family, born, phone in PEOPLE:
            writer.writerow(["CC", document, f"{given} {family}", born, phone])
    return path


def _two_columns() -> Path:
    """Given names and surnames apart. Imports with nothing to confirm."""
    path = HERE / "name_2_two_columns.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(
            ["TIPO DOC", "IDENTIFICACION", "NOMBRES", "APELLIDOS", "FECHA NACIMIENTO", "CELULAR"]
        )
        for document, given, family, born, phone in PEOPLE:
            writer.writerow(["CC", document, given, family, born, phone])
    return path


def _four_columns() -> Path:
    """The four parts separately, as the old RIPS archivo US carried them."""
    path = HERE / "name_3_four_columns.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = "Pacientes"
    sheet.append(
        [
            "TIPO DOC",
            "IDENTIFICACION",
            "PRIMER NOMBRE",
            "SEGUNDO NOMBRE",
            "PRIMER APELLIDO",
            "SEGUNDO APELLIDO",
            "FECHA NACIMIENTO",
            "CELULAR",
        ]
    )
    for document, given, family, born, phone in PEOPLE:
        first, _, second = given.partition(" ")
        # A particle belongs to the surname it modifies, so it is split from the
        # right: "de la Cruz Moreno" is "de la Cruz" then "Moreno".
        parts = family.rsplit(" ", 1)
        paternal, maternal = (parts[0], parts[1]) if len(parts) == 2 else (family, "")
        sheet.append(["CC", document, first, second, paternal, maternal, born, phone])
    workbook.save(path)
    return path


def _rips_field_names() -> Path:
    """RIPS's own camelCase field names, which a generated export may use."""
    path = HERE / "name_4_rips_fields.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(
            [
                "tipoDocumentoIdentificacion",
                "numDocumentoIdentificacion",
                "primerNombre",
                "segundoNombre",
                "primerApellido",
                "segundoApellido",
                "fechaNacimiento",
                "celular",
            ]
        )
        for document, given, family, born, phone in PEOPLE:
            first, _, second = given.partition(" ")
            parts = family.rsplit(" ", 1)
            paternal, maternal = (parts[0], parts[1]) if len(parts) == 2 else (family, "")
            writer.writerow(["CC", document, first, second, paternal, maternal, born, phone])
    return path


def main() -> list[Path]:
    produced = [_one_column(), _two_columns(), _four_columns(), _rips_field_names()]
    for path in produced:
        print(f"wrote {path.name}")
    print(
        "\nSame eight patients in each. Expect the one-column file to send rows to "
        "review\nand the other three to import with nothing to confirm."
    )
    return produced


if __name__ == "__main__":
    main()
