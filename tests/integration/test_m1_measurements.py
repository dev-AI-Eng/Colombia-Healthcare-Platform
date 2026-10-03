"""Measured end-to-end import results, for the milestone report.

Not a guard test: it asserts only that every file reaches the verdict it should,
and prints the numbers quoted in the M1 document so the client can re-derive
them rather than take them on trust.

    pytest tests/integration/test_m1_measurements.py -s -q
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.core.tenancy import ClinicScope, apply_clinic_scope
from tests.integration.factories import create_graph, scope_client

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "onboarding"

#: The clinic exports, in the order the report discusses them.
EXPORTS = ("1_clean_ips.xlsx", "3_excel_csv_es.csv", "2_receptionist.xlsx", "4_corrupted.xlsx")

#: Refused at upload: the file itself is the attack.
REFUSED = ("5_zip_bomb.xlsx", "5_not_really_csv.csv", "5_xxe.xlsx")

#: Read rather than refused, which is the harder case: the attack is in the
#: values, and the defence is that the reader neither evaluates nor trusts them.
#: The formula is kept as text so validation can reject it; the false dimension
#: is overridden so the sheet's real 5,000 rows are read, not the 1 it declares.
DISARMED = ("5_formula_injection.xlsx", "5_lying_dimension.xlsx")

#: The same eight patients, exported four ways. The argument for the client.
NAME_SHAPES = (
    "name_1_one_column.csv",
    "name_2_two_columns.csv",
    "name_3_four_columns.xlsx",
    "name_4_rips_fields.csv",
)


@pytest.fixture(scope="module", autouse=True)
def fixtures_exist() -> None:
    """Build whichever generated files are missing.

    The fixtures are not committed -- one is a zip bomb -- so a clean checkout
    has none of them. This file is the only one that reads the name shapes and
    the platform exports, and without this guard `pytest -q` fails on a fresh
    clone with a FileNotFoundError rather than a test result.
    """
    if not (FIXTURES / "1_clean_ips.xlsx").exists():
        subprocess.run(
            [sys.executable, "-m", "tests.fixtures.onboarding.generate"],
            check=True,
            capture_output=True,
        )
    if not (FIXTURES / "name_1_one_column.csv").exists():
        subprocess.run(
            [sys.executable, "-m", "tests.fixtures.onboarding.generate_name_shapes"],
            check=True,
            capture_output=True,
        )
    if not (FIXTURES / "6_rips_us.csv").exists():
        subprocess.run(
            [sys.executable, "-m", "tests.fixtures.onboarding.generate_platform_exports"],
            check=True,
            capture_output=True,
        )


@pytest.fixture
async def scoped(session, client: TestClient) -> Iterator[TestClient]:  # type: ignore[no-untyped-def]
    """A client already scoped to a real clinic, as every route requires."""
    graph = await create_graph(session)
    await session.commit()
    await apply_clinic_scope(session, ClinicScope(clinic_id=graph.clinic_id))
    yield scope_client(client, graph)


def _upload(client: TestClient, name: str) -> tuple[int, dict]:  # type: ignore[type-arg]
    with (FIXTURES / name).open("rb") as handle:
        response = client.post("/onboarding/uploads", files={"file": (name, handle)})
    return response.status_code, (response.json() if response.content else {})


def _validate(client: TestClient, name: str) -> dict[str, int]:
    """Upload, accept the proposed mapping, validate the whole file."""
    status, body = _upload(client, name)
    assert status == 201, f"{name}: HTTP {status}"
    session_id = body["session_id"]
    client.put(f"/onboarding/uploads/{session_id}/mapping", json={"columns": {}})
    report = client.post(f"/onboarding/uploads/{session_id}/validate")
    assert report.status_code == 200, report.text
    sheets = report.json()["sheets"]
    return {
        "sheets": len(sheets),
        "rows": sum(s["total_rows"] for s in sheets),
        "review": sum(s["review_rows"] for s in sheets),
        "invalid": sum(s["invalid_rows"] for s in sheets),
    }


async def test_measure_hostile_files(scoped: TestClient) -> None:
    print("\n--- refused at upload ---")
    for name in REFUSED:
        status, _ = _upload(scoped, name)
        print(f"  {name:28s} HTTP {status}  refused")
        assert status == 422, f"{name} was not refused (HTTP {status})"

    print("--- read, without being trusted ---")
    for name in DISARMED:
        status, body = _upload(scoped, name)
        assert status == 201, f"{name}: HTTP {status}"
        rows = sum(sheet["total_rows"] for sheet in body["sheets"])
        print(f"  {name:28s} HTTP {status}  rows read={rows}")


async def test_measure_the_name_export_shapes(scoped: TestClient) -> None:
    """One combined name column sends rows to review; separate columns do not."""
    print("\n--- name shapes (the same 8 patients) ---")
    for name in NAME_SHAPES:
        result = _validate(scoped, name)
        print(
            f"  {name:26s} rows={result['rows']:3d} "
            f"review={result['review']:3d} invalid={result['invalid']:3d}"
        )
        assert result["rows"] == 8, f"{name} lost rows"
        # Not just the count: the whole point of this measurement is that one
        # combined name column costs a decision and separate columns do not. A
        # version that guessed the split would still report 8 rows.
        expected_review = 5 if name == "name_1_one_column.csv" else 0
        assert result["review"] == expected_review, (
            f"{name}: expected {expected_review} rows in review, got {result['review']}"
        )


#: What each export measures to, as quoted in the milestone report's table.
#: Asserted rather than printed: these are the numbers the client is given, so a
#: change in any of them is either a regression or a figure to correct.
EXPECTED = {
    "1_clean_ips.xlsx": {"sheets": 3, "rows": 20, "review": 4, "invalid": 0},
    "3_excel_csv_es.csv": {"sheets": 1, "rows": 5, "review": 3, "invalid": 0},
    # 32, not 16: AGENDA is one row per visit, so each line yields a patient
    # and an appointment. Before multi-entity mapping its four patient columns
    # were silently dropped.
    "2_receptionist.xlsx": {"sheets": 2, "rows": 32, "review": 13, "invalid": 2},
    # 12, not 6: this sheet carries FECHA_CITA and HORA beside the patient's
    # own columns, so each line is a patient and an appointment.
    "4_corrupted.xlsx": {"sheets": 1, "rows": 12, "review": 6, "invalid": 0},
}


async def test_measure_the_clinic_exports(scoped: TestClient) -> None:
    print("\n--- clinic exports ---")
    for name in EXPORTS:
        result = _validate(scoped, name)
        print(
            f"  {name:24s} sheets={result['sheets']} rows={result['rows']:3d} "
            f"review={result['review']:3d} invalid={result['invalid']:3d}"
        )
        assert result == EXPECTED[name], name
