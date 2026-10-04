"""Schema validation, loader formats and Verse-as-text preservation."""

from __future__ import annotations

import pandas as pd
import pytest

from shloka_translator.io.loader import load_file
from shloka_translator.io.validator import validate_schema, validate_table
from tests.helpers import FIXTURES, REFERENCE

COLS = ["ID", "Source", "Verse", "Shloka"]


def _validate(name: str) -> list:
    """Load and validate a fixture.

    Args:
        name: Fixture file name.

    Returns:
        List of ``(frame, report)``.
    """
    return [validate_table(t, COLS) for t in load_file(FIXTURES / name)]


def test_valid_formats_accepted() -> None:
    """CSV (comma/semicolon), JSON list, JSON records, JSONL are accepted."""
    for name in ("valid.csv", "valid_semicolon.csv", "valid.json", "valid_records.json", "valid.jsonl"):
        ((frame, report),) = _validate(name)
        assert report.status == "Accepted", (name, report.reason)
        assert list(frame.columns) == COLS
        assert frame["Verse"].tolist() == ["2.47", "1.10", "Passage 2"]


def test_xlsx_sheets_validated_independently() -> None:
    """Good sheet accepted, bad sheet rejected, empty sheet ignored."""
    results = _validate("valid.xlsx")
    by_sheet = {r.sheet: (f, r) for f, r in results}
    assert set(by_sheet) == {"Sheet1", "BadSheet"}
    frame, rep = by_sheet["Sheet1"]
    assert rep.status == "Accepted"
    assert frame["Verse"].tolist() == ["99", "1.10", "1.43"]
    assert frame["ID"].tolist() == ["1", "2", "3"]
    assert by_sheet["BadSheet"][1].status == "Rejected"
    assert "missing=['Shloka']" in by_sheet["BadSheet"][1].reason


@pytest.mark.parametrize(
    "name,needle",
    [
        ("invalid_missing_column.csv", "missing=['Verse']"),
        ("invalid_extra_column.csv", "extra=['Notes']"),
        ("invalid_renamed.json", "missing=['ID']"),
        ("invalid_shape.json", "Unsupported JSON shape"),
        ("empty.csv", "Empty file"),
    ],
)
def test_invalid_files_rejected(name: str, needle: str) -> None:
    """Every invalid fixture is rejected with a precise reason.

    Args:
        name: Fixture.
        needle: Expected substring of the reason.
    """
    ((frame, report),) = _validate(name)
    assert frame is None
    assert report.status == "Rejected"
    assert needle in report.reason


def test_case_sensitive_and_whitespace() -> None:
    """Whitespace around names is ignored; case is not."""
    ok = pd.DataFrame(columns=[" ID ", "Source", "Verse", "Shloka "])
    assert validate_schema(ok, COLS, source_name="x").is_valid
    bad = pd.DataFrame(columns=["id", "Source", "Verse", "Shloka"])
    res = validate_schema(bad, COLS, source_name="x")
    assert not res.is_valid and res.missing == ["ID"] and res.extra == ["id"]


def test_duplicate_column_rejected() -> None:
    """Duplicated columns reject the table."""
    df = pd.DataFrame([[1, 2, 3, 4, 5]], columns=["ID", "Source", "Verse", "Shloka", "Shloka"])
    assert "duplicated=['Shloka']" in validate_schema(df, COLS, source_name="x").reason


def test_type_error() -> None:
    """Non-DataFrame input raises TypeError."""
    with pytest.raises(TypeError):
        validate_schema([1, 2], COLS, source_name="x")  # type: ignore[arg-type]


def test_empty_id_rejects_table(tmp_path) -> None:
    """A row without ID rejects the whole file.

    Args:
        tmp_path: Temp dir.
    """
    p = tmp_path / "noid.csv"
    p.write_text("ID,Source,Verse,Shloka\n,a,1,राम\n", encoding="utf-8")
    ((frame, report),) = [validate_table(t, COLS) for t in load_file(p)]
    assert frame is None and "Empty ID" in report.reason


def test_reference_workbook_schema() -> None:
    """The reference file has 215 rows, 4 columns, Verse kept as text."""
    ((frame, report),) = [validate_table(t, COLS) for t in load_file(REFERENCE)]
    assert report.status == "Accepted" and len(frame) == 215
    verses = set(frame["Verse"])
    assert {"99", "18.66", "5.12", "2.20", "Passage 2"} <= verses
    assert not any(v.endswith(".0") for v in verses)
    assert "Verse cell(s) stored as numbers" in report.reason
