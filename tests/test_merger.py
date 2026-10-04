"""Merge, de-duplication, provenance and deterministic order."""

from __future__ import annotations

import pandas as pd

from shloka_translator.io.loader import LoadedTable
from shloka_translator.io.merger import AcceptedTable, merge_tables

COLS = ["ID", "Source", "Verse", "Shloka"]


def _acc(name: str, rows: list[list[str]], numeric: set[int] | None = None) -> AcceptedTable:
    """Build an accepted table.

    Args:
        name: File name.
        rows: Data rows.
        numeric: Rows whose Verse was numeric.

    Returns:
        AcceptedTable.
    """
    frame = pd.DataFrame(rows, columns=COLS)
    return AcceptedTable(frame, LoadedTable(name, name, "", frame, None, numeric or set()))


def test_exact_duplicates_dropped_and_id_clash_flagged() -> None:
    """Same ID + same text is dropped; same ID + different text is kept and flagged."""
    a = _acc("a.csv", [["1", "S", "1", "राम"], ["2", "S", "2", "सीता"]])
    b = _acc("b.csv", [["1", "S", "1", " राम "], ["2", "S", "2", "लक्ष्मण"]])
    res = merge_tables([a, b], COLS)
    df = res.records
    assert len(df) == 3
    assert len(res.dropped_duplicates) == 1
    flagged = df[df["ID"] == "2"]
    assert len(flagged) == 2
    assert all("Duplicate ID across inputs" in r[0] for r in flagged["_remarks"])
    assert df["_order"].tolist() == [0, 1, 2]
    assert df["_source_file"].tolist() == ["a.csv", "a.csv", "b.csv"]
    assert df["_row_number"].tolist() == [2, 3, 3]


def test_numeric_verse_collision_flagged() -> None:
    """A numeric Verse colliding with another of the same Source gets a corruption remark."""
    a = _acc("x.xlsx", [["1", "K", "1.1", "a"], ["2", "K", "1.1", "b"], ["3", "K", "1.2", "c"]], {0, 1, 2})
    df = merge_tables([a], COLS).records
    assert "Possible Verse corruption" in df.iloc[0]["_remarks"][0]
    assert df.iloc[2]["_remarks"] == []


def test_empty() -> None:
    """No tables → empty frame with the expected columns."""
    df = merge_tables([], COLS).records
    assert df.empty and "Shloka" in df.columns
