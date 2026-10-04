"""Merge accepted tables, de-duplicate and attach provenance."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

from shloka_translator.io.loader import LoadedTable
from shloka_translator.utils.text import normalize_text

logger = logging.getLogger(__name__)

PROVENANCE_COLUMNS = ["_source_file", "_source_sheet", "_row_number"]


@dataclass
class AcceptedTable:
    """A validated table plus where it came from.

    Attributes:
        frame: Clean frame with exactly the expected columns.
        table: The original :class:`LoadedTable` (for provenance and flags).
    """

    frame: pd.DataFrame
    table: LoadedTable


@dataclass
class MergeResult:
    """Merged records plus bookkeeping.

    Attributes:
        records: Frame with expected columns + provenance + ``_remarks``
            (list of strings) + ``_order`` (deterministic position).
        dropped_duplicates: Human-readable notes for dropped exact duplicates.
    """

    records: pd.DataFrame
    dropped_duplicates: list[str] = field(default_factory=list)


def merge_tables(accepted: list[AcceptedTable], expected_columns: list[str]) -> MergeResult:
    """Concatenate accepted tables, remove exact duplicates, flag ID clashes.

    * Exact duplicates (same ``ID`` and same normalised ``Shloka``) → first kept.
    * Same ``ID`` with different content → both kept; ``Remarks`` gets
      ``Duplicate ID across inputs (<file>)``.
    * Numeric ``Verse`` cells that collide with another row of the same
      ``Source`` get a ``Possible Verse corruption`` remark.
    * Order: file order, then row order.

    Args:
        accepted: Validated tables in file order.
        expected_columns: Output column order.

    Returns:
        :class:`MergeResult`.

    Example:
        >>> merge_tables([], ["ID", "Source", "Verse", "Shloka"]).records.empty
        True
    """
    columns = list(expected_columns) + PROVENANCE_COLUMNS + ["_remarks", "_order"]
    rows: list[dict] = []
    for acc in accepted:
        numeric_rows = acc.table.numeric_verse_rows
        for pos, rec in enumerate(acc.frame.to_dict(orient="records")):
            row = {c: rec.get(c, "") for c in expected_columns}
            row["_source_file"] = acc.table.file_name
            row["_source_sheet"] = acc.table.sheet
            row["_row_number"] = pos + 2  # spreadsheet row (header is row 1)
            row["_remarks"] = []
            row["_numeric_verse"] = pos in numeric_rows
            rows.append(row)
    if not rows:
        return MergeResult(pd.DataFrame(columns=columns))

    kept: list[dict] = []
    seen_exact: dict[tuple[str, str], dict] = {}
    dropped: list[str] = []
    for row in rows:
        key = (str(row.get("ID", "")).strip(), normalize_text(row.get("Shloka", "")))
        if key in seen_exact:
            first = seen_exact[key]
            note = (
                f"Dropped exact duplicate ID={key[0]} from {row['_source_file']} row {row['_row_number']} "
                f"(kept {first['_source_file']} row {first['_row_number']})"
            )
            dropped.append(note)
            logger.info(note)
            continue
        seen_exact[key] = row
        kept.append(row)

    by_id: dict[str, list[dict]] = defaultdict(list)
    for row in kept:
        by_id[str(row.get("ID", "")).strip()].append(row)
    for rid, group in by_id.items():
        if len(group) > 1:
            for row in group:
                others = sorted({g["_source_file"] for g in group if g is not row})
                row["_remarks"].append(f"Duplicate ID across inputs ({', '.join(others)})")
            logger.warning("ID %s appears %d times with different content", rid, len(group))

    if "Source" in expected_columns and "Verse" in expected_columns:
        by_sv: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for row in kept:
            by_sv[(row["_source_file"], row.get("Source", ""), row.get("Verse", ""))].append(row)
        for group in by_sv.values():
            if len(group) > 1 and any(r.get("_numeric_verse") for r in group):
                for row in group:
                    if row.get("_numeric_verse"):
                        row["_remarks"].append(
                            "Possible Verse corruption: stored as a number in the workbook and duplicates "
                            "another Verse of the same Source (e.g. 1.10 saved as 1.1) — check the source file"
                        )

    for order, row in enumerate(kept):
        row["_order"] = order
        row.pop("_numeric_verse", None)
    frame = pd.DataFrame(kept, columns=columns)
    return MergeResult(frame, dropped)
