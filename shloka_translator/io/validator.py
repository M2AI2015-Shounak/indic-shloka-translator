"""Strict schema validation of loaded input tables."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from shloka_translator.io.loader import LoadedTable
from shloka_translator.utils.text import verse_to_text

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Outcome of comparing a table's columns with the expected schema.

    Attributes:
        is_valid: ``True`` if the columns match.
        missing: Expected columns that are absent.
        extra: Columns present but not expected.
        duplicated: Column names that appear more than once.
        reason: Human-readable explanation (empty when valid).
    """

    is_valid: bool
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    duplicated: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class ValidationReportRow:
    """One row of the ``Input_Validation`` sheet.

    Attributes:
        file: File name.
        sheet: Sheet name (XLSX) or ``""``.
        status: ``"Accepted"`` or ``"Rejected"``.
        rows: Number of data rows read.
        reason: Why the table was rejected, or notes for accepted tables.
    """

    file: str
    sheet: str
    status: str
    rows: int
    reason: str


def validate_schema(
    df: pd.DataFrame,
    expected_columns: list[str],
    *,
    source_name: str,
    allow_extra_columns: bool = False,
) -> ValidationResult:
    """Validate that a loaded table exactly matches the reference schema.

    Column names are compared after stripping surrounding whitespace; the
    comparison is case-sensitive. Order is not enforced.

    Args:
        df: Table loaded from one input file/sheet; all cells as ``str``.
        expected_columns: Required column names, e.g. ``["ID", "Source", "Verse", "Shloka"]``.
        source_name: Human-readable origin such as ``"batch1.xlsx::Sheet1"``, used in messages.
        allow_extra_columns: If ``True``, extra columns are tolerated and dropped.

    Returns:
        ValidationResult with ``is_valid``, ``missing``, ``extra`` and ``reason``.

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame(columns=["ID", "Source", "Verse", "Shloka"])
        >>> validate_schema(df, ["ID", "Source", "Verse", "Shloka"], source_name="a.csv").is_valid
        True
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
    actual = [str(c).strip() for c in df.columns]
    counts = Counter(actual)
    duplicated = sorted(c for c, n in counts.items() if n > 1)
    missing = [c for c in expected_columns if c not in counts]
    extra = [c for c in actual if c not in expected_columns]
    problems = []
    if missing:
        problems.append(f"missing={missing}")
    if extra and not allow_extra_columns:
        problems.append(f"extra={extra}")
    if duplicated:
        problems.append(f"duplicated={duplicated}")
    if problems:
        reason = "Schema mismatch: " + " ".join(problems) + f" (expected exactly {expected_columns})"
        logger.warning("%s rejected — %s", source_name, reason)
        return ValidationResult(False, missing, extra, duplicated, reason)
    return ValidationResult(True, [], extra, [], "")


def validate_table(
    table: LoadedTable,
    expected_columns: list[str],
    allow_extra_columns: bool = False,
) -> tuple[pd.DataFrame | None, ValidationReportRow]:
    """Fully validate one loaded table and normalise its accepted content.

    Rules: exact column set (see :func:`validate_schema`); every ``ID`` must be
    non-empty (otherwise the whole table is rejected); empty ``Shloka`` is
    allowed (the record is later ``Skipped``); ``Verse`` is kept as text.

    Args:
        table: Output of the loader.
        expected_columns: Required column names.
        allow_extra_columns: Tolerate and drop extra columns.

    Returns:
        ``(clean_frame_or_None, report_row)``. The frame has exactly the
        expected columns (stripped names) when accepted.
    """
    sheet = table.sheet
    if table.error or table.frame is None:
        reason = table.error or "Unreadable"
        logger.warning("%s rejected — %s", table.source_name, reason)
        return None, ValidationReportRow(table.file_name, sheet, "Rejected", 0, reason)
    frame = table.frame
    if frame.shape[0] == 0 and frame.shape[1] == 0:
        return None, ValidationReportRow(table.file_name, sheet, "Rejected", 0, "Empty file")
    result = validate_schema(
        frame, expected_columns, source_name=table.source_name, allow_extra_columns=allow_extra_columns
    )
    if not result.is_valid:
        return None, ValidationReportRow(table.file_name, sheet, "Rejected", len(frame), result.reason)
    frame = frame.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    frame = frame[list(expected_columns)]
    if len(frame) == 0:
        return None, ValidationReportRow(table.file_name, sheet, "Rejected", 0, "No data rows")
    if "ID" in frame.columns:
        empty_ids = [i + 2 for i, v in enumerate(frame["ID"]) if not str(v).strip()]
        if empty_ids:
            shown = empty_ids[:20]
            reason = f"Empty ID in {len(empty_ids)} row(s) (spreadsheet rows {shown})"
            logger.warning("%s rejected — %s", table.source_name, reason)
            return None, ValidationReportRow(table.file_name, sheet, "Rejected", len(frame), reason)
        frame["ID"] = [verse_to_text(v) for v in frame["ID"]]
    if "Verse" in frame.columns:
        frame["Verse"] = [verse_to_text(v) for v in frame["Verse"]]
    notes = []
    if "Shloka" in frame.columns:
        empty_shloka = sum(1 for v in frame["Shloka"] if not str(v).strip())
        if empty_shloka:
            notes.append(f"{empty_shloka} empty Shloka cell(s) will be Skipped")
    if table.numeric_verse_rows:
        notes.append(
            f"{len(table.numeric_verse_rows)} Verse cell(s) stored as numbers in the workbook "
            "(trailing zeros such as 1.10 may already be lost)"
        )
    logger.info("%s accepted (%d rows)", table.source_name, len(frame))
    return frame.reset_index(drop=True), ValidationReportRow(
        table.file_name, sheet, "Accepted", len(frame), "; ".join(notes)
    )
