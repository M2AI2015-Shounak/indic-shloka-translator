"""Output file naming and sanitisation."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from shloka_translator.output.naming import build_stem, sanitize_model_name, unique_path


def test_sanitize() -> None:
    """Colons, slashes, backslashes and spaces become dashes."""
    assert sanitize_model_name("qwen3:32b") == "qwen3-32b"
    assert sanitize_model_name("aya-expanse:32b") == "aya-expanse-32b"
    assert sanitize_model_name("a/b\\c d") == "a-b-c-d"
    assert sanitize_model_name("gpt-4.1-mini") == "gpt-4.1-mini"


def test_stem_format() -> None:
    """<model>_<YYYY-MM-DD>_<HH-MM-SS> in the configured zone."""
    when = datetime(2026, 10, 2, 22, 51, 7, tzinfo=ZoneInfo("Asia/Kolkata"))
    assert build_stem("qwen3:32b", when=when) == "qwen3-32b_2026-10-02_22-51-07"


def test_stem_uses_timezone() -> None:
    """The default stem is built in the requested timezone."""
    stem = build_stem("gpt-4.1", "Asia/Kolkata")
    assert stem.startswith("gpt-4.1_") and len(stem.split("_")) == 3


def test_never_overwrite(tmp_path) -> None:
    """Existing names get _1, _2 suffixes.

    Args:
        tmp_path: Temp dir.
    """
    p1 = unique_path(tmp_path, "m_2026-01-01_00-00-00", "xlsx")
    p1.write_text("x")
    p2 = unique_path(tmp_path, "m_2026-01-01_00-00-00", ".xlsx")
    p2.write_text("x")
    p3 = unique_path(tmp_path, "m_2026-01-01_00-00-00", "xlsx")
    assert (p1.name, p2.name, p3.name) == (
        "m_2026-01-01_00-00-00.xlsx",
        "m_2026-01-01_00-00-00_1.xlsx",
        "m_2026-01-01_00-00-00_2.xlsx",
    )
