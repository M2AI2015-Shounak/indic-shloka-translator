"""Unicode normalisation, Excel-safe cleaning and small text helpers."""

from __future__ import annotations

import hashlib
import logging
import math
import re
import unicodedata
from typing import Any

logger = logging.getLogger(__name__)

ZWJ = "\u200d"
ZWNJ = "\u200c"
EXCEL_CELL_LIMIT = 32_767

# Zero-width / invisible formatting characters to remove (ZWJ and ZWNJ are kept
# on purpose: they change conjunct rendering in Indic scripts).
_ZERO_WIDTH = re.compile("[\u200b\u200e\u200f\u2060\ufeff\u00ad\u180e\u202a-\u202e\u2066-\u2069]")
# Control characters except tab, newline, carriage return.
_CONTROL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
# Characters that openpyxl refuses to write (illegal in XML 1.0).
_ILLEGAL_XML = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff￾￿]")
_WS = re.compile(r"[ \t ]+")


def normalize_text(value: Any) -> str:
    """Return NFC-normalised text without zero-width or control characters.

    ZWJ (U+200D) and ZWNJ (U+200C) are preserved because they are meaningful in
    Indic scripts. Runs of spaces/tabs are collapsed; surrounding whitespace is
    stripped. ``None``/NaN become ``""``.

    Args:
        value: Any cell value.

    Returns:
        Cleaned string.

    Example:
        >>> normalize_text("  राम\u200b  ")
        'राम'
    """
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value)
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = _CONTROL.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(_WS.sub(" ", line).strip() for line in text.split("\n"))
    return text.strip()


def verse_to_text(value: Any) -> str:
    """Render a ``Verse`` value as text without float corruption.

    Integers (or integral floats such as ``99.0``) lose the spurious ``.0``;
    strings are kept exactly as typed (so ``"1.10"`` stays ``"1.10"``).

    Args:
        value: Raw cell value (int, float or str).

    Returns:
        Verse label as text.

    Example:
        >>> verse_to_text(99.0), verse_to_text("1.10"), verse_to_text("Passage 2")
        ('99', '1.10', 'Passage 2')
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer():
            return str(int(value))
        return repr(value)
    text = normalize_text(value)
    if re.fullmatch(r"-?\d+\.0+", text):
        return text.split(".")[0]
    return text


def excel_safe(value: Any, limit: int = EXCEL_CELL_LIMIT) -> tuple[Any, bool]:
    """Make a value safe for an Excel cell.

    Removes characters that are illegal in XML and truncates strings longer
    than Excel's 32,767-character limit.

    Args:
        value: Cell value; non-strings are returned unchanged.
        limit: Maximum characters allowed.

    Returns:
        Tuple ``(clean_value, was_truncated)``.
    """
    if not isinstance(value, str):
        return value, False
    clean = _ILLEGAL_XML.sub("", value)
    if len(clean) > limit:
        return clean[: limit - 20] + " …[truncated]", True
    return clean, False


def text_hash(*parts: str) -> str:
    """Return a SHA-256 hex digest of the NFC-normalised, joined parts.

    Args:
        *parts: Strings to hash; joined with a unit separator.

    Returns:
        64-character hex digest.

    Example:
        >>> len(text_hash("gpt-4.1", "v1", "राम"))
        64
    """
    joined = "\x1f".join(normalize_text(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def strip_think_tags(text: str) -> str:
    """Remove ``<think>...</think>`` blocks emitted by reasoning models.

    Args:
        text: Raw model output.

    Returns:
        Output without thinking blocks.

    Example:
        >>> strip_think_tags("<think>x</think>{}")
        '{}'
    """
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    return re.sub(r"^.*?</think>", "", text, flags=re.DOTALL).strip()


def truncate(text: str, length: int = 300) -> str:
    """Shorten text for remarks/log lines.

    Args:
        text: Input text.
        length: Maximum length.

    Returns:
        The text, cut to ``length`` characters with an ellipsis if needed.
    """
    text = str(text)
    return text if len(text) <= length else text[: length - 1] + "…"
