"""Output file naming: ``<model_name>_<YYYY-MM-DD>_<HH-MM-SS>.<ext>``."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

from shloka_translator.utils.timing import now_in_tz

logger = logging.getLogger(__name__)

_UNSAFE = re.compile(r"[:/\\\s<>\"|?*]+")


def sanitize_model_name(name: str) -> str:
    """Make a model name safe for Drive/Windows file names.

    ``:`` ``/`` ``\\`` and whitespace (plus other characters Windows forbids)
    become ``-``.

    Args:
        name: Model name, e.g. ``"qwen3:32b"``.

    Returns:
        Safe name, e.g. ``"qwen3-32b"``.

    Example:
        >>> sanitize_model_name("qwen3:32b"), sanitize_model_name("ALL MODELS")
        ('qwen3-32b', 'ALL-MODELS')
    """
    safe = _UNSAFE.sub("-", name.strip()).strip("-.")
    return safe or "model"


def build_stem(model_name: str, timezone: str = "Asia/Kolkata", when: datetime | None = None) -> str:
    """Build the file stem shared by the workbook and the log.

    Args:
        model_name: Model name (sanitised inside).
        timezone: IANA time zone for the timestamp.
        when: Timestamp override (tests).

    Returns:
        e.g. ``"gpt-4.1-mini_2026-10-02_22-51-07"``.
    """
    ts = when or now_in_tz(timezone)
    return f"{sanitize_model_name(model_name)}_{ts:%Y-%m-%d}_{ts:%H-%M-%S}"


def unique_path(folder: Path, stem: str, extension: str) -> Path:
    """Return a path that does not exist yet, appending ``_1``, ``_2``... if needed.

    Args:
        folder: Target folder (created if missing).
        stem: File stem.
        extension: Extension with or without the dot.

    Returns:
        Non-existing path.

    Example:
        >>> import tempfile
        >>> unique_path(Path(tempfile.gettempdir()), "x_never_exists_123", "xlsx").name
        'x_never_exists_123.xlsx'
    """
    ext = extension if extension.startswith(".") else f".{extension}"
    folder.mkdir(parents=True, exist_ok=True)
    candidate = folder / f"{stem}{ext}"
    n = 1
    while candidate.exists():
        candidate = folder / f"{stem}_{n}{ext}"
        n += 1
    return candidate
