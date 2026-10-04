"""Unicode-block script profiling and Indic → Devanagari code-point mapping."""

from __future__ import annotations

import logging
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# (script name, first code point, last code point)
SCRIPT_BLOCKS: tuple[tuple[str, int, int], ...] = (
    ("Devanagari", 0x0900, 0x097F),
    ("Devanagari", 0xA8E0, 0xA8FF),  # Devanagari Extended
    ("Devanagari", 0x1CD0, 0x1CFF),  # Vedic Extensions
    ("Bengali", 0x0980, 0x09FF),
    ("Gurmukhi", 0x0A00, 0x0A7F),
    ("Gujarati", 0x0A80, 0x0AFF),
    ("Odia", 0x0B00, 0x0B7F),
    ("Tamil", 0x0B80, 0x0BFF),
    ("Grantha", 0x11300, 0x1137F),
    ("Telugu", 0x0C00, 0x0C7F),
    ("Kannada", 0x0C80, 0x0CFF),
    ("Malayalam", 0x0D00, 0x0D7F),
    ("Sinhala", 0x0D80, 0x0DFF),
    ("Ol Chiki", 0x1C50, 0x1C7F),
    ("Meetei Mayek", 0xABC0, 0xABFF),
    ("Meetei Mayek", 0xAAE0, 0xAAFF),
    ("Arabic", 0x0600, 0x06FF),
    ("Arabic", 0x0750, 0x077F),
    ("Arabic", 0x08A0, 0x08FF),
    ("Arabic", 0xFB50, 0xFDFF),
    ("Arabic", 0xFE70, 0xFEFF),
    ("Latin", 0x0041, 0x005A),
    ("Latin", 0x0061, 0x007A),
    ("Latin", 0x00C0, 0x024F),
    ("Latin", 0x1E00, 0x1EFF),
    ("Cyrillic", 0x0400, 0x04FF),
    ("Greek", 0x0370, 0x03FF),
    ("Hebrew", 0x0590, 0x05FF),
    ("Thai", 0x0E00, 0x0E7F),
    ("Tibetan", 0x0F00, 0x0FFF),
    ("Myanmar", 0x1000, 0x109F),
    ("Han", 0x4E00, 0x9FFF),
    ("Han", 0x3400, 0x4DBF),
    ("Hiragana", 0x3040, 0x309F),
    ("Katakana", 0x30A0, 0x30FF),
    ("Hangul", 0xAC00, 0xD7AF),
)

# Scripts whose blocks are laid out parallel to Devanagari (ISCII heritage).
PARALLEL_INDIC = {
    "Bengali": 0x0980,
    "Gurmukhi": 0x0A00,
    "Gujarati": 0x0A80,
    "Odia": 0x0B00,
    "Tamil": 0x0B80,
    "Telugu": 0x0C00,
    "Kannada": 0x0C80,
    "Malayalam": 0x0D00,
}
INDIC_SCRIPTS = {"Devanagari", *PARALLEL_INDIC, "Ol Chiki", "Meetei Mayek", "Grantha"}
DANDA = "।"
DOUBLE_DANDA = "॥"


@dataclass
class ScriptProfile:
    """Character statistics for one text.

    Attributes:
        counts: Letters/marks per script name.
        letters: Total script-bearing characters counted.
        dominant: Script with the most characters (``""`` if none).
        dominant_ratio: Share of ``dominant`` among counted characters.
        digits: Count of digit characters (any script).
        is_linguistic: ``False`` for empty / punctuation-only / numeric-only text.
    """

    counts: dict[str, int] = field(default_factory=dict)
    letters: int = 0
    dominant: str = ""
    dominant_ratio: float = 0.0
    digits: int = 0
    is_linguistic: bool = False


def script_of_char(ch: str) -> str | None:
    """Return the script name of a character, if it is in a known block.

    Args:
        ch: Single character.

    Returns:
        Script name, or ``None`` for punctuation/unknown blocks.

    Example:
        >>> script_of_char("क"), script_of_char("a"), script_of_char("।")
        ('Devanagari', 'Latin', None)
    """
    cp = ord(ch)
    if ch in (DANDA, DOUBLE_DANDA):
        return None  # shared punctuation used by all Indic scripts
    for name, lo, hi in SCRIPT_BLOCKS:
        if lo <= cp <= hi:
            return name
    return None


def profile_script(text: str) -> ScriptProfile:
    """Count characters per Unicode script block.

    Digits (including Devanagari numerals) and punctuation are not counted as
    letters, so numeric-only text is reported as non-linguistic.

    Args:
        text: Input text.

    Returns:
        :class:`ScriptProfile`.

    Example:
        >>> profile_script("धर्मक्षेत्रे कुरुक्षेत्रे").dominant
        'Devanagari'
    """
    counts: Counter[str] = Counter()
    digits = 0
    for ch in text or "":
        cat = unicodedata.category(ch)
        if cat == "Nd":
            digits += 1
            continue
        if not (cat.startswith("L") or cat.startswith("M")):
            continue
        script = script_of_char(ch)
        if script:
            counts[script] += 1
        else:
            counts["Other"] += 1
    letters = sum(counts.values())
    if not letters:
        return ScriptProfile({}, 0, "", 0.0, digits, False)
    dominant, top = counts.most_common(1)[0]
    if dominant == "Other" and len(counts) > 1:
        dominant, top = counts.most_common(2)[1]
    return ScriptProfile(dict(counts), letters, dominant, top / letters, digits, True)


def to_devanagari(text: str, script: str) -> str:
    """Map a parallel-layout Indic script to Devanagari code points.

    This is a lossless-enough *approximation* used for rule-based Sanskrit
    detection and as a last-resort transliteration; proper transliteration
    (``indic_transliteration`` / ``aksharamukha``) is preferred when installed.

    Args:
        text: Text in ``script``.
        script: One of the parallel Indic scripts (Bengali, Telugu, ...).

    Returns:
        Text with that script's characters shifted into the Devanagari block.

    Example:
        >>> to_devanagari("ధర్మ", "Telugu")
        'धर्म'
    """
    base = PARALLEL_INDIC.get(script)
    if base is None:
        return text
    out = []
    for ch in text:
        cp = ord(ch)
        if base <= cp <= base + 0x7F:
            out.append(chr(0x0900 + (cp - base)))
        else:
            out.append(ch)
    return "".join(out)
