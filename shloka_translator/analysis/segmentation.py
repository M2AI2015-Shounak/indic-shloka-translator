"""Rule-based segmentation fallbacks (no LLM available).

* Sanskrit → ``sanskrit_parser`` sandhi splitting, word by word, with a
  per-record timeout; words that cannot be split are kept whole.
* Other languages → ``indic-nlp-library`` tokenisation, else a regex split.
"""

from __future__ import annotations

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

logger = logging.getLogger(__name__)

_PUNCT = re.compile(r"[।॥|,.;:!?\"'“”‘’()\[\]{}\-—–0-9०-९]+")
_parser_state: dict[str, object] = {}
_parser_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sandhi")

ISO_FOR_INDICNLP = {
    "Hindi": "hi",
    "Marathi": "mr",
    "Nepali": "ne",
    "Bengali": "bn",
    "Assamese": "as",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "Odia": "or",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Sanskrit": "sa",
    "Konkani": "kok",
    "Maithili": "mai",
    "Sindhi": "sd",
    "Kashmiri": "ks",
    "Manipuri": "mni",
    "Bodo": "brx",
    "Dogri": "doi",
    "Santali": "sat",
}


def simple_tokens(text: str) -> list[str]:
    """Whitespace/punctuation tokenisation.

    Args:
        text: Input text.

    Returns:
        Tokens without punctuation or digits.

    Example:
        >>> simple_tokens("राम: वनं गच्छति ॥ १ ॥")
        ['राम', 'वनं', 'गच्छति']
    """
    return [t for t in _PUNCT.sub(" ", text or "").split() if t]


def _get_parser() -> object | None:
    """Create the ``sanskrit_parser`` Parser once.

    Returns:
        Parser instance or ``None`` if the package is unavailable.
    """
    with _parser_lock:
        if "parser" not in _parser_state:
            try:
                from sanskrit_parser import Parser  # type: ignore[import-not-found]

                _parser_state["parser"] = Parser(
                    input_encoding="Devanagari", output_encoding="Devanagari", replace_ending_visarga=None
                )
            except Exception as exc:  # noqa: BLE001
                logger.info("sanskrit_parser unavailable: %s", exc)
                _parser_state["parser"] = None
    return _parser_state["parser"]


def _split_word(parser: object, word: str) -> list[str]:
    """Sandhi-split one word with ``sanskrit_parser`` (best first split).

    Args:
        parser: Parser instance.
        word: Devanagari word.

    Returns:
        Constituent padas (or ``[word]``).
    """
    try:
        splits = parser.split(word, limit=1)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return [word]
    if not splits:
        return [word]
    first = splits[0]
    parts = [str(p) for p in getattr(first, "split", first)] if not isinstance(first, str) else first.split()
    parts = [p.strip() for p in parts if str(p).strip()]
    return parts or [word]


def sanskrit_sandhi_split(text: str, timeout_s: float = 20) -> tuple[list[str], str]:
    """Split a Sanskrit text into padas using ``sanskrit_parser``.

    Args:
        text: Devanagari Sanskrit.
        timeout_s: Time budget for the whole record.

    Returns:
        ``(segments, method_note)``.

    Raises:
        RuntimeError: If ``sanskrit_parser`` is unavailable or the timeout expires.
    """
    parser = _get_parser()
    if parser is None:
        raise RuntimeError("sanskrit_parser not installed")
    words = simple_tokens(text)

    def work() -> list[str]:
        """Split all words.

        Returns:
            Segments.
        """
        out: list[str] = []
        for w in words:
            out.extend(_split_word(parser, w))
        return out

    future = _pool.submit(work)
    try:
        return future.result(timeout=timeout_s), "sanskrit_parser sandhi split"
    except FutureTimeout as exc:
        future.cancel()
        raise RuntimeError(f"sanskrit_parser timed out after {timeout_s:.0f}s") from exc


def tokenize_language(text: str, language: str) -> tuple[list[str], str]:
    """Word tokenisation for non-Sanskrit languages.

    Args:
        text: Input text.
        language: Display name.

    Returns:
        ``(tokens, method_note)``.
    """
    try:
        from indicnlp.tokenize import indic_tokenize  # type: ignore[import-not-found]

        lang = ISO_FOR_INDICNLP.get(language, "hi")
        toks = [t for t in indic_tokenize.trivial_tokenize(text, lang) if not _PUNCT.fullmatch(t)]
        return toks, "indic-nlp-library tokenisation"
    except Exception:  # noqa: BLE001
        return simple_tokens(text), "whitespace tokenisation"


def rule_based_segmentation(text: str, language: str, timeout_s: float = 20) -> tuple[list[str], str]:
    """Best available rule-based segmentation for a language.

    Args:
        text: Input text (Devanagari for Sanskrit).
        language: Display name.
        timeout_s: Sanskrit splitter time budget.

    Returns:
        ``(segments, method_note)``.
    """
    if language == "Sanskrit":
        try:
            return sanskrit_sandhi_split(text, timeout_s)
        except RuntimeError as exc:
            toks, _ = tokenize_language(text, language)
            return toks, f"word split only (sandhi split unavailable: {exc})"
    return tokenize_language(text, language)
