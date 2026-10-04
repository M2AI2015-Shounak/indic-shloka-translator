"""Tiered language identification tuned for shlokas (script ≠ language).

Tiers, in order:

1. **Script profiling** (always): empty / punctuation / numeric-only text is
   rejected; unique scripts (Ol Chiki, Meetei Mayek, Han, ...) are decided here.
2. **IndicLID** (AI4Bharat) when installed — primary classifier.
3. **fastText NLLB LID** (``facebook/fasttext-language-identification``) when
   IndicLID is unavailable or returns a non-Indic label with low confidence.
4. **Domain rules** (always): Sanskrit markers (visarga, verse-final ``॥``,
   case endings, avagraha, indeclinables) vs. function words of Hindi, Marathi,
   Nepali, Maithili, Konkani, Bodo, Dogri; Sanskrit in regional / Latin
   scripts; Sindhi vs Kashmiri vs Urdu in Arabic script.
5. **LLM tiebreaker** (optional) for low-confidence results and Sanskrit-like
   text in non-Devanagari scripts.

Results are cached by text hash (in memory and in a JSON file) so detection
runs once per text, not once per model.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from shloka_translator.langid.languages import ISO_TO_NAME, SCRIPT_TAG_TO_SCRIPT, LanguageTable
from shloka_translator.langid.script_detector import (
    DOUBLE_DANDA,
    PARALLEL_INDIC,
    ScriptProfile,
    profile_script,
    to_devanagari,
)
from shloka_translator.utils.text import normalize_text, text_hash

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- data


@dataclass
class DetectionResult:
    """Language-identification outcome for one text.

    Attributes:
        language_name: Display name, e.g. ``"Sanskrit"`` (``""`` if detection failed).
        code: Internal code, e.g. ``"san_Deva"`` (or ISO label for unsupported).
        script: Dominant script name, e.g. ``"Devanagari"``.
        confidence: Confidence in ``[0, 1]``.
        method: Which tier decided, e.g. ``"script+sanskrit_rules"``.
        is_supported: Whether the language is in the 21-language list.
        reason: Skip reason / notes (empty when supported).
        tiebreaker_input_tokens: Tokens used by the LLM tiebreaker (input).
        tiebreaker_output_tokens: Tokens used by the LLM tiebreaker (output).
    """

    language_name: str
    code: str
    script: str
    confidence: float
    method: str
    is_supported: bool
    reason: str = ""
    tiebreaker_input_tokens: int = 0
    tiebreaker_output_tokens: int = 0


@dataclass
class TiebreakOutcome:
    """Structured answer from the LLM tiebreaker.

    Attributes:
        language: Display name from the supported list, or ``"OTHER"``.
        iso_code: Code reported by the model (e.g. ``"san_Deva"``).
        confidence: Model-reported confidence in ``[0, 1]``.
        reason: Short justification.
        model: Model that answered.
        input_tokens: Prompt tokens (provider-reported).
        output_tokens: Completion tokens (provider-reported).
    """

    language: str
    iso_code: str
    confidence: float
    reason: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


TiebreakerFn = Callable[[str, list[str], str], "TiebreakOutcome | None"]

# --------------------------------------------------------------------------- lexicons

_TOKEN_SPLIT = re.compile(r"[\s।॥,.;:!?\-—–\"'“”‘’()\[\]{}|/\\0-9०-९*_]+")

# Function words (Devanagari) — chosen to be rare as stand-alone words in Sanskrit.
DEVANAGARI_STOPWORDS: dict[str, set[str]] = {
    "Hindi": {
        "है",
        "हैं",
        "का",
        "की",
        "के",
        "में",
        "से",
        "को",
        "और",
        "नहीं",
        "था",
        "थे",
        "थी",
        "ने",
        "भी",
        "यह",
        "वह",
        "ये",
        "वे",
        "कि",
        "गया",
        "रहा",
        "लिए",
        "साथ",
        "अपने",
        "होता",
        "होती",
        "करते",
        "हुआ",
        "हम",
        "मैं",
        "आप",
        "क्या",
        "रहे",
        "गई",
        "इस",
        "उस",
    },
    "Marathi": {
        "आहे",
        "आहेत",
        "आणि",
        "नाही",
        "होते",
        "होता",
        "केले",
        "मध्ये",
        "साठी",
        "म्हणून",
        "आम्ही",
        "मी",
        "त्याने",
        "तिने",
        "झाले",
        "आपण",
        "करतो",
        "करते",
        "त्या",
        "हा",
        "ही",
        "आला",
    },
    "Nepali": {
        "छ",
        "छन्",
        "लाई",
        "थियो",
        "हुन्छ",
        "पनि",
        "गर्न",
        "भएको",
        "गरेको",
        "मेरो",
        "त्यो",
        "यो",
        "र",
        "हो",
        "गर्छ",
        "थिए",
        "हरू",
        "छु",
        "गर्दै",
        "भन्ने",
    },
    "Maithili": {"अछि", "छथि", "छल", "अहाँ", "सँ", "भेल", "गेल", "कहलनि", "हमर", "केँ", "छलाह", "रहल", "करैत", "छी", "एहि"},
    "Konkani": {
        "आसा",
        "आसात",
        "हांव",
        "आनी",
        "जाल्यार",
        "म्हळ्यार",
        "तुमी",
        "आमी",
        "खातीर",
        "हाका",
        "ताका",
        "कित्याक",
        "जालें",
        "नाका",
        "आसलें",
    },
    "Bodo": {"आरो", "दं", "जाबाय", "मोनसे", "बिसोर", "बिसोरनि", "आं", "नों", "गोनां", "जायो", "खालाम", "नाङा", "दङ", "मोन"},
    "Dogri": {"दा", "दी", "ऐ", "ऐन", "नेईं", "असें", "कन्नै", "गी", "तुस", "बी", "इक", "होई", "करदा", "ओह्"},
}

SANSKRIT_INDECLINABLES = {
    "च",
    "न",
    "हि",
    "तु",
    "वा",
    "एव",
    "इति",
    "अपि",
    "यथा",
    "तथा",
    "सः",
    "सा",
    "तत्",
    "यत्",
    "अथ",
    "किम्",
    "इव",
    "ततः",
    "यदा",
    "तदा",
    "सह",
    "विना",
    "प्रति",
    "अहम्",
    "त्वम्",
    "वयम्",
    "यूयम्",
    "तम्",
    "नमः",
    "उवाच",
    "स्म",
    "पुनः",
    "सर्वे",
    "यः",
    "तस्य",
    "मम",
    "तव",
    "मे",
    "ते",
}
SANSKRIT_ENDINGS = (
    "स्य",
    "ाय",
    "ेन",
    "ानाम्",
    "ाणाम्",
    "ेषु",
    "ैः",
    "ाभ्याम्",
    "न्ति",
    "ामि",
    "न्ते",
    "स्मिन्",
    "ाः",
    "ेभ्यः",
    "ासु",
    "त्वा",
    "य्य",
    "ाम्",
)

NATIVE_STOPWORDS: dict[str, set[str]] = {
    "Bengali": {
        "এবং",
        "করে",
        "হয়",
        "আমি",
        "তুমি",
        "এই",
        "সেই",
        "না",
        "ছিল",
        "করা",
        "হবে",
        "আছে",
        "জন্য",
        "থেকে",
        "কিন্তু",
        "যে",
        "তার",
        "একটি",
        "আমার",
    },
    "Assamese": {"আৰু", "কৰে", "মই", "আছে", "নাই", "কৰা", "বাবে", "পৰা", "হয়", "এই", "তেওঁ"},
    "Manipuri": {"অমসুং", "ঐহাক", "নহাক", "মহাক", "অদুগী", "নুংশি", "লৈ", "অসি", "য়াম্না", "ঐখোয়"},
    "Gujarati": {"અને", "છે", "છું", "હતું", "આ", "તે", "માં", "થી", "નો", "ની", "નું", "કે", "પણ", "એક"},
    "Punjabi": {"ਅਤੇ", "ਹੈ", "ਹਨ", "ਦਾ", "ਦੀ", "ਦੇ", "ਵਿੱਚ", "ਨੂੰ", "ਤੋਂ", "ਇਹ", "ਉਹ", "ਨਹੀਂ"},
    "Odia": {"ଏବଂ", "ଅଛି", "ଏହି", "ସେ", "ମୁଁ", "ଆମେ", "ପାଇଁ", "ରେ", "କୁ", "ନାହିଁ", "ହେଉଛି"},
    "Tamil": {"மற்றும்", "ஒரு", "இது", "அது", "நான்", "என்", "இந்த", "அந்த", "உள்ளது", "இல்லை", "என்று", "ஆகும்"},
    "Telugu": {"మరియు", "ఒక", "ఈ", "ఆ", "నేను", "ఉంది", "లో", "కు", "అని", "కాదు", "చాలా", "అది"},
    "Kannada": {"ಮತ್ತು", "ಒಂದು", "ಈ", "ಆ", "ನಾನು", "ಇದೆ", "ಅವರು", "ಅದು", "ಇಲ್ಲ", "ಎಂದು", "ಬಹಳ"},
    "Malayalam": {"ഒരു", "ഈ", "ആ", "ഞാൻ", "ആണ്", "ഉണ്ട്", "എന്ന്", "ഇല്ല", "അവൻ", "അവൾ", "കൂടെ", "വളരെ", "ഇത്", "അത്"},
}
SCRIPT_DEFAULT_LANGUAGE = {
    "Bengali": "Bengali",
    "Gurmukhi": "Punjabi",
    "Gujarati": "Gujarati",
    "Odia": "Odia",
    "Tamil": "Tamil",
    "Telugu": "Telugu",
    "Kannada": "Kannada",
    "Malayalam": "Malayalam",
}
UNIQUE_SCRIPT_LANGUAGE = {
    "Ol Chiki": ("Santali", "sat_Olck"),
    "Meetei Mayek": ("Manipuri", "mni_Mtei"),
    "Han": ("Chinese", "zho_Hans"),
    "Hiragana": ("Japanese", "jpn_Jpan"),
    "Katakana": ("Japanese", "jpn_Jpan"),
    "Hangul": ("Korean", "kor_Hang"),
    "Cyrillic": ("Russian (Cyrillic script)", "rus_Cyrl"),
    "Greek": ("Greek", "ell_Grek"),
    "Hebrew": ("Hebrew", "heb_Hebr"),
    "Thai": ("Thai", "tha_Thai"),
    "Sinhala": ("Sinhala", "sin_Sinh"),
    "Tibetan": ("Tibetan", "bod_Tibt"),
    "Myanmar": ("Burmese", "mya_Mymr"),
}
SINDHI_LETTERS = set("ٻڄڃڇڊڌڍڏڙڪڳڱٺٽٿڦ")
KASHMIRI_LETTERS = set("ٲٳۄێٕٟٚ")
URDU_LETTERS = set("ےںٹڈڑھۂۓ")
PERSIAN_LETTERS = set("پچژگ")
IAST_CHARS = set("āīūṛṝḷḹṃṁḥśṣṇṭḍñṅĀĪŪṚṜḶṂḤŚṢṆṬḌÑṄ")
ENGLISH_STOPWORDS = {
    "the",
    "and",
    "is",
    "of",
    "to",
    "in",
    "that",
    "it",
    "was",
    "for",
    "with",
    "as",
    "on",
    "are",
    "this",
    "be",
    "by",
    "he",
    "she",
    "they",
    "his",
    "her",
    "a",
    "an",
    "from",
}
FRENCH_STOPWORDS = {
    "le",
    "la",
    "les",
    "et",
    "est",
    "des",
    "une",
    "un",
    "du",
    "dans",
    "que",
    "pour",
    "qui",
    "pas",
    "sur",
    "au",
    "avec",
    "il",
    "elle",
    "nous",
    "vous",
    "sont",
    "ce",
    "cette",
}
ROMAN_HINDI_STOPWORDS = {
    "hai",
    "hain",
    "ka",
    "ki",
    "ke",
    "mein",
    "se",
    "ko",
    "aur",
    "nahi",
    "nahin",
    "tha",
    "thi",
    "the",
    "ne",
    "bhi",
    "yeh",
    "woh",
    "kya",
    "hum",
    "main",
    "aap",
}

# --------------------------------------------------------------------------- rule helpers


def tokenize_words(text: str) -> list[str]:
    """Split text into words on whitespace, dandas, punctuation and digits.

    Args:
        text: Input text.

    Returns:
        Non-empty word tokens.

    Example:
        >>> tokenize_words("धर्मक्षेत्रे कुरुक्षेत्रे ॥ १ ॥")
        ['धर्मक्षेत्रे', 'कुरुक्षेत्रे']
    """
    return [t for t in _TOKEN_SPLIT.split(text or "") if t]


def sanskrit_score(deva_text: str, script: str = "Devanagari", raw_text: str = "") -> float:
    """Score how Sanskrit-like a (Devanagari-mapped) text is.

    Counts words ending in visarga, virama (not for Tamil/Malayalam, where
    word-final virama is native), Sanskrit case/verb endings, Sanskrit
    indeclinables, plus bonuses for verse-final ``॥`` and avagraha ``ऽ``.

    Args:
        deva_text: Text in (or mapped to) Devanagari.
        script: Original script name (controls virama handling).
        raw_text: Original text, used to look for ``॥``.

    Returns:
        Score ≥ 0; values above ~0.12 are strongly Sanskrit.

    Example:
        >>> sanskrit_score("धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥") > 0.12
        True
    """
    words = tokenize_words(deva_text)
    if not words:
        return 0.0
    count_virama = script not in ("Tamil", "Malayalam")
    hits = 0.0
    for w in words:
        if w.endswith("ः") or (count_virama and w.endswith("्")):
            hits += 1
        elif w.endswith(SANSKRIT_ENDINGS):
            hits += 0.75
        elif w.endswith("ं"):
            hits += 0.3
        if w in SANSKRIT_INDECLINABLES:
            hits += 0.5
        if "ऽ" in w:
            hits += 0.5
    score = hits / len(words)
    source = raw_text or deva_text
    if DOUBLE_DANDA in source or "||" in source:
        score += 0.1
    return score


def stopword_ratios(words: list[str], lexicons: dict[str, set[str]]) -> dict[str, float]:
    """Share of words that are function words of each language.

    Args:
        words: Tokenised words.
        lexicons: Language → stop-word set.

    Returns:
        Language → ratio in ``[0, 1]``.
    """
    if not words:
        return {lang: 0.0 for lang in lexicons}
    return {lang: sum(1 for w in words if w in lex) / len(words) for lang, lex in lexicons.items()}


def _best(ratios: dict[str, float]) -> tuple[str, float, float]:
    """Return the best language, its ratio and the runner-up ratio.

    Args:
        ratios: Language → ratio.

    Returns:
        ``(language, best_ratio, second_ratio)``.
    """
    ordered = sorted(ratios.items(), key=lambda kv: kv[1], reverse=True)
    best_lang, best = ordered[0] if ordered else ("", 0.0)
    second = ordered[1][1] if len(ordered) > 1 else 0.0
    return best_lang, best, second


# --------------------------------------------------------------------------- classifiers


class IndicLIDClassifier:
    """Thin lazy wrapper around AI4Bharat IndicLID (if installed).

    The official repo is cloned to ``indiclid_dir`` and its model zips are
    unpacked into ``<indiclid_dir>/Inference/ai4bharat/models`` by
    :func:`install_indiclid`. Loading is attempted once; failures are logged
    and the detector falls back to fastText / rules.
    """

    def __init__(self, indiclid_dir: Path) -> None:
        """Create an unloaded wrapper.

        Args:
            indiclid_dir: Folder holding the cloned IndicLID repository.
        """
        self.indiclid_dir = indiclid_dir
        self._model = None
        self._tried = False
        self.error: str = ""

    def _load(self) -> bool:
        """Import and initialise IndicLID once.

        Returns:
            ``True`` if the model is ready.
        """
        if self._tried:
            return self._model is not None
        self._tried = True
        inference_dir = self.indiclid_dir / "Inference" / "ai4bharat"
        if not (inference_dir / "IndicLID.py").is_file():
            self.error = f"IndicLID not installed at {self.indiclid_dir}"
            logger.info("%s — using fallback classifiers", self.error)
            return False
        import os

        prev = Path.cwd()
        try:
            import torch

            patch_fasttext_numpy2()
            sys.path.insert(0, str(inference_dir))
            os.chdir(inference_dir)  # IndicLID uses paths relative to its folder
            original_load = torch.load

            def _load_full(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
                """Call ``torch.load`` with ``weights_only=False`` (IndicLID pickles full modules).

                Args:
                    *args: Positional args for ``torch.load``.
                    **kwargs: Keyword args for ``torch.load``.

                Returns:
                    The loaded object.
                """
                kwargs.setdefault("weights_only", False)
                return original_load(*args, **kwargs)

            torch.load = _load_full
            try:
                from IndicLID import IndicLID  # type: ignore[import-not-found]

                self._model = IndicLID(input_threshold=0.5, roman_lid_threshold=0.6)
            finally:
                torch.load = original_load
            logger.info("IndicLID loaded from %s", inference_dir)
            return True
        except Exception as exc:  # noqa: BLE001 - third-party init can fail many ways
            self.error = f"IndicLID load failed: {type(exc).__name__}: {exc}"
            logger.warning(self.error)
            self._model = None
            return False
        finally:
            os.chdir(prev)

    @property
    def available(self) -> bool:
        """Whether IndicLID could be loaded.

        Returns:
            ``True`` if ready.
        """
        return self._load()

    def predict(self, texts: list[str], batch_size: int = 32) -> list[tuple[str, float]]:
        """Predict labels for many texts.

        Args:
            texts: Input texts.
            batch_size: IndicLID batch size.

        Returns:
            ``(label, confidence)`` per text; label like ``"hin_Deva"`` or ``"other"``.
        """
        if not texts or not self._load():
            return [("", 0.0) for _ in texts]
        try:
            outputs = self._model.batch_predict(texts, batch_size)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 - fall back to other tiers
            self.error = f"IndicLID prediction failed: {type(exc).__name__}: {exc}"
            logger.warning(self.error)
            self._model = None
            return [("", 0.0) for _ in texts]
        by_text: dict[str, tuple[str, float]] = {}
        for item in outputs:
            text, label, score = item[0], str(item[1]), float(item[2])
            by_text[text] = (_normalise_label(label), score)
        return [by_text.get(t, ("", 0.0)) for t in texts]


class FastTextLID:
    """Lazy wrapper around ``facebook/fasttext-language-identification`` (NLLB LID)."""

    def __init__(self) -> None:
        """Create an unloaded wrapper."""
        self._model = None
        self._tried = False
        self.error = ""

    def _load(self) -> bool:
        """Download (cached) and load the fastText model once.

        Returns:
            ``True`` if ready.
        """
        if self._tried:
            return self._model is not None
        self._tried = True
        try:
            import fasttext  # type: ignore[import-not-found]
            from huggingface_hub import hf_hub_download

            patch_fasttext_numpy2()
            path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
            self._model = fasttext.load_model(path)
            logger.info("fastText NLLB LID loaded")
            return True
        except Exception as exc:  # noqa: BLE001
            self.error = f"fastText LID unavailable: {type(exc).__name__}: {exc}"
            logger.info(self.error)
            return False

    @property
    def available(self) -> bool:
        """Whether the model is loaded.

        Returns:
            ``True`` if ready.
        """
        return self._load()

    def predict(self, text: str) -> tuple[str, float]:
        """Predict a FLORES-style label for one text.

        Args:
            text: Input text.

        Returns:
            ``(label, confidence)``, e.g. ``("eng_Latn", 0.98)``.
        """
        if not self._load():
            return "", 0.0
        labels, probs = self._model.predict(text.replace("\n", " "), k=1)  # type: ignore[union-attr]
        if not labels:
            return "", 0.0
        return labels[0].replace("__label__", ""), float(probs[0])


def patch_fasttext_numpy2() -> None:
    """Make ``fasttext`` (0.9.x wheels) work with NumPy 2.

    ``fasttext.FastText._FastText.predict`` calls ``np.array(probs, copy=False)``,
    which raises ``ValueError`` on NumPy ≥ 2 (Colab's default). The patch
    retries with a copying conversion. Safe to call repeatedly.
    """
    try:
        import fasttext.FastText as ft  # type: ignore[import-not-found]
        import numpy as np
    except ImportError:
        return
    cls = ft._FastText
    if getattr(cls.predict, "_numpy2_patched", False):
        return
    original = cls.predict

    def predict(self: Any, text: Any, k: int = 1, threshold: float = 0.0, on_unicode_error: str = "strict") -> Any:
        """NumPy-2-safe replacement for ``_FastText.predict``.

        Args:
            self: fastText model.
            text: String or list of strings.
            k: Number of labels.
            threshold: Probability threshold.
            on_unicode_error: Unicode error policy.

        Returns:
            ``(labels, probabilities)`` like the original.
        """
        try:
            return original(self, text, k, threshold, on_unicode_error)
        except ValueError:
            if isinstance(text, list):
                all_labels, all_probs = [], []
                for t in text:
                    labels, probs = predict(self, t, k, threshold, on_unicode_error)
                    all_labels.append(labels)
                    all_probs.append(probs)
                return all_labels, all_probs
            pairs = self.f.predict(text.replace("\n", " "), k, threshold, on_unicode_error)
            probs, labels = zip(*pairs, strict=True) if pairs else ((), ())
            return labels, np.asarray(probs)

    predict._numpy2_patched = True  # type: ignore[attr-defined]
    cls.predict = predict


def _normalise_label(label: str) -> str:
    """Normalise IndicLID labels (e.g. ``tam_Tamil`` → ``tam_Taml``, ``nep`` → ``npi``).

    Args:
        label: Raw label.

    Returns:
        FLORES-style label.

    Example:
        >>> _normalise_label("tam_Tamil")
        'tam_Taml'
    """
    fixes = {"Tamil": "Taml", "Telugu": "Telu", "Bengali": "Beng"}
    if "_" not in label:
        return label
    lang, script = label.split("_", 1)
    script = fixes.get(script, script)
    lang = {"nep": "npi", "kok": "gom", "ori": "ory"}.get(lang, lang)
    return f"{lang}_{script}"


def install_indiclid(target_dir: Path) -> bool:
    """Clone IndicLID and download its three model archives (Colab helper).

    Asset URLs follow the official README (release ``v1.0``):
    ``indiclid-ftn.zip``, ``indiclid-ftr.zip``, ``indiclid-bert.zip``.

    Args:
        target_dir: Where to clone the repository.

    Returns:
        ``True`` on success, ``False`` (with a log message) otherwise.
    """
    import subprocess
    import urllib.request
    import zipfile

    try:
        if not (target_dir / "Inference").is_dir():
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/AI4Bharat/IndicLID.git", str(target_dir)],
                check=True,
                capture_output=True,
                timeout=600,
            )
        models_dir = target_dir / "Inference" / "ai4bharat" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        base = "https://github.com/AI4Bharat/IndicLID/releases/download/v1.0/"
        for name in ("indiclid-ftn", "indiclid-ftr", "indiclid-bert"):
            if (models_dir / name).is_dir():
                continue
            zip_path = models_dir / f"{name}.zip"
            logger.info("Downloading %s ...", name)
            urllib.request.urlretrieve(base + f"{name}.zip", zip_path)  # noqa: S310 - fixed https URL
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(models_dir)
            zip_path.unlink(missing_ok=True)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "IndicLID installation failed (%s: %s); rule-based/fastText detection will be used", type(exc).__name__, exc
        )
        return False


# --------------------------------------------------------------------------- detector


class LanguageDetector:
    """Tiered, cached language detector.

    Example:
        >>> det = LanguageDetector(LanguageTable.from_config(None), use_indiclid=False, use_fasttext=False)
        >>> det.detect("कर्मण्येवाधिकारस्ते मा फलेषु कदाचन ।").language_name
        'Sanskrit'
    """

    def __init__(
        self,
        languages: LanguageTable,
        *,
        min_confidence: float = 0.60,
        use_indiclid: bool = True,
        indiclid_dir: Path | None = None,
        use_fasttext: bool = True,
        tiebreaker: TiebreakerFn | None = None,
        tiebreaker_name: str = "",
        cache_file: Path | None = None,
    ) -> None:
        """Configure the detector.

        Args:
            languages: Supported-language table.
            min_confidence: Below this, the LLM tiebreaker is consulted.
            use_indiclid: Try IndicLID as primary classifier.
            indiclid_dir: Location of the cloned IndicLID repository.
            use_fasttext: Try the NLLB fastText LID as fallback.
            tiebreaker: Optional callable ``(text, candidates, hint) -> TiebreakOutcome``.
            tiebreaker_name: Model name used by the tiebreaker (for audit).
            cache_file: JSON file for the persistent detection cache.
        """
        self.languages = languages
        self.min_confidence = min_confidence
        self.indiclid = IndicLIDClassifier(indiclid_dir) if (use_indiclid and indiclid_dir) else None
        self.fasttext = FastTextLID() if use_fasttext else None
        self.tiebreaker = tiebreaker
        self.tiebreaker_name = tiebreaker_name
        self.cache_file = cache_file
        self._cache: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._load_cache()

    # ------------------------------------------------------------------ cache
    def _load_cache(self) -> None:
        """Load the persistent cache file if present."""
        if self.cache_file and self.cache_file.is_file():
            try:
                self._cache = json.loads(self.cache_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Ignoring unreadable langid cache %s: %s", self.cache_file, exc)
                self._cache = {}

    def save_cache(self) -> None:
        """Persist the detection cache (best effort)."""
        if not self.cache_file:
            return
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = json.dumps(self._cache, ensure_ascii=False)
            self.cache_file.write_text(data, encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not write langid cache: %s", exc)

    def _cache_key(self, text: str) -> str:
        """Cache key for a text (includes the supported list so edits invalidate it).

        Args:
            text: Normalised text.

        Returns:
            SHA-256 hex digest.
        """
        return text_hash("langid-v1", ",".join(sorted(self.languages.names)), text)

    # ------------------------------------------------------------------ public
    def detect_many(self, texts: list[str]) -> list[DetectionResult]:
        """Detect languages for many texts, using the cache and batching IndicLID.

        Args:
            texts: Input texts (one per record).

        Returns:
            One :class:`DetectionResult` per input, in order.
        """
        normalised = [normalize_text(t) for t in texts]
        results: list[DetectionResult | None] = [None] * len(texts)
        pending: dict[str, list[int]] = {}
        for i, text in enumerate(normalised):
            key = self._cache_key(text)
            with self._lock:
                cached = self._cache.get(key)
            if cached:
                res = DetectionResult(**cached)
                res.tiebreaker_input_tokens = 0
                res.tiebreaker_output_tokens = 0
                results[i] = res
            else:
                pending.setdefault(text, []).append(i)
        unique = list(pending)
        classifier_labels: dict[str, tuple[str, float, str]] = {}
        if unique and self.indiclid is not None and self.indiclid.available:
            for text, (label, conf) in zip(unique, self.indiclid.predict(unique), strict=True):
                classifier_labels[text] = (label, conf, "indiclid")
        for text in unique:
            try:
                res = self._detect_one(text, classifier_labels.get(text))
            except Exception as exc:
                logger.exception("Language detection failed")
                res = DetectionResult(
                    "", "", "", 0.0, "error", False, f"Language detection failed: {type(exc).__name__}: {exc}"
                )
            for i in pending[text]:
                results[i] = res
            if res.method != "error":
                stored = asdict(res)
                with self._lock:
                    self._cache[self._cache_key(text)] = stored
        return [r for r in results if r is not None]

    def detect(self, text: str) -> DetectionResult:
        """Detect the language of one text.

        Args:
            text: Input text.

        Returns:
            :class:`DetectionResult`.
        """
        return self.detect_many([text])[0]

    # ------------------------------------------------------------------ core
    def _detect_one(self, text: str, classifier: tuple[str, float, str] | None) -> DetectionResult:
        """Run tiers 1, 3, 4 and 5 for one text (tier 2 output passed in).

        Args:
            text: Normalised text.
            classifier: IndicLID ``(label, confidence, method)`` if available.

        Returns:
            Final :class:`DetectionResult`.
        """
        profile = profile_script(text)
        if not profile.is_linguistic:
            return DetectionResult("", "", "", 1.0, "script_profile", False, "Empty or non-linguistic shloka")
        script = profile.dominant
        rule = self._rules(text, profile)
        result, wants_tiebreak = self._combine(text, profile, rule, classifier)
        if (wants_tiebreak or result.confidence < self.min_confidence) and self.tiebreaker is not None:
            result = self._apply_tiebreaker(text, script, result)
        return self._finalise(result)

    def _rules(self, text: str, profile: ScriptProfile) -> tuple[str, float, str, bool]:
        """Rule-based decision from script + lexical evidence.

        Args:
            text: Normalised text.
            profile: Script profile of ``text``.

        Returns:
            ``(language_name, confidence, code_or_label, sanskrit_markers_strong)``.
        """
        script = profile.dominant
        words = tokenize_words(text)
        if script in UNIQUE_SCRIPT_LANGUAGE:
            name, code = UNIQUE_SCRIPT_LANGUAGE[script]
            return name, 0.99, code, False
        if script == "Grantha" or ("Grantha" in profile.counts and script == "Tamil"):
            return "Sanskrit", 0.85, "san_Deva", True
        if script == "Devanagari":
            return self._rules_devanagari(text, words)
        if script in PARALLEL_INDIC:
            return self._rules_regional(text, words, script)
        if script == "Arabic":
            return self._rules_arabic(text)
        if script == "Latin":
            return self._rules_latin(text, words)
        return f"Unknown ({script} script)", 0.5, "und", False

    def _rules_devanagari(self, text: str, words: list[str]) -> tuple[str, float, str, bool]:
        """Sanskrit vs Hindi/Marathi/Nepali/Maithili/Konkani/Bodo/Dogri.

        Args:
            text: Normalised text.
            words: Tokenised words.

        Returns:
            Same tuple as :meth:`_rules`.
        """
        s_score = sanskrit_score(text, "Devanagari", text)
        ratios = stopword_ratios(words, DEVANAGARI_STOPWORDS)
        best_lang, best, second = _best(ratios)
        best_hits = round(best * len(words))
        strong = s_score >= 0.12
        if best >= 0.06 and (best_hits >= 2 or best >= 0.3) and best * 2.5 >= s_score:
            margin = best - second
            conf = min(0.95, 0.55 + best * 2 + margin)
            return best_lang, conf, self.languages.code_for(best_lang, "Devanagari") or best_lang, strong
        if strong:
            return "Sanskrit", min(0.98, 0.62 + s_score), "san_Deva", True
        if DOUBLE_DANDA in text:
            return "Sanskrit", 0.5, "san_Deva", False
        if best > 0:
            return best_lang, 0.5, self.languages.code_for(best_lang, "Devanagari"), False
        return "Hindi", 0.4, "hin_Deva", False

    def _rules_regional(self, text: str, words: list[str], script: str) -> tuple[str, float, str, bool]:
        """Native language vs Sanskrit written in a regional Indic script.

        Args:
            text: Normalised text.
            words: Tokenised words (native script).
            script: Dominant script.

        Returns:
            Same tuple as :meth:`_rules`.
        """
        mapped = to_devanagari(text, script)
        s_score = sanskrit_score(mapped, script, text)
        native = SCRIPT_DEFAULT_LANGUAGE[script]
        candidates = {native: NATIVE_STOPWORDS.get(native, set())}
        if script == "Bengali":
            candidates = {k: NATIVE_STOPWORDS[k] for k in ("Bengali", "Assamese", "Manipuri")}
        ratios = stopword_ratios(words, candidates)
        if script == "Bengali" and ("ৰ" in text or "ৱ" in text):
            ratios["Assamese"] = ratios.get("Assamese", 0) + 0.2
        best_lang, best, _ = _best(ratios)
        if s_score >= 0.15 and best < s_score / 2:
            return "Sanskrit", min(0.9, 0.5 + s_score), "san_Deva", True
        if best > 0:
            return best_lang, min(0.97, 0.75 + best), self.languages.code_for(best_lang, script), False
        return native, 0.8 if s_score < 0.1 else 0.55, self.languages.code_for(native, script), s_score >= 0.1

    def _rules_arabic(self, text: str) -> tuple[str, float, str, bool]:
        """Sindhi vs Kashmiri vs Urdu vs Persian/Arabic in Perso-Arabic script.

        Args:
            text: Normalised text.

        Returns:
            Same tuple as :meth:`_rules`.
        """
        chars = set(text)
        if chars & SINDHI_LETTERS:
            return "Sindhi", 0.9, "snd_Arab", False
        if chars & KASHMIRI_LETTERS:
            return "Kashmiri", 0.85, "kas_Arab", False
        if chars & URDU_LETTERS:
            return "Urdu", 0.8, "urd_Arab", False
        if chars & PERSIAN_LETTERS:
            return "Persian", 0.6, "pes_Arab", False
        return "Arabic", 0.6, "arb_Arab", False

    def _rules_latin(self, text: str, words: list[str]) -> tuple[str, float, str, bool]:
        """Romanised Sanskrit (IAST) / romanised Hindi vs English / French.

        Args:
            text: Normalised text.
            words: Tokenised words.

        Returns:
            Same tuple as :meth:`_rules`.
        """
        lower = [w.lower() for w in words]
        if not lower:
            return "Unknown (Latin script)", 0.3, "und_Latn", False
        iast_ratio = sum(1 for w in words if set(w) & IAST_CHARS) / len(words)
        if iast_ratio >= 0.2:
            return "Sanskrit", min(0.9, 0.55 + iast_ratio), "san_Deva", True
        ratios = stopword_ratios(
            lower, {"English": ENGLISH_STOPWORDS, "French": FRENCH_STOPWORDS, "Hindi": ROMAN_HINDI_STOPWORDS}
        )
        best_lang, best, second = _best(ratios)
        if best >= 0.08:
            code = {"English": "eng_Latn", "French": "fra_Latn", "Hindi": "hin_Latn"}[best_lang]
            return best_lang, min(0.97, 0.6 + best + (best - second)), code, False
        if self.fasttext is not None and self.fasttext.available:
            label, conf = self.fasttext.predict(text)
            if label:
                return self.languages.name_for_code(label), conf, label, False
        return "Unknown (Latin script)", 0.4, "und_Latn", False

    def _combine(
        self,
        text: str,
        profile: ScriptProfile,
        rule: tuple[str, float, str, bool],
        classifier: tuple[str, float, str] | None,
    ) -> tuple[DetectionResult, bool]:
        """Merge classifier output with domain rules.

        Args:
            text: Normalised text.
            profile: Script profile.
            rule: Output of :meth:`_rules`.
            classifier: IndicLID result, if any.

        Returns:
            ``(result, wants_tiebreak)``.
        """
        script = profile.dominant
        r_name, r_conf, r_code, sanskrit_strong = rule
        rule_res = DetectionResult(r_name, r_code, script, round(r_conf, 3), "script+rules", False)
        if r_name == "Sanskrit":
            rule_res.method = "script+sanskrit_rules"
        unique_script = script in UNIQUE_SCRIPT_LANGUAGE
        label, conf, method = classifier if classifier else ("", 0.0, "")
        indic_codes = self.languages.all_codes() | {"urd_Arab"}
        weak_label = not label or (label not in indic_codes and conf < self.min_confidence)
        if (
            weak_label
            and not unique_script
            and script in ("Devanagari", *PARALLEL_INDIC)
            and self.fasttext is not None
            and self.fasttext.available
        ):
            label, conf = self.fasttext.predict(text)
            method = "fasttext"
        if not label or label == "other" or unique_script:
            return rule_res, sanskrit_strong and script != "Devanagari"
        cls_name = self.languages.name_for_code(label)
        cls_res = DetectionResult(cls_name, label, script, round(conf, 3), method, False)
        if script == "Devanagari":
            if cls_name == "Sanskrit":
                return cls_res, False
            words = tokenize_words(text)
            sw = stopword_ratios(words, DEVANAGARI_STOPWORDS).get(cls_name, 0.0)
            if sanskrit_strong and sw < 0.04:
                rule_res.method = f"{method}+sanskrit_rules (classifier said {label})"
                return rule_res, False
            if r_name != "Sanskrit" and r_conf >= 0.8 and r_name != cls_name and conf < 0.8:
                return rule_res, True
            return cls_res, conf < self.min_confidence
        if script in PARALLEL_INDIC or script == "Latin":
            if sanskrit_strong:
                rule_res.method = f"{method}+sanskrit_rules (classifier said {label})"
                return rule_res, True
            if label.endswith("_Latn") and cls_name in self.languages.supported:
                cls_res.code = self.languages.code_for(cls_name)
                cls_res.method = f"{method} (romanised)"
            return cls_res, conf < self.min_confidence
        if script == "Arabic":
            if r_name in ("Sindhi", "Kashmiri") and cls_name not in ("Sindhi", "Kashmiri"):
                return rule_res, conf < self.min_confidence
            return cls_res, conf < self.min_confidence
        return cls_res, conf < self.min_confidence

    def _apply_tiebreaker(self, text: str, script: str, current: DetectionResult) -> DetectionResult:
        """Ask the LLM tiebreaker and adopt its answer when it is more confident.

        Args:
            text: Normalised text.
            script: Dominant script.
            current: Result before the tiebreak.

        Returns:
            Possibly updated result, always carrying tiebreaker token counts.
        """
        candidates = [*self.languages.names, "OTHER"]
        hint = f"script={script}; current guess={current.language_name} ({current.confidence:.2f})"
        try:
            outcome = self.tiebreaker(text, candidates, hint) if self.tiebreaker else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM tiebreaker failed: %s: %s", type(exc).__name__, exc)
            return current
        if outcome is None:
            return current
        current.tiebreaker_input_tokens += outcome.input_tokens
        current.tiebreaker_output_tokens += outcome.output_tokens
        lang = outcome.language.strip()
        if lang.upper() == "OTHER":
            other_name = ISO_TO_NAME.get(outcome.iso_code.split("_")[0].lower(), "Other (not in supported list)")
            if outcome.confidence >= current.confidence or not self.languages.is_supported_name(current.language_name):
                current.language_name = other_name
                current.code = outcome.iso_code or "und"
                current.confidence = round(outcome.confidence, 3)
                current.method = f"llm_tiebreaker({outcome.model})"
            return current
        if self.languages.is_supported_name(lang) and (
            outcome.confidence >= current.confidence or lang == current.language_name
        ):
            current.language_name = lang
            current.code = self.languages.code_for(lang, script if script != "Latin" else None)
            current.confidence = round(
                max(outcome.confidence, current.confidence if lang == current.language_name else 0.0), 3
            )
            current.method = f"llm_tiebreaker({outcome.model})"
        return current

    def _finalise(self, result: DetectionResult) -> DetectionResult:
        """Set ``is_supported`` and the skip reason.

        Args:
            result: Intermediate result.

        Returns:
            The same object, completed.
        """
        result.is_supported = self.languages.is_supported_name(result.language_name)
        if result.is_supported:
            if not result.code:
                result.code = self.languages.code_for(result.language_name, result.script)
            result.reason = ""
        else:
            code = result.code or "und"
            result.reason = (
                f"Unsupported language: {result.language_name or 'Unknown'} ({code}), "
                f"confidence {result.confidence:.2f} — skipped"
            )
        return result


def script_tag_name(code: str) -> str:
    """Return the script name implied by a FLORES code.

    Args:
        code: e.g. ``"kas_Arab"``.

    Returns:
        e.g. ``"Arabic"``; ``""`` when unknown.
    """
    return SCRIPT_TAG_TO_SCRIPT.get(code.rsplit("_", maxsplit=1)[-1], "") if "_" in code else ""
