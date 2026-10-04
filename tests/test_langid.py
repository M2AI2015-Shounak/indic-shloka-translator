"""Language detection: reference file, mixed-language fixture, edge cases."""

from __future__ import annotations

import pandas as pd

from shloka_translator.langid.detector import LanguageDetector, TiebreakOutcome
from shloka_translator.langid.languages import LanguageTable
from shloka_translator.langid.script_detector import profile_script, to_devanagari
from tests.helpers import FIXTURES, REFERENCE


def _detector(**kw) -> LanguageDetector:
    """Offline detector (rules only).

    Args:
        **kw: Extra constructor arguments.

    Returns:
        LanguageDetector.
    """
    return LanguageDetector(LanguageTable.from_config(None), use_indiclid=False, use_fasttext=False, **kw)


def test_all_reference_records_are_sanskrit() -> None:
    """All 215 reference shlokas are detected as Sanskrit (acceptance criterion)."""
    df = pd.read_excel(REFERENCE, dtype=str)
    results = _detector().detect_many(df["Shloka"].tolist())
    assert len(results) == 215
    wrong = [(i, r.language_name) for i, r in enumerate(results) if r.language_name != "Sanskrit"]
    assert not wrong, wrong
    assert all(r.is_supported and r.code == "san_Deva" for r in results)


def test_mixed_language_fixture_accuracy() -> None:
    """Report accuracy on the mixed fixture; unsupported languages must be skipped."""
    df = pd.read_csv(FIXTURES / "mixed_languages.csv", dtype=str, keep_default_na=False)
    results = _detector().detect_many(df["Shloka"].tolist())
    expected = [s.split(": ", 1)[1] for s in df["Source"]]
    correct = 0
    for exp, res in zip(expected, results, strict=True):
        got = res.language_name or "NON-LINGUISTIC"
        correct += got == exp
        if exp in ("English", "Urdu", "French", "Chinese", "NON-LINGUISTIC"):
            assert not res.is_supported and res.reason, exp
    accuracy = correct / len(expected)
    print(f"Mixed-language fixture accuracy (rules only): {accuracy:.1%} ({correct}/{len(expected)})")
    assert accuracy >= 0.9


def test_sanskrit_in_regional_scripts_and_iast() -> None:
    """Sanskrit written in Bengali/Telugu/Kannada and IAST is Sanskrit, with the real script recorded."""
    det = _detector()
    rows = pd.read_csv(FIXTURES / "mixed_languages.csv", dtype=str, keep_default_na=False)
    sanskrit = rows[rows["Source"].str.endswith("Sanskrit")]["Shloka"].tolist()
    results = det.detect_many(sanskrit)
    assert [r.language_name for r in results] == ["Sanskrit"] * len(sanskrit)
    assert {r.script for r in results} >= {"Devanagari", "Bengali", "Telugu", "Kannada", "Latin"}


def test_non_linguistic() -> None:
    """Empty, punctuation-only and numeral-only text is skipped."""
    det = _detector()
    for text in ["", "   ", "॥ १२ ॥", "12345", "--- ..."]:
        res = det.detect(text)
        assert not res.is_supported and res.reason == "Empty or non-linguistic shloka"


def test_hindi_not_confused_with_sanskrit() -> None:
    """Plain Hindi prose stays Hindi."""
    res = _detector().detect("राम ने रावण को मारा और सीता को वापस लाया। यह कथा बहुत पुरानी है।")
    assert res.language_name == "Hindi"


def test_tiebreaker_used_and_tokens_recorded() -> None:
    """Low-confidence results consult the tiebreaker; its tokens are recorded."""
    calls = []

    def tb(text: str, candidates: list[str], hint: str) -> TiebreakOutcome:
        """Fake tiebreaker.

        Args:
            text: Text.
            candidates: Candidates.
            hint: Hint.

        Returns:
            Outcome.
        """
        calls.append(hint)
        assert "OTHER" in candidates
        return TiebreakOutcome("Sanskrit", "san_Deva", 0.95, "test", "fake", 12, 3)

    det = _detector(tiebreaker=tb, tiebreaker_name="fake")
    res = det.detect("কর্মণ্যেবাধিকারস্তে মা ফলেষু কদাচন । মা কর্মফলহেতুর্ভূর্মা তে সঙ্গোঽস্ত্বকর্মণি ॥")
    assert calls and res.language_name == "Sanskrit"
    assert res.tiebreaker_input_tokens == 12 and res.tiebreaker_output_tokens == 3


def test_cache_file_roundtrip(tmp_path) -> None:
    """Detections are cached to disk and re-used (with zero tiebreaker tokens).

    Args:
        tmp_path: Temp dir.
    """
    cache = tmp_path / "lid.json"
    d1 = _detector(cache_file=cache)
    d1.detect("धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः ।")
    d1.save_cache()
    d2 = _detector(cache_file=cache)
    assert d2._cache and d2.detect("धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः ।").language_name == "Sanskrit"


def test_script_profile() -> None:
    """Script profiling basics."""
    assert profile_script("ᱤᱧ ᱫᱚ").dominant == "Ol Chiki"
    assert profile_script("१२३").is_linguistic is False
    assert to_devanagari("ధర్మ", "Telugu") == "धर्म"
