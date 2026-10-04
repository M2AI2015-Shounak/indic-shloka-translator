"""Shared helpers for the offline test-suite (no network, no GPU)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from shloka_translator.config import AppConfig, load_config
from shloka_translator.models.base import TranslationRequest, TranslationResult, TranslatorBackend, WordMeaning

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"
REFERENCE = REPO / "reference" / "Combined_All_Slokas.xlsx"


def make_config(tmp: Path, inputs: list[Path], **overrides: Any) -> AppConfig:
    """Build a config whose folders live under ``tmp`` and copy ``inputs`` into its input folder.

    Args:
        tmp: Temporary directory.
        inputs: Files to place in the input folder.
        **overrides: Extra nested overrides merged into the config.

    Returns:
        Validated config.
    """
    in_dir = tmp / "in"
    in_dir.mkdir(parents=True, exist_ok=True)
    for f in inputs:
        shutil.copy(f, in_dir / f.name)
    base = {
        "paths": {
            "input_folder": str(in_dir),
            "output_folder": str(tmp / "out"),
            "log_folder": str(tmp / "logs"),
            "cache_folder": str(tmp / "cache"),
            "custom_model_dir": str(tmp / "custom"),
        },
        "language_detection": {
            "use_indiclid": False,
            "use_fasttext_fallback": False,
            "use_llm_tiebreaker": False,
            "cache_file": str(tmp / "cache" / "langid.json"),
        },
    }
    for key, value in overrides.items():
        base.setdefault(key, {}).update(value)
    return load_config(overrides=base)


class MockBackend(TranslatorBackend):
    """Deterministic offline backend that counts calls and can fail on demand."""

    provider = "openai"
    produces_analysis = True
    parallel_mode = "threads"

    def __init__(self, fail_ids: set[str] | None = None, interrupt_after: int | None = None) -> None:
        """Create the mock.

        Args:
            fail_ids: Record IDs whose translation raises.
            interrupt_after: Raise KeyboardInterrupt after this many calls.
        """
        self.name = "mock-model"
        self.fail_ids = fail_ids or set()
        self.interrupt_after = interrupt_after
        self.calls = 0

    def load(self) -> None:
        """No-op."""

    def is_available(self) -> tuple[bool, str]:
        """Always available.

        Returns:
            ``(True, "ready")``.
        """
        return True, "ready"

    def unload(self) -> None:
        """No-op."""

    def translate_batch(self, items: list[TranslationRequest]) -> list[TranslationResult]:
        """Return fake translations.

        Args:
            items: Requests.

        Returns:
            Results.

        Raises:
            RuntimeError: For IDs in ``fail_ids``.
            KeyboardInterrupt: When ``interrupt_after`` is reached.
        """
        out = []
        for it in items:
            self.calls += 1
            if self.interrupt_after is not None and self.calls > self.interrupt_after:
                raise KeyboardInterrupt
            if it.record_id in self.fail_ids:
                raise RuntimeError(f"simulated failure for {it.record_id}")
            out.append(
                TranslationResult(
                    translation=f"English for {it.record_id}",
                    segmentation=it.text.split()[:2],
                    word_meanings=[WordMeaning("w", "m")],
                    input_tokens=11,
                    output_tokens=7,
                    total_tokens=18,
                    latency_s=0.001,
                    has_analysis=True,
                    translator_notes="ambiguous word" if it.record_id == "2" else "",
                )
            )
        return out
