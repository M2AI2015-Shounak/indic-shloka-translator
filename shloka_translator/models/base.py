"""Backend interface and the request/result data classes shared by all models."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)


@dataclass
class TranslationRequest:
    """One record to translate.

    Attributes:
        index: Position in the merged input (used to keep output order).
        record_id: ``ID`` from the input.
        text: Shloka text (normalised).
        source: ``Source`` field.
        verse: ``Verse`` field (text).
        language_name: Detected language, e.g. ``"Sanskrit"``.
        language_code: Internal code, e.g. ``"san_Deva"``.
        script: Script the text is written in, e.g. ``"Bengali"``.
    """

    index: int
    record_id: str
    text: str
    source: str = ""
    verse: str = ""
    language_name: str = ""
    language_code: str = ""
    script: str = ""


@dataclass
class WordMeaning:
    """A segment and its contextual English meaning.

    Attributes:
        word: Segment in the original script.
        meaning: English meaning.
    """

    word: str
    meaning: str


@dataclass
class TranslationResult:
    """Output of one backend call for one record.

    Attributes:
        translation: English translation (``""`` on failure).
        segmentation: Word segmentation (may be empty for seq2seq models).
        word_meanings: Word meanings (may be empty for seq2seq models).
        input_tokens: Provider/tokenizer-reported input tokens.
        output_tokens: Provider/tokenizer-reported output tokens.
        total_tokens: Provider total or input + output.
        latency_s: Wall-clock seconds for this record.
        raw_response: Raw text (kept only in the debug cache).
        error: Error message if the record failed, else ``None``.
        reasoning_tokens: Reasoning tokens when the provider reports them.
        retries: Number of retries spent on this record.
        remarks: Notes for the ``Remarks`` column (chunking, transliteration, ...).
        translator_notes: Model's own ambiguity notes.
        has_analysis: Whether segmentation/meanings came from the model itself.
    """

    translation: str = ""
    segmentation: list[str] = field(default_factory=list)
    word_meanings: list[WordMeaning] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_s: float = 0.0
    raw_response: str = ""
    error: str | None = None
    reasoning_tokens: int = 0
    retries: int = 0
    remarks: list[str] = field(default_factory=list)
    translator_notes: str = ""
    has_analysis: bool = False

    @property
    def ok(self) -> bool:
        """Whether the record succeeded.

        Returns:
            ``True`` if there is a translation and no error.
        """
        return self.error is None and bool(self.translation.strip())

    def to_cache(self) -> dict[str, Any]:
        """Serialise for the JSONL checkpoint.

        Returns:
            JSON-friendly dict.
        """
        return {
            "translation": self.translation,
            "segmentation": self.segmentation,
            "word_meanings": [[w.word, w.meaning] for w in self.word_meanings],
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_s": self.latency_s,
            "raw_response": self.raw_response[:20000],
            "error": self.error,
            "reasoning_tokens": self.reasoning_tokens,
            "retries": self.retries,
            "remarks": self.remarks,
            "translator_notes": self.translator_notes,
            "has_analysis": self.has_analysis,
        }

    @classmethod
    def from_cache(cls, data: dict[str, Any]) -> TranslationResult:
        """Rebuild from a checkpoint entry.

        Args:
            data: Dict produced by :meth:`to_cache`.

        Returns:
            :class:`TranslationResult`.
        """
        meanings = [WordMeaning(str(w), str(m)) for w, m in data.get("word_meanings", [])]
        return cls(
            translation=data.get("translation", ""),
            segmentation=list(data.get("segmentation", [])),
            word_meanings=meanings,
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            total_tokens=int(data.get("total_tokens", 0)),
            latency_s=float(data.get("latency_s", 0.0)),
            raw_response=data.get("raw_response", ""),
            error=data.get("error"),
            reasoning_tokens=int(data.get("reasoning_tokens", 0)),
            retries=int(data.get("retries", 0)),
            remarks=list(data.get("remarks", [])),
            translator_notes=data.get("translator_notes", ""),
            has_analysis=bool(data.get("has_analysis", False)),
        )


class TranslatorBackend(ABC):
    """Abstract translator backend.

    Attributes:
        name: Registry name shown in the dropdown (e.g. ``"gpt-4.1"``).
        provider: ``"openai"``, ``"ollama"``, ``"hf_seq2seq"`` or ``"custom"``.
        produces_analysis: Whether the backend returns segmentation/meanings itself.
        parallel_mode: ``"threads"`` (I/O-bound API) or ``"batch"`` (single-process GPU batching).
        runs_on_local_gpu: Whether inference uses the local GPU.
    """

    name: str = ""
    provider: str = ""
    produces_analysis: bool = True
    parallel_mode: Literal["threads", "batch"] = "threads"
    runs_on_local_gpu: bool = False

    @abstractmethod
    def load(self) -> None:
        """Prepare the backend (create clients, load weights)."""

    @abstractmethod
    def is_available(self) -> tuple[bool, str]:
        """Check whether the backend can run right now.

        Returns:
            ``(ok, reason)``; ``reason`` is shown in the widget.
        """

    @abstractmethod
    def translate_batch(self, items: list[TranslationRequest]) -> list[TranslationResult]:
        """Translate a batch of records.

        Implementations must return one result per item, in order, and must
        not raise for per-record problems (set ``error`` instead).

        Args:
            items: Requests to translate.

        Returns:
            Results aligned with ``items``.
        """

    @abstractmethod
    def unload(self) -> None:
        """Release resources (VRAM, clients) between models."""

    def max_workers(self) -> int:
        """Preferred number of concurrent workers for ``threads`` mode.

        Returns:
            Worker count (1 by default).
        """
        return 1
