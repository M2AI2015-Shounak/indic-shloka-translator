"""Analysis helper: word segmentation + word meanings for seq2seq models.

IndicTrans2 and the custom model only produce a translation. This helper asks
an LLM (``analysis.helper_model``, then ``helper_fallback_order``) for the
segmentation and meanings, or falls back to the rule-based path. The same
helper-LLM wrapper also powers the language-ID tiebreaker.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from shloka_translator.analysis.segmentation import rule_based_segmentation
from shloka_translator.config import AppConfig, ModelSpec
from shloka_translator.langid.detector import TiebreakOutcome
from shloka_translator.models.base import TranslationRequest, WordMeaning
from shloka_translator.prompting.schemas import (
    ANALYSIS_JSON_SCHEMA,
    TIEBREAK_JSON_SCHEMA,
    AnalysisOutput,
    LanguageTiebreak,
    StructuredOutputError,
    parse_structured,
)
from shloka_translator.prompting.templates import PromptSet
from shloka_translator.utils.text import truncate

logger = logging.getLogger(__name__)

RULE_BASED = "rule_based"


@dataclass
class AnalysisResult:
    """Segmentation/meanings produced by the helper.

    Attributes:
        segmentation: Segments.
        word_meanings: Meanings aligned with segments (may be shorter).
        method: Human-readable method, e.g. ``"helper gpt-4.1-mini"``.
        input_tokens: Helper input tokens.
        output_tokens: Helper output tokens.
        error: Why analysis is unavailable, if it failed.
        notes: Extra remarks.
    """

    segmentation: list[str] = field(default_factory=list)
    word_meanings: list[WordMeaning] = field(default_factory=list)
    method: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None
    notes: list[str] = field(default_factory=list)


class HelperLLM:
    """Uniform structured-call wrapper over an OpenAI or Ollama model."""

    def __init__(self, name: str, spec: ModelSpec, cfg: AppConfig, prompts: PromptSet, openai_key: str | None) -> None:
        """Create the wrapper (lazy).

        Args:
            name: Registry model name.
            spec: Registry entry.
            cfg: Config.
            prompts: Prompt templates.
            openai_key: OpenAI key.
        """
        self.name = name
        self.spec = spec
        self.cfg = cfg
        self.prompts = prompts
        self.openai_key = openai_key
        self._backend: Any = None

    def available(self) -> tuple[bool, str]:
        """Check availability.

        Returns:
            ``(ok, reason)``.
        """
        backend = self._get()
        return backend.is_available()

    def _get(self) -> Any:
        """Build the underlying backend once.

        Returns:
            OpenAI or Ollama backend.

        Raises:
            ValueError: For providers that cannot serve as helpers.
        """
        if self._backend is None:
            if self.spec.provider == "openai":
                from shloka_translator.models.openai_backend import OpenAIBackend

                self._backend = OpenAIBackend(self.name, self.cfg, self.prompts, self.openai_key)
            elif self.spec.provider == "ollama":
                from shloka_translator.models.ollama_backend import OllamaBackend

                self._backend = OllamaBackend(self.name, self.spec, self.cfg, self.prompts)
            else:
                raise ValueError(f"{self.name} ({self.spec.provider}) cannot be used as a helper LLM")
        return self._backend

    def call(self, messages: list[dict[str, str]], schema: dict[str, Any], schema_name: str) -> tuple[str, int, int]:
        """Make a structured call.

        Args:
            messages: Chat messages.
            schema: JSON schema.
            schema_name: Schema name (OpenAI).

        Returns:
            ``(text, input_tokens, output_tokens)``.
        """
        backend = self._get()
        if self.spec.provider == "openai":
            out = backend.structured_call(
                messages, schema, schema_name, max_tokens=self.cfg.generation.max_output_tokens
            )
            return out.text, out.input_tokens, out.output_tokens
        return backend.structured_raw(messages, schema)


def resolve_helpers(
    cfg: AppConfig, prompts: PromptSet, openai_key: str | None, order: list[str]
) -> list[HelperLLM | str]:
    """Turn a list of names into usable helpers (``"rule_based"`` kept as a string).

    Unknown or unavailable models are skipped with a log message.

    Args:
        cfg: Config.
        prompts: Prompt templates.
        openai_key: OpenAI key.
        order: Names in preference order.

    Returns:
        Ordered list of available helpers.
    """
    helpers: list[HelperLLM | str] = []
    seen: set[str] = set()
    for name in order:
        if name in seen:
            continue
        seen.add(name)
        if name == RULE_BASED:
            helpers.append(RULE_BASED)
            continue
        spec = cfg.models.get(name)
        if spec is None:
            spec = ModelSpec(provider="ollama") if ":" in name else ModelSpec(provider="openai")
        if spec.provider not in ("openai", "ollama"):
            continue
        helper = HelperLLM(name, spec, cfg, prompts, openai_key)
        try:
            ok, reason = helper.available()
        except Exception as exc:  # noqa: BLE001
            ok, reason = False, str(exc)
        if ok:
            helpers.append(helper)
        else:
            logger.info("Helper %s unavailable: %s", name, reason)
    return helpers


class AnalysisHelper:
    """Produces segmentation/meanings for translation-only backends."""

    def __init__(
        self,
        cfg: AppConfig,
        prompts: PromptSet,
        openai_key: str | None,
        word_translator: Callable[[list[str], str], list[str]] | None = None,
    ) -> None:
        """Resolve helpers in configured order.

        Args:
            cfg: Config.
            prompts: Prompt templates.
            openai_key: OpenAI key.
            word_translator: Optional ``(words, src_lang) -> glosses`` (IndicTrans2) for the rule-based path.
        """
        order = [cfg.analysis.helper_model, *cfg.analysis.helper_fallback_order]
        self.cfg = cfg
        self.prompts = prompts
        self.helpers = resolve_helpers(cfg, prompts, openai_key, order)
        if RULE_BASED not in self.helpers:
            self.helpers.append(RULE_BASED)
        self.word_translator = word_translator

    def describe(self) -> str:
        """Names of helpers in use.

        Returns:
            Comma-separated list.
        """
        return ", ".join(h if isinstance(h, str) else h.name for h in self.helpers)

    def analyse(self, req: TranslationRequest, translation: str) -> AnalysisResult:
        """Segment and gloss one record, trying helpers in order.

        Args:
            req: Original request.
            translation: Translation already produced.

        Returns:
            :class:`AnalysisResult` (``error`` set if every helper failed).
        """
        errors: list[str] = []
        total_in = total_out = 0
        for helper in self.helpers:
            if helper == RULE_BASED:
                res = self._rule_based(req)
            elif isinstance(helper, HelperLLM):
                res = self._llm(helper, req, translation)
            else:
                continue
            total_in += res.input_tokens
            total_out += res.output_tokens
            if res.error is None and res.segmentation:
                res.input_tokens, res.output_tokens = total_in, total_out
                if errors:
                    res.notes.append("Earlier helpers failed: " + "; ".join(errors))
                return res
            errors.append(f"{res.method}: {res.error or 'empty result'}")
        return AnalysisResult(
            method="none", input_tokens=total_in, output_tokens=total_out, error=truncate("; ".join(errors), 300)
        )

    def _llm(self, helper: HelperLLM, req: TranslationRequest, translation: str) -> AnalysisResult:
        """Ask an LLM helper for segmentation + meanings.

        Args:
            helper: Helper wrapper.
            req: Request.
            translation: Existing translation.

        Returns:
            :class:`AnalysisResult`.
        """
        method = f"helper {helper.name}"
        no_think = helper.spec.disable_thinking
        messages = self.prompts.analysis_messages(
            language=req.language_name,
            source=req.source,
            verse=req.verse,
            shloka=req.text,
            translation=translation,
            no_think=no_think,
        )
        try:
            text, tin, tout = helper.call(messages, ANALYSIS_JSON_SCHEMA, "shloka_analysis")
            parsed = parse_structured(text, AnalysisOutput)
            if not isinstance(parsed, AnalysisOutput):
                raise StructuredOutputError("unexpected parse type")
            return AnalysisResult(
                [s for s in parsed.word_segmentation if s.strip()],
                [WordMeaning(w.word, w.meaning) for w in parsed.word_meanings],
                method,
                tin,
                tout,
            )
        except Exception as exc:  # noqa: BLE001 - includes StructuredOutputError
            return AnalysisResult(method=method, error=f"{type(exc).__name__}: {truncate(str(exc), 200)}")

    def _rule_based(self, req: TranslationRequest) -> AnalysisResult:
        """Rule-based segmentation, plus IndicTrans2 word glosses when available.

        Args:
            req: Request.

        Returns:
            :class:`AnalysisResult`.
        """
        text = req.text
        if req.language_name == "Sanskrit" and req.script not in ("", "Devanagari"):
            from shloka_translator.models.indictrans2_backend import transliterate_to_devanagari

            with contextlib.suppress(ValueError):
                text, _ = transliterate_to_devanagari(text, req.script)
        segments, method = rule_based_segmentation(text, req.language_name, self.cfg.analysis.sanskrit_parser_timeout_s)
        result = AnalysisResult(segments, [], f"rule_based ({method})")
        if self.word_translator and segments:
            try:
                glosses = self.word_translator(segments, req.language_code or "san_Deva")
                result.word_meanings = [WordMeaning(w, g) for w, g in zip(segments, glosses, strict=False) if g]
                result.method += " + IndicTrans2 word glosses"
            except Exception as exc:  # noqa: BLE001
                result.notes.append(f"word meanings unavailable ({type(exc).__name__})")
        else:
            result.notes.append("word meanings unavailable (no helper LLM or IndicTrans2 for word glosses)")
        return result


def make_llm_tiebreaker(
    cfg: AppConfig, prompts: PromptSet, openai_key: str | None
) -> tuple[Callable[[str, list[str], str], TiebreakOutcome | None] | None, str]:
    """Build the language-ID tiebreaker callable from config.

    Uses ``language_detection.tiebreaker_model`` if available, else the first
    available Ollama model from the registry/helper order.

    Args:
        cfg: Config.
        prompts: Prompt templates.
        openai_key: OpenAI key.

    Returns:
        ``(callable_or_None, model_name)``.
    """
    if not cfg.language_detection.use_llm_tiebreaker:
        return None, ""
    order = (
        [cfg.language_detection.tiebreaker_model]
        + [n for n in cfg.analysis.helper_fallback_order if n != RULE_BASED]
        + [n for n, s in cfg.models.items() if s.provider == "ollama"]
    )
    helpers = [h for h in resolve_helpers(cfg, prompts, openai_key, order) if isinstance(h, HelperLLM)]
    if not helpers:
        logger.info("No LLM available for the language-ID tiebreaker; rules/classifiers only")
        return None, ""
    helper = helpers[0]

    def tiebreak(text: str, candidates: list[str], hint: str) -> TiebreakOutcome | None:
        """Ask the helper which language ``text`` is in.

        Args:
            text: Text.
            candidates: Allowed answers.
            hint: Script / current guess.

        Returns:
            :class:`TiebreakOutcome` or ``None`` on parse failure.
        """
        messages = PromptSet.tiebreak_messages(text, candidates, hint)
        raw, tin, tout = helper.call(messages, TIEBREAK_JSON_SCHEMA, "language_tiebreak")
        try:
            parsed = parse_structured(raw, LanguageTiebreak)
        except StructuredOutputError:
            return TiebreakOutcome("", "", 0.0, "unparseable", helper.name, tin, tout)
        if not isinstance(parsed, LanguageTiebreak):
            return None
        return TiebreakOutcome(
            parsed.language, parsed.iso_code, float(parsed.confidence), parsed.reason, helper.name, tin, tout
        )

    return tiebreak, helper.name
