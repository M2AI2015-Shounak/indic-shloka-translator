"""Pydantic v2 models for structured LLM output and helpers to parse it."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from shloka_translator.utils.text import strip_think_tags

logger = logging.getLogger(__name__)


class WordMeaningModel(BaseModel):
    """One segment and its contextual meaning."""

    model_config = ConfigDict(extra="ignore")
    word: str
    meaning: str


class TranslationOutput(BaseModel):
    """Structured translation returned by LLM backends."""

    model_config = ConfigDict(extra="ignore")
    translation: str = Field(min_length=1)
    word_segmentation: list[str] = Field(default_factory=list)
    word_meanings: list[WordMeaningModel] = Field(default_factory=list)
    translator_notes: str = ""

    @field_validator("translation")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        """Reject whitespace-only translations.

        Args:
            value: Candidate translation.

        Returns:
            Stripped translation.

        Raises:
            ValueError: If the translation is blank.
        """
        if not value.strip():
            raise ValueError("translation is empty")
        return value.strip()


class AnalysisOutput(BaseModel):
    """Segmentation + meanings returned by the analysis helper."""

    model_config = ConfigDict(extra="ignore")
    word_segmentation: list[str] = Field(default_factory=list)
    word_meanings: list[WordMeaningModel] = Field(default_factory=list)


class LanguageTiebreak(BaseModel):
    """LLM tiebreaker answer for language identification."""

    model_config = ConfigDict(extra="ignore")
    language: str
    iso_code: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    reason: str = ""


def _strict(schema: dict[str, Any]) -> dict[str, Any]:
    """Return an OpenAI-strict copy of a JSON schema (all keys required, no extras).

    Args:
        schema: Plain JSON schema.

    Returns:
        Schema with ``additionalProperties: false`` and ``required`` set at every object level.
    """
    out = json.loads(json.dumps(schema))

    def walk(node: Any) -> None:
        """Recursively enforce strictness.

        Args:
            node: Schema node.
        """
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"].keys())
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(out)
    return out


_WORD_MEANING_SCHEMA = {
    "type": "object",
    "properties": {"word": {"type": "string"}, "meaning": {"type": "string"}},
}
TRANSLATION_JSON_SCHEMA: dict[str, Any] = _strict(
    {
        "type": "object",
        "properties": {
            "translation": {"type": "string"},
            "word_segmentation": {"type": "array", "items": {"type": "string"}},
            "word_meanings": {"type": "array", "items": _WORD_MEANING_SCHEMA},
            "translator_notes": {"type": "string"},
        },
    }
)
ANALYSIS_JSON_SCHEMA: dict[str, Any] = _strict(
    {
        "type": "object",
        "properties": {
            "word_segmentation": {"type": "array", "items": {"type": "string"}},
            "word_meanings": {"type": "array", "items": _WORD_MEANING_SCHEMA},
        },
    }
)
TIEBREAK_JSON_SCHEMA: dict[str, Any] = _strict(
    {
        "type": "object",
        "properties": {
            "language": {"type": "string"},
            "iso_code": {"type": "string"},
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        },
    }
)


class StructuredOutputError(ValueError):
    """Raised when a model response cannot be parsed into the expected schema."""


def extract_json_text(raw: str) -> str:
    """Pull the JSON object out of a model response.

    Handles ``<think>`` blocks, Markdown code fences and leading/trailing prose.

    Args:
        raw: Raw model output.

    Returns:
        The substring most likely to be the JSON object.

    Example:
        >>> extract_json_text('```json\\n{"a": 1}\\n```')
        '{"a": 1}'
    """
    text = strip_think_tags(raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


def parse_structured(raw: str, model_cls: type[BaseModel]) -> BaseModel:
    """Parse and validate a model response into ``model_cls``.

    Args:
        raw: Raw model output.
        model_cls: Pydantic model to validate against.

    Returns:
        Validated model instance.

    Raises:
        StructuredOutputError: If the text is not valid JSON or fails validation.
    """
    candidate = extract_json_text(raw)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(f"Invalid JSON from model: {exc}") from exc
    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        raise StructuredOutputError(f"JSON did not match schema: {exc.errors()[:3]}") from exc
