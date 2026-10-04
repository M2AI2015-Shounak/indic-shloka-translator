"""Prompt templates loaded from ``config/prompts`` and rendered safely."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

JSON_REMINDER = (
    "Your previous answer was not valid JSON for the required schema. "
    "Return ONLY one valid JSON object matching the schema — no markdown, no prose."
)
NO_THINK_SUFFIX = "\n/no_think"


def render(template: str, **values: str) -> str:
    """Fill ``{placeholders}`` without interpreting other braces.

    ``str.format`` would choke on JSON examples or braces inside a shloka, so
    only the named placeholders are replaced.

    Args:
        template: Template text.
        **values: Placeholder values.

    Returns:
        Rendered text.

    Example:
        >>> render("A {x} {\\"k\\": 1}", x="B")
        'A B {"k": 1}'
    """
    out = template
    for key, value in values.items():
        out = out.replace("{" + key + "}", "" if value is None else str(value))
    return out


@dataclass
class PromptSet:
    """The three prompt templates.

    Attributes:
        system: System prompt for translation.
        user: User prompt template with ``{detected_language}``, ``{source}``, ``{verse}``, ``{shloka}``.
        analysis: Analysis-only template (adds ``{translation}``).
        version: ``runtime.prompt_version`` (part of the cache key).
    """

    system: str
    user: str
    analysis: str
    version: str = "v1"

    @classmethod
    def load(cls, folder: Path, version: str = "v1") -> PromptSet:
        """Read prompt files from a folder.

        Args:
            folder: ``config/prompts`` directory.
            version: Prompt version label.

        Returns:
            Loaded :class:`PromptSet`.

        Raises:
            FileNotFoundError: If a prompt file is missing.
        """

        def read(name: str) -> str:
            """Read one prompt file.

            Args:
                name: File name.

            Returns:
                File content.
            """
            path = folder / name
            if not path.is_file():
                raise FileNotFoundError(f"Prompt file missing: {path}")
            return path.read_text(encoding="utf-8").strip()

        return cls(read("translate_system.txt"), read("translate_user.txt"), read("analysis_only.txt"), version)

    def translation_messages(
        self, *, language: str, source: str, verse: str, shloka: str, no_think: bool = False
    ) -> list[dict[str, str]]:
        """Build chat messages for a translation request.

        Args:
            language: Detected language name.
            source: ``Source`` field.
            verse: ``Verse`` field.
            shloka: Text to translate.
            no_think: Append ``/no_think`` (Qwen3 soft switch).

        Returns:
            ``[{"role": "system", ...}, {"role": "user", ...}]``.
        """
        user = render(self.user, detected_language=language, source=source, verse=verse, shloka=shloka)
        if no_think:
            user += NO_THINK_SUFFIX
        return [{"role": "system", "content": self.system}, {"role": "user", "content": user}]

    def analysis_messages(
        self, *, language: str, source: str, verse: str, shloka: str, translation: str, no_think: bool = False
    ) -> list[dict[str, str]]:
        """Build chat messages for an analysis-only request.

        Args:
            language: Detected language name.
            source: ``Source`` field.
            verse: ``Verse`` field.
            shloka: Original text.
            translation: Translation already produced by the selected model.
            no_think: Append ``/no_think``.

        Returns:
            Single-user-message list.
        """
        content = render(
            self.analysis,
            detected_language=language,
            source=source,
            verse=verse,
            shloka=shloka,
            translation=translation,
        )
        if no_think:
            content += NO_THINK_SUFFIX
        return [{"role": "user", "content": content}]

    @staticmethod
    def tiebreak_messages(text: str, candidates: list[str], hint: str) -> list[dict[str, str]]:
        """Build messages for the language-ID tiebreaker.

        Args:
            text: Text to classify.
            candidates: Allowed answers (supported names + ``OTHER``).
            hint: Script/current-guess hint.

        Returns:
            Chat messages.
        """
        system = (
            "You identify the LANGUAGE (not the script) of Indian texts. Sanskrit is often written in "
            "Bengali, Telugu, Kannada, Malayalam, Odia, Gujarati, Tamil/Grantha or Latin (IAST/ITRANS) "
            "script — judge by vocabulary and grammar (sandhi, case endings, visarga), not by script. "
            f"Answer with one of: {', '.join(candidates)}. Use OTHER for any language not listed "
            "(e.g. Urdu, English) and put its ISO 639-3 code in iso_code. Return only JSON: "
            '{"language": string, "iso_code": string, "confidence": number 0-1, "reason": string (≤20 words)}'
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": f"[{hint}]\n\n{text}"}]
