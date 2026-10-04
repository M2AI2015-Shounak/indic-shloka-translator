"""Supported-language table (21 Indian languages) and code helpers.

The authoritative list lives in ``config.yaml`` under
``language_detection.supported_languages``; :data:`DEFAULT_SUPPORTED` mirrors
it so the package still works if that key is omitted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_SUPPORTED: dict[str, list[str]] = {
    "Assamese": ["asm_Beng"],
    "Bengali": ["ben_Beng"],
    "Bodo": ["brx_Deva"],
    "Dogri": ["doi_Deva"],
    "Gujarati": ["guj_Gujr"],
    "Hindi": ["hin_Deva"],
    "Kannada": ["kan_Knda"],
    "Kashmiri": ["kas_Arab", "kas_Deva"],
    "Konkani": ["gom_Deva"],
    "Maithili": ["mai_Deva"],
    "Malayalam": ["mal_Mlym"],
    "Manipuri": ["mni_Beng", "mni_Mtei"],
    "Marathi": ["mar_Deva"],
    "Nepali": ["npi_Deva"],
    "Odia": ["ory_Orya"],
    "Punjabi": ["pan_Guru"],
    "Sanskrit": ["san_Deva"],
    "Santali": ["sat_Olck"],
    "Sindhi": ["snd_Arab", "snd_Deva"],
    "Tamil": ["tam_Taml"],
    "Telugu": ["tel_Telu"],
}

# ISO 639-3 (or FLORES) language prefix → display name, for supported and
# common unsupported languages (used to build readable Skipped remarks).
ISO_TO_NAME: dict[str, str] = {
    "asm": "Assamese",
    "ben": "Bengali",
    "brx": "Bodo",
    "doi": "Dogri",
    "guj": "Gujarati",
    "hin": "Hindi",
    "kan": "Kannada",
    "kas": "Kashmiri",
    "gom": "Konkani",
    "kok": "Konkani",
    "mai": "Maithili",
    "mal": "Malayalam",
    "mni": "Manipuri",
    "mar": "Marathi",
    "npi": "Nepali",
    "nep": "Nepali",
    "ory": "Odia",
    "ori": "Odia",
    "pan": "Punjabi",
    "san": "Sanskrit",
    "sat": "Santali",
    "snd": "Sindhi",
    "tam": "Tamil",
    "tel": "Telugu",
    "urd": "Urdu",
    "eng": "English",
    "fra": "French",
    "zho": "Chinese",
    "cmn": "Chinese",
    "yue": "Chinese",
    "jpn": "Japanese",
    "kor": "Korean",
    "rus": "Russian",
    "deu": "German",
    "spa": "Spanish",
    "ara": "Arabic",
    "pes": "Persian",
    "fas": "Persian",
    "arb": "Arabic",
    "ita": "Italian",
    "por": "Portuguese",
    "sin": "Sinhala",
    "tha": "Thai",
    "mya": "Burmese",
    "bod": "Tibetan",
    "pbt": "Pashto",
    "awa": "Awadhi",
    "bho": "Bhojpuri",
    "mag": "Magahi",
    "hne": "Chhattisgarhi",
}

# Native script for each FLORES script tag (used when a code needs a script).
SCRIPT_TAG_TO_SCRIPT: dict[str, str] = {
    "Deva": "Devanagari",
    "Beng": "Bengali",
    "Gujr": "Gujarati",
    "Guru": "Gurmukhi",
    "Knda": "Kannada",
    "Mlym": "Malayalam",
    "Orya": "Odia",
    "Taml": "Tamil",
    "Telu": "Telugu",
    "Olck": "Ol Chiki",
    "Mtei": "Meetei Mayek",
    "Arab": "Arabic",
    "Latn": "Latin",
    "Hans": "Han",
    "Hant": "Han",
    "Cyrl": "Cyrillic",
}
SCRIPT_TO_TAG: dict[str, str] = {
    "Devanagari": "Deva",
    "Bengali": "Beng",
    "Gujarati": "Gujr",
    "Gurmukhi": "Guru",
    "Kannada": "Knda",
    "Malayalam": "Mlym",
    "Odia": "Orya",
    "Tamil": "Taml",
    "Telugu": "Telu",
    "Ol Chiki": "Olck",
    "Meetei Mayek": "Mtei",
    "Arabic": "Arab",
    "Latin": "Latn",
}


@dataclass(frozen=True)
class LanguageTable:
    """Lookup helper around the supported-language mapping.

    Attributes:
        supported: Mapping display name → list of internal codes.
    """

    supported: dict[str, list[str]]

    @classmethod
    def from_config(cls, mapping: dict[str, list[str]] | None) -> LanguageTable:
        """Build the table from config (falls back to :data:`DEFAULT_SUPPORTED`).

        Args:
            mapping: ``language_detection.supported_languages`` from config.

        Returns:
            A :class:`LanguageTable`.
        """
        return cls(dict(mapping) if mapping else dict(DEFAULT_SUPPORTED))

    @property
    def names(self) -> list[str]:
        """Supported language names.

        Returns:
            List of display names.
        """
        return list(self.supported)

    def is_supported_name(self, name: str) -> bool:
        """Whether a display name is in the supported list.

        Args:
            name: Display name, e.g. ``"Sanskrit"``.

        Returns:
            ``True`` if supported.
        """
        return name in self.supported

    def name_for_code(self, code: str) -> str:
        """Map an internal code (``san_Deva``) or ISO prefix to a display name.

        Args:
            code: Code such as ``"hin_Deva"`` or ``"eng"``.

        Returns:
            Display name, or the code itself if unknown.

        Example:
            >>> LanguageTable.from_config(None).name_for_code("san_Deva")
            'Sanskrit'
        """
        for name, codes in self.supported.items():
            if code in codes:
                return name
        prefix = code.split("_", maxsplit=1)[0].lower()
        return ISO_TO_NAME.get(prefix, code)

    def code_for(self, name: str, script: str | None = None) -> str:
        """Pick the internal code for a language, preferring the given script.

        For Sanskrit written in a non-Devanagari script the result is still
        ``san_Deva`` (translators receive transliterated/Devanagari input or
        handle the script themselves); the real script is recorded separately.

        Args:
            name: Display name.
            script: Script name (e.g. ``"Arabic"``) to prefer when several codes exist.

        Returns:
            Internal code, or ``""`` when the language is unsupported.
        """
        codes = self.supported.get(name, [])
        if not codes:
            return ""
        if script:
            tag = SCRIPT_TO_TAG.get(script)
            for c in codes:
                if tag and c.endswith("_" + tag):
                    return c
        return codes[0]

    def all_codes(self) -> set[str]:
        """All supported internal codes.

        Returns:
            Set of codes such as ``{"san_Deva", "hin_Deva", ...}``.
        """
        return {c for codes in self.supported.values() for c in codes}
