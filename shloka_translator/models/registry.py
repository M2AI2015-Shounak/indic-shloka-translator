"""Build backends from registry names and report their availability."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from shloka_translator.config import AppConfig
from shloka_translator.models.base import TranslatorBackend
from shloka_translator.prompting.templates import PromptSet

logger = logging.getLogger(__name__)

ALL_MODELS = "ALL MODELS"
PROVIDER_LABELS = {
    "openai": "OpenAI (API)",
    "ollama": "Ollama (local in Colab)",
    "hf_seq2seq": "Hugging Face (local GPU)",
    "custom": "Custom fine-tuned (local GPU)",
}


@dataclass
class ModelStatus:
    """Availability of one registry model (shown in the dropdown).

    Attributes:
        name: Registry name.
        provider: Provider key.
        available: Whether it can run now.
        reason: Why not / ``"ready"``.
        setup_on_run: ``True`` when the missing piece is installed/downloaded automatically on Run.
    """

    name: str
    provider: str
    available: bool
    reason: str
    setup_on_run: bool = False

    @property
    def badge(self) -> str:
        """Dropdown label with a status badge.

        Returns:
            e.g. ``"✓ gpt-4.1"``, ``"⬇ qwen3:32b — installed & downloaded on Run"`` or
            ``"⚠ gpt-5 — not available for this key"``.
        """
        if self.available:
            return f"✓ {self.name}"
        if self.setup_on_run:
            return f"⬇ {self.name} — {self.reason}"
        return f"⚠ {self.name} — {self.reason}"


def auto_setup_reason(provider: str, reason: str) -> str | None:
    """Decide whether a not-ready model will be fixed automatically when Run is clicked.

    Ollama models need the Ollama server + model weights inside the Colab VM, and the
    ``ollama`` / ``IndicTransToolkit`` Python packages can be pip-installed on demand —
    all of that happens on Run. Missing keys, GPUs, licences or trained checkpoints cannot be.

    Args:
        provider: Provider key.
        reason: ``is_available`` reason.

    Returns:
        Replacement text for the badge, or ``None`` if the user must act.

    Example:
        >>> auto_setup_reason("ollama", "ollama Python package not installed")
        'not set up yet — Run installs Ollama in this VM and downloads the model'
    """
    low = reason.lower()
    if "gpu" in low:
        return None
    if provider == "ollama" and ("package" in low or "server" in low or "not downloaded" in low):
        return "not set up yet — Run installs Ollama in this VM and downloads the model"
    if provider in ("hf_seq2seq", "custom") and "indictranstoolkit" in low:
        return "IndicTransToolkit missing — Run installs it automatically"
    return None


def build_backend(
    name: str, cfg: AppConfig, prompts: PromptSet, *, openai_key: str | None = None, hf_token: str | None = None
) -> TranslatorBackend:
    """Instantiate the backend for a registry model name.

    Args:
        name: Key from ``config.models``.
        cfg: Config.
        prompts: Prompt templates.
        openai_key: OpenAI API key.
        hf_token: Hugging Face token.

    Returns:
        A (not yet loaded) :class:`TranslatorBackend`.

    Raises:
        KeyError: If ``name`` is not in the registry.
        ValueError: If the provider is unknown.
    """
    if name not in cfg.models:
        raise KeyError(f"Model {name!r} is not in config.models")
    spec = cfg.models[name]
    if spec.provider == "openai":
        from shloka_translator.models.openai_backend import OpenAIBackend

        return OpenAIBackend(name, cfg, prompts, openai_key)
    if spec.provider == "ollama":
        from shloka_translator.models.ollama_backend import OllamaBackend

        return OllamaBackend(name, spec, cfg, prompts)
    if spec.provider == "hf_seq2seq":
        from shloka_translator.models.indictrans2_backend import IndicTrans2Backend

        return IndicTrans2Backend(name, spec.hf_id or name, cfg, hf_token=hf_token, gated=spec.gated)
    if spec.provider == "custom":
        from shloka_translator.models.custom_backend import CustomBackend

        return CustomBackend(name, cfg, spec.checkpoint, hf_token=hf_token)
    raise ValueError(f"Unknown provider {spec.provider}")


def model_statuses(
    cfg: AppConfig, prompts: PromptSet, *, openai_key: str | None = None, hf_token: str | None = None
) -> list[ModelStatus]:
    """Check every registry model's availability (one ``models.list()`` for OpenAI).

    Args:
        cfg: Config.
        prompts: Prompt templates.
        openai_key: OpenAI API key.
        hf_token: HF token.

    Returns:
        Statuses in registry order.
    """
    statuses = []
    for name, spec in cfg.models.items():
        try:
            backend = build_backend(name, cfg, prompts, openai_key=openai_key, hf_token=hf_token)
            ok, reason = backend.is_available()
        except Exception as exc:  # noqa: BLE001
            ok, reason = False, f"{type(exc).__name__}: {exc}"
        auto = None if ok else auto_setup_reason(spec.provider, reason)
        statuses.append(ModelStatus(name, spec.provider, ok, auto or reason, setup_on_run=auto is not None))
    return statuses
