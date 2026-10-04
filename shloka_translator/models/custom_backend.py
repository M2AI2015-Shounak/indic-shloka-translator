"""Custom fine-tuned IndicTrans2 model (``custom-shloka-mt``)."""

from __future__ import annotations

import logging
from pathlib import Path

from shloka_translator.config import AppConfig
from shloka_translator.models.indictrans2_backend import IndicTrans2Backend

logger = logging.getLogger(__name__)

NOT_TRAINED = "Custom model not trained yet — run notebooks/02_train_custom_model.ipynb"


def resolve_checkpoint(custom_dir: Path, checkpoint: str | None) -> Path | None:
    """Find the checkpoint directory to load.

    A valid checkpoint directory contains ``config.json``. ``"latest"`` (or
    ``None``) picks the newest ``custom-shloka-mt_*`` folder by name (the names
    embed a sortable timestamp), falling back to modification time.

    Args:
        custom_dir: ``models/custom`` directory.
        checkpoint: ``"latest"``, a folder name, or an absolute path.

    Returns:
        Path of the checkpoint, or ``None`` if none exists.

    Example:
        >>> resolve_checkpoint(Path("/nonexistent"), "latest") is None
        True
    """
    if checkpoint and checkpoint != "latest":
        p = Path(checkpoint)
        p = p if p.is_absolute() else custom_dir / p
        return p if (p / "config.json").is_file() else None
    if not custom_dir.is_dir():
        return None
    candidates = [d for d in custom_dir.iterdir() if d.is_dir() and (d / "config.json").is_file()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda d: (d.name, d.stat().st_mtime))[-1]


class CustomBackend(IndicTrans2Backend):
    """Loads the newest (or named) fine-tuned checkpoint from ``models/custom``."""

    provider = "custom"

    def __init__(
        self, name: str, cfg: AppConfig, checkpoint: str | None = "latest", hf_token: str | None = None
    ) -> None:
        """Create the backend.

        Args:
            name: Registry name (``custom-shloka-mt``).
            cfg: Config.
            checkpoint: ``"latest"`` or a specific folder.
            hf_token: HF token (not needed for local checkpoints).
        """
        self.checkpoint_dir = resolve_checkpoint(cfg.path(cfg.paths.custom_model_dir), checkpoint)
        super().__init__(name, str(self.checkpoint_dir or ""), cfg, hf_token=hf_token, gated=False)

    def is_available(self) -> tuple[bool, str]:
        """Report whether a trained checkpoint exists (plus the usual checks).

        Returns:
            ``(ok, reason)``.
        """
        if self.checkpoint_dir is None:
            return False, NOT_TRAINED
        ok, reason = super().is_available()
        return ok, (f"{reason} — {self.checkpoint_dir.name}" if ok else reason)

    def load(self) -> None:
        """Load the checkpoint.

        Raises:
            RuntimeError: If no checkpoint exists.
        """
        if self.checkpoint_dir is None:
            raise RuntimeError(NOT_TRAINED)
        super().load()
