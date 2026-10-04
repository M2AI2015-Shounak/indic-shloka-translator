"""Resumable JSONL checkpoint cache (survives Colab disconnects).

Each finished record is appended to
``Output_Folder/.cache/<model>/<prompt_version>.jsonl`` keyed by
``sha256(model + prompt_version + normalised shloka)``. On a re-run with
``resume_from_checkpoint: true`` cached records are reused and not re-billed.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from shloka_translator.output.naming import sanitize_model_name
from shloka_translator.utils.text import text_hash

logger = logging.getLogger(__name__)


def checkpoint_key(model: str, prompt_version: str, shloka: str) -> str:
    """Cache key for one record.

    Args:
        model: Registry model name.
        prompt_version: ``runtime.prompt_version``.
        shloka: Shloka text (normalised inside).

    Returns:
        SHA-256 hex digest.

    Example:
        >>> checkpoint_key("gpt-4.1", "v1", "राम") == checkpoint_key("gpt-4.1", "v1", " राम ")
        True
    """
    return text_hash(model, prompt_version, shloka)


class CheckpointStore:
    """Append-only JSONL store of finished records for one model + prompt version."""

    def __init__(self, cache_root: Path, model: str, prompt_version: str) -> None:
        """Open (or prepare) the store.

        Args:
            cache_root: ``paths.cache_folder``.
            model: Registry model name.
            prompt_version: Prompt version label.
        """
        self.model = model
        self.prompt_version = prompt_version
        self.path = cache_root / sanitize_model_name(model) / f"{sanitize_model_name(prompt_version)}.jsonl"
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = {}

    def load(self) -> int:
        """Read existing entries (later lines win; corrupt lines are skipped).

        Returns:
            Number of usable cached records.
        """
        self._entries = {}
        if not self.path.is_file():
            return 0
        with self.path.open(encoding="utf-8") as fh:
            for line_no, raw_line in enumerate(fh, 1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    self._entries[entry["key"]] = entry
                except (json.JSONDecodeError, KeyError):
                    logger.warning("Skipping corrupt checkpoint line %d in %s", line_no, self.path.name)
        logger.info("Checkpoint %s: %d cached record(s)", self.path, len(self._entries))
        return len(self._entries)

    def get(self, key: str) -> dict[str, Any] | None:
        """Look up a successful cached result.

        Only ``Success`` entries are reused; failed ones are retried.

        Args:
            key: :func:`checkpoint_key` value.

        Returns:
            Cached payload dict or ``None``.
        """
        entry = self._entries.get(key)
        if entry and entry.get("status") == "Success":
            return entry.get("result")
        return None

    def append(self, key: str, record_id: str, status: str, result: dict[str, Any]) -> None:
        """Append one finished record (flushed immediately).

        Args:
            key: Cache key.
            record_id: Input ``ID`` (for humans reading the file).
            status: ``Success`` / ``Failed``.
            result: Serialised result payload.
        """
        entry = {
            "key": key,
            "id": record_id,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "status": status,
            "result": result,
        }
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
            self._entries[key] = entry

    def __len__(self) -> int:
        """Number of loaded entries.

        Returns:
            Count.
        """
        return len(self._entries)
