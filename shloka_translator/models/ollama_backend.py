"""Ollama backend — models downloaded and run *locally inside the Colab VM*.

Ollama is not a hosted API: ``scripts/setup_ollama.sh`` installs the Ollama
server in the Colab runtime, starts it on ``http://localhost:11434`` and pulls
the selected model. This module talks to that local server through the
official ``ollama`` Python client.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from shloka_translator.config import AppConfig, ModelSpec
from shloka_translator.models.base import TranslationRequest, TranslationResult, TranslatorBackend, WordMeaning
from shloka_translator.pipeline.executor import RetryPolicy, call_with_retries
from shloka_translator.prompting.schemas import (
    TRANSLATION_JSON_SCHEMA,
    StructuredOutputError,
    TranslationOutput,
    parse_structured,
)
from shloka_translator.prompting.templates import JSON_REMINDER, PromptSet
from shloka_translator.utils.text import strip_think_tags, truncate

logger = logging.getLogger(__name__)


def ollama_server_ok(host: str) -> tuple[bool, str]:
    """Check whether the local Ollama server answers.

    Args:
        host: Server URL, e.g. ``http://localhost:11434``.

    Returns:
        ``(ok, reason)``.
    """
    try:
        import ollama
    except ImportError:
        return False, "ollama Python package not installed — re-run the Install cell (pip install ollama)"
    try:
        ollama.Client(host=host, timeout=5).list()
        return True, "server running"
    except Exception as exc:  # noqa: BLE001
        return False, (
            f"Ollama server not running ({type(exc).__name__}) — click 'Set up Ollama for selection' "
            "or Run (installs Ollama in this VM and downloads the model)"
        )


def pulled_models(host: str) -> set[str]:
    """List model tags already pulled into the local Ollama store.

    Args:
        host: Server URL.

    Returns:
        Set of tags such as ``{"qwen3:32b"}`` (empty if the server is down).
    """
    try:
        import ollama

        listing = ollama.Client(host=host, timeout=10).list()
    except Exception:  # noqa: BLE001
        return set()
    models = getattr(listing, "models", None)
    if models is None and isinstance(listing, dict):
        models = listing.get("models", [])
    names: set[str] = set()
    for m in models or []:
        name = getattr(m, "model", None) or (m.get("model") or m.get("name") if isinstance(m, dict) else None)
        if name:
            names.add(str(name))
            if str(name).endswith(":latest"):
                names.add(str(name)[: -len(":latest")])
    return names


def free_vram_gb() -> tuple[float | None, float | None, str]:
    """Return free and total VRAM of GPU 0 via NVML.

    Returns:
        ``(free_gb, total_gb, gpu_name)``; ``(None, None, "")`` without a GPU.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        name = name.decode() if isinstance(name, bytes) else str(name)
        return mem.free / 1024**3, mem.total / 1024**3, name
    except Exception:  # noqa: BLE001
        return None, None, ""


def vram_preflight(spec: ModelSpec, headroom_gb: float) -> tuple[bool, str]:
    """Compare a model's approximate VRAM need with free VRAM.

    Args:
        spec: Registry entry (``approx_vram_gb``).
        headroom_gb: Extra GB for the KV cache / context.

    Returns:
        ``(fits, message)``. ``fits`` is ``False`` when layers would spill to CPU.
    """
    need = (spec.approx_vram_gb or 0) + headroom_gb
    free, total, name = free_vram_gb()
    if free is None:
        return False, "No GPU detected — the model would run on CPU (extremely slow)"
    if need <= free:
        return True, f"OK: needs ~{need:.0f} GB, {free:.1f} GB free on {name}"
    return False, (
        f"{name} has {free:.1f} GB free of {total:.0f} GB but this model needs ~{need:.0f} GB: layers will "
        "spill to CPU and inference will be very slow. An L4 (24 GB) or A100 (40/80 GB) runtime is recommended."
    )


def run_setup_script(repo_root: Path, models: list[str], cfg: AppConfig) -> int:
    """Run ``scripts/setup_ollama.sh`` for the given models.

    Args:
        repo_root: Repository root.
        models: Model tags to pull.
        cfg: Config (concurrency → ``OLLAMA_NUM_PARALLEL``).

    Returns:
        Script exit code.
    """
    from shloka_translator.utils.packages import ensure_packages

    missing = ensure_packages(["ollama"])
    if missing:
        print(f"✗ Could not install the ollama Python client: {missing['ollama']}")
        return 1
    script = repo_root / "scripts" / "setup_ollama.sh"
    env = dict(os.environ)
    env["OLLAMA_NUM_PARALLEL"] = str(cfg.concurrency.ollama_max_workers)
    env["OLLAMA_KEEP_ALIVE"] = cfg.ollama.keep_alive
    env["OLLAMA_MAX_LOADED_MODELS"] = "1"
    proc = subprocess.run(["bash", str(script), *models], env=env, check=False)
    return proc.returncode


class OllamaBackend(TranslatorBackend):
    """Translator backed by a local Ollama model."""

    provider = "ollama"
    produces_analysis = True
    parallel_mode = "threads"
    runs_on_local_gpu = True

    def __init__(
        self, name: str, spec: ModelSpec, cfg: AppConfig, prompts: PromptSet, client: Any | None = None
    ) -> None:
        """Create the backend.

        Args:
            name: Ollama tag, e.g. ``"qwen3:32b"``.
            spec: Registry entry.
            cfg: Config.
            prompts: Prompt templates.
            client: Optional pre-built client (tests).
        """
        self.name = name
        self.spec = spec
        self.cfg = cfg
        self.prompts = prompts
        self._client = client
        self._think_supported = True
        self.policy = RetryPolicy(max_retries=cfg.concurrency.max_retries, base_delay=2.0)

    def load(self) -> None:
        """Connect to the local server and make sure the model is pulled.

        Raises:
            RuntimeError: If the server is not running or the pull fails.
        """
        if self._client is None:
            try:
                import ollama
            except ImportError as exc:
                raise RuntimeError("ollama package not installed") from exc
            ok, reason = ollama_server_ok(self.cfg.ollama.host)
            if not ok:
                raise RuntimeError(reason)
            self._client = ollama.Client(host=self.cfg.ollama.host, timeout=self.cfg.concurrency.request_timeout_s)
            if self.name not in pulled_models(self.cfg.ollama.host):
                logger.info("Pulling %s (first time can take a while)...", self.name)
                self._client.pull(self.name)

    def is_available(self) -> tuple[bool, str]:
        """Check server status and whether the model is pulled.

        Returns:
            ``(ok, reason)``.
        """
        ok, reason = ollama_server_ok(self.cfg.ollama.host)
        if not ok:
            return False, reason
        if self.name not in pulled_models(self.cfg.ollama.host):
            return False, "not downloaded yet (Run / Set up Ollama downloads it)"
        return True, "ready"

    def unload(self) -> None:
        """Ask Ollama to evict the model from VRAM (``keep_alive=0``)."""
        if self._client is None:
            return
        try:
            self._client.generate(model=self.name, prompt="", keep_alive=0)
            logger.info("Unloaded %s from VRAM", self.name)
        except Exception:
            logger.debug("unload failed", exc_info=True)

    def max_workers(self) -> int:
        """Concurrent requests (matches ``OLLAMA_NUM_PARALLEL``).

        Returns:
            ``concurrency.ollama_max_workers``.
        """
        return max(1, self.cfg.concurrency.ollama_max_workers)

    def _chat(self, messages: list[dict[str, str]]) -> Any:
        """Call ``chat`` with JSON-schema format, retries and thinking disabled.

        Args:
            messages: Chat messages.

        Returns:
            ``(response, retries)`` from :func:`call_with_retries`.
        """
        gen = self.cfg.generation
        options = {
            "temperature": gen.temperature,
            "seed": gen.seed,
            "num_ctx": gen.ollama_num_ctx,
            "num_predict": gen.max_output_tokens,
        }

        def _do() -> Any:
            """Single chat request.

            Returns:
                Ollama response.
            """
            kwargs: dict[str, Any] = {
                "model": self.name,
                "messages": messages,
                "format": TRANSLATION_JSON_SCHEMA,
                "options": options,
                "keep_alive": self.cfg.ollama.keep_alive,
            }
            if self.spec.disable_thinking and self._think_supported:
                kwargs["think"] = False
            try:
                return self._client.chat(**kwargs)  # type: ignore[union-attr]
            except TypeError:
                self._think_supported = False
                kwargs.pop("think", None)
                return self._client.chat(**kwargs)  # type: ignore[union-attr]

        def retryable(exc: BaseException) -> bool:
            """Treat connection errors / 5xx / timeouts as transient.

            Args:
                exc: Exception.

            Returns:
                ``True`` if transient.
            """
            code = getattr(exc, "status_code", None)
            return (
                isinstance(exc, (ConnectionError, TimeoutError))
                or (isinstance(code, int) and code >= 500)
                or "timed out" in str(exc).lower()
            )

        return call_with_retries(_do, is_retryable=retryable, policy=self.policy)

    @staticmethod
    def _field(resp: Any, key: str) -> Any:
        """Read a field from an Ollama response (object or dict).

        Args:
            resp: Response.
            key: Field name.

        Returns:
            Value or ``None``.
        """
        if isinstance(resp, dict):
            return resp.get(key)
        return getattr(resp, key, None)

    def translate_one(self, req: TranslationRequest) -> TranslationResult:
        """Translate one record (never raises).

        Args:
            req: Request.

        Returns:
            :class:`TranslationResult`.
        """
        start = time.perf_counter()
        result = TranslationResult()
        no_think = self.spec.disable_thinking
        messages = self.prompts.translation_messages(
            language=req.language_name, source=req.source, verse=req.verse, shloka=req.text, no_think=no_think
        )
        try:
            if self._client is None:
                self.load()
            raw = self._call_and_count(messages, result)
            try:
                parsed = parse_structured(raw, TranslationOutput)
            except StructuredOutputError:
                retry_msgs = [
                    *messages,
                    {"role": "assistant", "content": raw[:4000]},
                    {"role": "user", "content": JSON_REMINDER},
                ]
                raw = self._call_and_count(retry_msgs, result)
                parsed = parse_structured(raw, TranslationOutput)
            if not isinstance(parsed, TranslationOutput):
                raise StructuredOutputError("unexpected parse type")
            result.translation = parsed.translation
            result.segmentation = [s for s in parsed.word_segmentation if s.strip()]
            result.word_meanings = [WordMeaning(w.word, w.meaning) for w in parsed.word_meanings]
            result.translator_notes = parsed.translator_notes.strip()
            result.has_analysis = bool(result.segmentation or result.word_meanings)
            result.raw_response = raw
        except Exception as exc:  # noqa: BLE001
            result.error = f"{type(exc).__name__}: {truncate(str(exc), 300)}"
        result.latency_s = time.perf_counter() - start
        result.total_tokens = result.input_tokens + result.output_tokens
        return result

    def _call_and_count(self, messages: list[dict[str, str]], acc: TranslationResult) -> str:
        """Run one chat call and add its token counts to ``acc``.

        Args:
            messages: Chat messages.
            acc: Result accumulating tokens.

        Returns:
            Model text with ``<think>`` blocks removed.
        """
        resp, retries = self._chat(messages)
        acc.retries += retries
        acc.input_tokens += int(self._field(resp, "prompt_eval_count") or 0)
        acc.output_tokens += int(self._field(resp, "eval_count") or 0)
        eval_ns = self._field(resp, "eval_duration") or 0
        if eval_ns and self._field(resp, "eval_count"):
            tps = float(self._field(resp, "eval_count")) / (float(eval_ns) / 1e9)
            logger.debug("%s: %.1f tokens/s", self.name, tps)
        message = self._field(resp, "message")
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
        return strip_think_tags(content or "")

    def structured_raw(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> tuple[str, int, int]:
        """Generic structured call used by the analysis helper / LID tiebreaker.

        Args:
            messages: Chat messages.
            schema: JSON schema for ``format``.

        Returns:
            ``(text, input_tokens, output_tokens)``.
        """
        if self._client is None:
            self.load()
        gen = self.cfg.generation
        kwargs: dict[str, Any] = {
            "model": self.name,
            "messages": messages,
            "format": schema,
            "options": {
                "temperature": gen.temperature,
                "seed": gen.seed,
                "num_ctx": gen.ollama_num_ctx,
                "num_predict": gen.max_output_tokens,
            },
        }
        if self.spec.disable_thinking:
            kwargs["think"] = False
        try:
            resp = self._client.chat(**kwargs)  # type: ignore[union-attr]
        except TypeError:
            kwargs.pop("think", None)
            resp = self._client.chat(**kwargs)  # type: ignore[union-attr]
        message = self._field(resp, "message")
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
        return (
            strip_think_tags(content or ""),
            int(self._field(resp, "prompt_eval_count") or 0),
            int(self._field(resp, "eval_count") or 0),
        )

    def translate_batch(self, items: list[TranslationRequest]) -> list[TranslationResult]:
        """Translate items sequentially (executor provides concurrency).

        Args:
            items: Requests.

        Returns:
            Results aligned with ``items``.
        """
        return [self.translate_one(i) for i in items]
