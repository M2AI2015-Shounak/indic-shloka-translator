"""OpenAI backend (official ``openai`` SDK v1+), with model-capability adaptation.

* Chat Completions by default; ``openai.use_responses_api: true`` switches to
  the Responses API.
* Structured Outputs (JSON schema, strict) validated with pydantic; one
  "return valid JSON only" retry, then ``Failed``.
* If the API rejects a parameter with HTTP 400 (e.g. ``temperature`` on
  GPT-5-family models, ``max_tokens`` vs ``max_completion_tokens``), the
  parameter is dropped/renamed, the call retried once, and the capability is
  cached per model for the rest of the run.
* 429 / 5xx / timeouts are retried with exponential jitter, honouring
  ``Retry-After``. Token counts come only from the provider's ``usage``.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from shloka_translator.config import AppConfig
from shloka_translator.models.base import TranslationRequest, TranslationResult, TranslatorBackend, WordMeaning
from shloka_translator.pipeline.executor import RateLimiter, RetryPolicy, call_with_retries
from shloka_translator.prompting.schemas import (
    TRANSLATION_JSON_SCHEMA,
    StructuredOutputError,
    TranslationOutput,
    parse_structured,
)
from shloka_translator.prompting.templates import JSON_REMINDER, PromptSet
from shloka_translator.secrets import redact
from shloka_translator.utils.text import truncate

logger = logging.getLogger(__name__)

MODEL_NOT_AVAILABLE = "Model not available to this API key"
_models_cache: dict[str, tuple[set[str] | None, str]] = {}
_models_lock = threading.Lock()


@dataclass
class ModelCapabilities:
    """Per-model parameter support learned from 400 responses during a run.

    Attributes:
        token_param: Name of the output-token limit parameter.
        dropped: Parameters the model rejected (never sent again).
        response_format: ``"json_schema"``, ``"json_object"`` or ``"none"``.
    """

    token_param: str = "max_completion_tokens"  # noqa: S105 - parameter name, not a secret
    dropped: set[str] = field(default_factory=set)
    response_format: str = "json_schema"


_capabilities: dict[str, ModelCapabilities] = {}
_cap_lock = threading.Lock()


@dataclass
class CallOutcome:
    """Raw outcome of one successful API call.

    Attributes:
        text: Model output text.
        input_tokens: Provider-reported prompt tokens.
        output_tokens: Provider-reported completion tokens.
        total_tokens: Provider-reported total.
        reasoning_tokens: Reasoning tokens (subset of output) if reported.
        finish_reason: Finish reason, e.g. ``"stop"`` / ``"length"``.
        retries: Transient-error retries used.
    """

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    finish_reason: str = ""
    retries: int = 0


def list_available_models(api_key: str | None, *, refresh: bool = False) -> tuple[set[str] | None, str]:
    """Return model IDs visible to an API key (one ``models.list()`` call, cached).

    Args:
        api_key: OpenAI key (never logged).
        refresh: Ignore the cache.

    Returns:
        ``(model_ids_or_None, error_message)``.
    """
    if not api_key:
        return None, "OPENAI_API_KEY not set"
    cache_key = hashlib.sha256(api_key.encode()).hexdigest()
    with _models_lock:
        if not refresh and cache_key in _models_cache:
            return _models_cache[cache_key]
    try:
        import openai

        client = openai.OpenAI(api_key=api_key, max_retries=2, timeout=30)
        ids = {m.id for m in client.models.list()}
        outcome: tuple[set[str] | None, str] = (ids, "")
    except ImportError:
        outcome = (None, "openai package not installed")
    except Exception as exc:  # noqa: BLE001 - network/auth errors vary
        outcome = (None, redact(f"models.list failed: {type(exc).__name__}: {exc}"))
    with _models_lock:
        _models_cache[cache_key] = outcome
    return outcome


def _status_code(exc: BaseException) -> int | None:
    """Extract an HTTP status code from an OpenAI exception.

    Args:
        exc: Exception raised by the SDK.

    Returns:
        Status code or ``None``.
    """
    code = getattr(exc, "status_code", None)
    if code is None:
        response = getattr(exc, "response", None)
        code = getattr(response, "status_code", None)
    return int(code) if isinstance(code, int) else None


def is_retryable_openai(exc: BaseException) -> bool:
    """Whether an exception is transient (429, 5xx, timeout, connection).

    Args:
        exc: Exception from the SDK.

    Returns:
        ``True`` if a retry may succeed.
    """
    name = type(exc).__name__
    if name in ("RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError", "TimeoutError"):
        # insufficient_quota is a 429 that never succeeds on retry
        return "insufficient_quota" not in str(exc)
    code = _status_code(exc)
    return code is not None and (code == 429 or code >= 500)


def retry_after_seconds(exc: BaseException) -> float | None:
    """Read ``Retry-After`` / ``retry-after-ms`` headers from an exception.

    Args:
        exc: Exception from the SDK.

    Returns:
        Seconds or ``None``.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        if headers.get("retry-after-ms"):
            return float(headers["retry-after-ms"]) / 1000.0
        if headers.get("retry-after"):
            return float(headers["retry-after"])
    except (TypeError, ValueError):
        return None
    return None


def _adapt_capabilities(caps: ModelCapabilities, message: str, use_responses: bool) -> bool:
    """Update capabilities from a 400 error message.

    Args:
        caps: Capabilities to update (mutated).
        message: Error text from the API.
        use_responses: Whether the Responses API is in use.

    Returns:
        ``True`` if something was changed (so a retry makes sense).
    """
    msg = message.lower()
    if not use_responses and "max_tokens" in msg and "max_completion_tokens" in msg:
        new = "max_completion_tokens" if caps.token_param == "max_tokens" else "max_tokens"  # noqa: S105
        if new != caps.token_param:
            caps.token_param = new
            return True
    for param in ("temperature", "seed", "top_p"):
        if param in msg and param not in caps.dropped:
            caps.dropped.add(param)
            return True
    if ("response_format" in msg or "json_schema" in msg or "text.format" in msg) and caps.response_format != "none":
        caps.response_format = "json_object" if caps.response_format == "json_schema" else "none"
        return True
    return False


class OpenAIBackend(TranslatorBackend):
    """Translator backed by an OpenAI model."""

    provider = "openai"
    produces_analysis = True
    parallel_mode = "threads"
    runs_on_local_gpu = False

    def __init__(
        self, name: str, cfg: AppConfig, prompts: PromptSet, api_key: str | None, client: Any | None = None
    ) -> None:
        """Create the backend (no network calls yet).

        Args:
            name: OpenAI model ID from the registry.
            cfg: Application config.
            prompts: Prompt templates.
            api_key: OpenAI API key (never logged).
            client: Optional pre-built client (used by tests).
        """
        self.name = name
        self.cfg = cfg
        self.prompts = prompts
        self._api_key = api_key
        self._client = client
        self._model_available: bool | None = None if client is None else True
        self.limiter = RateLimiter(cfg.concurrency.openai_requests_per_minute)
        self.policy = RetryPolicy(max_retries=cfg.concurrency.max_retries)

    # ---------------------------------------------------------------- lifecycle
    def load(self) -> None:
        """Create the SDK client and check model availability.

        Raises:
            RuntimeError: If the ``openai`` package is missing or no key is set.
        """
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("OPENAI_API_KEY not set")
            try:
                import openai
            except ImportError as exc:
                raise RuntimeError("openai package not installed (pip install -r requirements.txt)") from exc
            self._client = openai.OpenAI(
                api_key=self._api_key, max_retries=0, timeout=self.cfg.concurrency.request_timeout_s
            )
        if self._model_available is None:
            ids, err = list_available_models(self._api_key)
            self._model_available = True if ids is None else (self.name in ids)
            if ids is None:
                logger.warning("Could not verify availability of %s (%s); will try anyway", self.name, err)
            elif not self._model_available:
                logger.warning("%s is not in models.list() for this key — records will be Failed", self.name)

    def is_available(self) -> tuple[bool, str]:
        """Check the key and whether the model is listed for it.

        Returns:
            ``(ok, reason)``.
        """
        if not self._api_key and self._client is None:
            return False, "OPENAI_API_KEY not set"
        if self._client is not None and self._model_available:
            return True, "ready"
        ids, err = list_available_models(self._api_key)
        if ids is None:
            return False, err
        if self.name not in ids:
            return False, "not available for this key"
        return True, "ready"

    def unload(self) -> None:
        """Close the HTTP client."""
        close = getattr(self._client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.debug("client close failed", exc_info=True)
        self._client = None

    def max_workers(self) -> int:
        """Concurrent request threads.

        Returns:
            ``concurrency.openai_max_workers``.
        """
        return max(1, self.cfg.concurrency.openai_max_workers)

    # ---------------------------------------------------------------- calls
    def _caps(self) -> ModelCapabilities:
        """Get (or create) this model's cached capabilities.

        Returns:
            :class:`ModelCapabilities`.
        """
        with _cap_lock:
            return _capabilities.setdefault(self.name, ModelCapabilities())

    def _build_kwargs(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        schema_name: str,
        max_tokens: int,
        caps: ModelCapabilities,
    ) -> dict[str, Any]:
        """Assemble request parameters according to known capabilities.

        Args:
            messages: Chat messages.
            schema: JSON schema for structured output.
            schema_name: Schema name.
            max_tokens: Output-token limit.
            caps: Model capabilities.

        Returns:
            Keyword arguments for the SDK call.
        """
        gen = self.cfg.generation
        use_responses = self.cfg.openai.use_responses_api
        kwargs: dict[str, Any] = {"model": self.name}
        if use_responses:
            kwargs["input"] = messages
            kwargs["max_output_tokens"] = max_tokens
            if caps.response_format == "json_schema":
                kwargs["text"] = {
                    "format": {"type": "json_schema", "name": schema_name, "schema": schema, "strict": True}
                }
            elif caps.response_format == "json_object":
                kwargs["text"] = {"format": {"type": "json_object"}}
        else:
            kwargs["messages"] = messages
            kwargs[caps.token_param] = max_tokens
            if "seed" not in caps.dropped:
                kwargs["seed"] = gen.seed
            if caps.response_format == "json_schema":
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "schema": schema, "strict": True},
                }
            elif caps.response_format == "json_object":
                kwargs["response_format"] = {"type": "json_object"}
        if "temperature" not in caps.dropped:
            kwargs["temperature"] = gen.temperature
        return kwargs

    def _parse_response(self, response: Any) -> CallOutcome:
        """Extract text and usage from a Chat Completions or Responses object.

        Args:
            response: SDK response object.

        Returns:
            :class:`CallOutcome` (retries filled in by the caller).
        """
        usage = getattr(response, "usage", None)
        if self.cfg.openai.use_responses_api:
            text = getattr(response, "output_text", "") or ""
            inp = int(getattr(usage, "input_tokens", 0) or 0)
            out = int(getattr(usage, "output_tokens", 0) or 0)
            details = getattr(usage, "output_tokens_details", None)
            finish = str(getattr(response, "status", "") or "")
            incomplete = getattr(response, "incomplete_details", None)
            if incomplete is not None and getattr(incomplete, "reason", "") == "max_output_tokens":
                finish = "length"
        else:
            choice = response.choices[0]
            text = choice.message.content or ""
            finish = str(getattr(choice, "finish_reason", "") or "")
            inp = int(getattr(usage, "prompt_tokens", 0) or 0)
            out = int(getattr(usage, "completion_tokens", 0) or 0)
            details = getattr(usage, "completion_tokens_details", None)
        total = int(getattr(usage, "total_tokens", 0) or 0) or (inp + out)
        reasoning = int(getattr(details, "reasoning_tokens", 0) or 0) if details is not None else 0
        return CallOutcome(text, inp, out, total, reasoning, finish)

    def structured_call(
        self, messages: list[dict[str, str]], schema: dict[str, Any], schema_name: str, max_tokens: int | None = None
    ) -> CallOutcome:
        """Make one structured-output call with capability adaptation and retries.

        Args:
            messages: Chat messages.
            schema: Strict JSON schema.
            schema_name: Name sent with the schema.
            max_tokens: Output-token limit (defaults to ``generation.max_output_tokens``).

        Returns:
            :class:`CallOutcome` with provider-reported usage.

        Raises:
            RuntimeError: If the model is not available to the key.
            Exception: SDK errors that are not recoverable.
        """
        if self._client is None:
            self.load()
        if self._model_available is False:
            raise RuntimeError(MODEL_NOT_AVAILABLE)
        limit = max_tokens or self.cfg.generation.max_output_tokens
        caps = self._caps()
        retries_total = 0
        for _adapt_round in range(5):
            kwargs = self._build_kwargs(messages, schema, schema_name, limit, caps)

            def _do(kw: dict[str, Any] = kwargs) -> Any:
                """Perform the SDK request after the rate limiter.

                Args:
                    kw: Request kwargs.

                Returns:
                    SDK response.
                """
                self.limiter.acquire()
                if self.cfg.openai.use_responses_api:
                    return self._client.responses.create(**kw)  # type: ignore[union-attr]
                return self._client.chat.completions.create(**kw)  # type: ignore[union-attr]

            try:
                response, retries = call_with_retries(
                    _do, is_retryable=is_retryable_openai, retry_after=retry_after_seconds, policy=self.policy
                )
            except Exception as exc:
                code = _status_code(exc)
                if code == 404 or type(exc).__name__ == "NotFoundError":
                    self._model_available = False
                    raise RuntimeError(MODEL_NOT_AVAILABLE) from exc
                if code == 400 or type(exc).__name__ == "BadRequestError":
                    with _cap_lock:
                        changed = _adapt_capabilities(caps, str(exc), self.cfg.openai.use_responses_api)
                    if changed:
                        logger.info("%s: adapted parameters after 400 (%s)", self.name, truncate(str(exc), 160))
                        continue
                raise
            retries_total += retries
            outcome = self._parse_response(response)
            outcome.retries = retries_total
            return outcome
        raise RuntimeError("Too many parameter adaptations; giving up")

    def translate_one(self, req: TranslationRequest) -> TranslationResult:
        """Translate one record (never raises).

        Args:
            req: Translation request.

        Returns:
            :class:`TranslationResult` with ``error`` set on failure.
        """
        start = time.perf_counter()
        result = TranslationResult()
        messages = self.prompts.translation_messages(
            language=req.language_name, source=req.source, verse=req.verse, shloka=req.text
        )
        try:
            outcome = self._call_with_truncation_retry(messages, result)
            raw = outcome.text
            try:
                parsed = parse_structured(raw, TranslationOutput)
            except StructuredOutputError as first_err:
                logger.info("Record %s: invalid JSON, retrying once with reminder", req.record_id)
                retry_msgs = [
                    *messages,
                    {"role": "assistant", "content": raw[:4000]},
                    {"role": "user", "content": JSON_REMINDER},
                ]
                outcome = self._call_with_truncation_retry(retry_msgs, result)
                raw = outcome.text
                try:
                    parsed = parse_structured(raw, TranslationOutput)
                except StructuredOutputError as second_err:
                    raise StructuredOutputError(f"{second_err} (after retry; first: {first_err})") from second_err
            if not isinstance(parsed, TranslationOutput):
                raise StructuredOutputError("unexpected parse type")
            result.translation = parsed.translation
            result.segmentation = [s for s in parsed.word_segmentation if s.strip()]
            result.word_meanings = [WordMeaning(w.word, w.meaning) for w in parsed.word_meanings]
            result.translator_notes = parsed.translator_notes.strip()
            result.has_analysis = bool(result.segmentation or result.word_meanings)
            result.raw_response = raw
        except Exception as exc:  # noqa: BLE001 - per-record isolation
            result.error = redact(f"{type(exc).__name__}: {truncate(str(exc), 300)}")
            if MODEL_NOT_AVAILABLE in str(exc):
                result.error = MODEL_NOT_AVAILABLE
        result.latency_s = time.perf_counter() - start
        result.total_tokens = result.total_tokens or (result.input_tokens + result.output_tokens)
        return result

    def _call_with_truncation_retry(self, messages: list[dict[str, str]], acc: TranslationResult) -> CallOutcome:
        """Call the model; if output was cut off by the token limit, retry once with double the limit.

        Token usage of every attempt is added to ``acc`` (all attempts are billed).

        Args:
            messages: Chat messages.
            acc: Result accumulating tokens/retries/remarks.

        Returns:
            The final :class:`CallOutcome`.
        """
        limit = self.cfg.generation.max_output_tokens
        outcome = self.structured_call(messages, TRANSLATION_JSON_SCHEMA, "shloka_translation", limit)
        self._accumulate(acc, outcome)
        if outcome.finish_reason == "length":
            acc.remarks.append(f"Output hit max_output_tokens={limit}; retried with {limit * 2}")
            outcome = self.structured_call(messages, TRANSLATION_JSON_SCHEMA, "shloka_translation", limit * 2)
            self._accumulate(acc, outcome)
        return outcome

    @staticmethod
    def _accumulate(acc: TranslationResult, outcome: CallOutcome) -> None:
        """Add one call's usage to the running totals.

        Args:
            acc: Result being built.
            outcome: Call outcome.
        """
        acc.input_tokens += outcome.input_tokens
        acc.output_tokens += outcome.output_tokens
        acc.total_tokens += outcome.total_tokens
        acc.reasoning_tokens += outcome.reasoning_tokens
        acc.retries += outcome.retries

    def translate_batch(self, items: list[TranslationRequest]) -> list[TranslationResult]:
        """Translate items sequentially (parallelism is handled by the executor).

        Args:
            items: Requests.

        Returns:
            Results aligned with ``items``.
        """
        return [self.translate_one(item) for item in items]
