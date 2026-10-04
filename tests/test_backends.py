"""Backend behaviour with fake clients (no network)."""

from __future__ import annotations

import contextlib
import json
from types import SimpleNamespace

from shloka_translator.config import load_config
from shloka_translator.models import openai_backend as ob
from shloka_translator.models.base import TranslationRequest
from shloka_translator.models.indictrans2_backend import split_for_length, transliterate_to_devanagari
from shloka_translator.models.ollama_backend import OllamaBackend
from shloka_translator.pipeline.executor import RetryPolicy, call_with_retries
from shloka_translator.prompting.schemas import TRANSLATION_JSON_SCHEMA, TranslationOutput, parse_structured
from shloka_translator.prompting.templates import PromptSet, render

GOOD = json.dumps(
    {
        "translation": "Hello",
        "word_segmentation": ["अस्ति", "उत्तरस्याम्"],
        "word_meanings": [{"word": "अस्ति", "meaning": "there is"}],
        "translator_notes": "",
    }
)


class BadRequestError(Exception):
    """Mimics openai.BadRequestError."""

    status_code = 400


class RateLimitError(Exception):
    """Mimics openai.RateLimitError."""

    status_code = 429


def _response(text: str, finish: str = "stop") -> SimpleNamespace:
    """Fake Chat Completions response.

    Args:
        text: Content.
        finish: Finish reason.

    Returns:
        Namespace shaped like the SDK object.
    """
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=40,
        total_tokens=140,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=5),
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)], usage=usage
    )


class FakeCompletions:
    """Scripted ``chat.completions`` endpoint."""

    def __init__(self, script: list) -> None:
        """Store the script.

        Args:
            script: Items are responses or exceptions, consumed in order.
        """
        self.script = script
        self.calls: list[dict] = []

    def create(self, **kwargs):
        """Return the next scripted item.

        Args:
            **kwargs: Request parameters (recorded).

        Returns:
            Fake response.

        Raises:
            Exception: Scripted exception.
        """
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _backend(script: list, name: str = "gpt-test") -> tuple[ob.OpenAIBackend, FakeCompletions]:
    """OpenAI backend with a fake client.

    Args:
        script: Scripted responses.
        name: Model name.

    Returns:
        ``(backend, completions)``.
    """
    cfg = load_config()
    prompts = PromptSet.load(cfg.path(cfg.paths.prompts_folder))
    comp = FakeCompletions(script)
    client = SimpleNamespace(chat=SimpleNamespace(completions=comp))
    ob._capabilities.pop(name, None)
    b = ob.OpenAIBackend(name, cfg, prompts, api_key=None, client=client)
    b.policy = RetryPolicy(max_retries=2, base_delay=0.001, max_delay=0.002)
    return b, comp


REQ = TranslationRequest(0, "1", "अस्त्युत्तरस्यां दिशि", "कुमारसम्भवम्", "1.1", "Sanskrit", "san_Deva", "Devanagari")


def test_openai_success_and_tokens() -> None:
    """Tokens come from usage; structured output parsed."""
    b, comp = _backend([_response(GOOD)])
    res = b.translate_one(REQ)
    assert res.ok and res.translation == "Hello" and res.segmentation == ["अस्ति", "उत्तरस्याम्"]
    assert (res.input_tokens, res.output_tokens, res.total_tokens, res.reasoning_tokens) == (100, 40, 140, 5)
    sent = comp.calls[0]
    assert sent["response_format"]["type"] == "json_schema" and sent["temperature"] == 0.0 and sent["seed"] == 42


def test_openai_capability_adaptation_is_cached() -> None:
    """A 400 on temperature drops it, retries once, and remembers it for the model."""
    err = BadRequestError("Unsupported value: 'temperature' does not support 0.0 with this model.")
    b, comp = _backend([err, _response(GOOD), _response(GOOD)], name="gpt-5-test")
    assert b.translate_one(REQ).ok
    assert "temperature" not in comp.calls[1]
    b.translate_one(REQ)
    assert "temperature" not in comp.calls[2] and len(comp.calls) == 3


def test_openai_max_tokens_rename() -> None:
    """max_completion_tokens ↔ max_tokens rename on 400."""
    err = BadRequestError("Unsupported parameter: 'max_completion_tokens'. Use 'max_tokens' instead.")
    b, comp = _backend([err, _response(GOOD)], name="old-model")
    assert b.translate_one(REQ).ok
    assert "max_tokens" in comp.calls[1] and "max_completion_tokens" not in comp.calls[1]


def test_openai_invalid_json_retry_then_fail() -> None:
    """Invalid JSON → one reminder retry → Failed; tokens of both calls counted."""
    b, _ = _backend([_response("not json"), _response("still not json")])
    res = b.translate_one(REQ)
    assert not res.ok and "StructuredOutputError" in res.error and res.input_tokens == 200
    b, _ = _backend([_response("oops"), _response(GOOD)])
    assert b.translate_one(REQ).ok


def test_openai_rate_limit_retried() -> None:
    """429 is retried with backoff."""
    b, comp = _backend([RateLimitError("slow down"), _response(GOOD)])
    res = b.translate_one(REQ)
    assert res.ok and res.retries == 1 and len(comp.calls) == 2


def test_openai_model_not_available() -> None:
    """Unavailable model → every record Failed with the standard remark."""
    b, _ = _backend([])
    b._model_available = False
    assert b.translate_one(REQ).error == "Model not available to this API key"


def test_ollama_parses_think_and_counts_tokens() -> None:
    """Ollama: <think> stripped, prompt_eval_count/eval_count used."""
    cfg = load_config()
    prompts = PromptSet.load(cfg.path(cfg.paths.prompts_folder))

    class FakeOllama:
        """Fake ollama.Client."""

        def __init__(self) -> None:
            """Init."""
            self.kwargs = None

        def chat(self, **kwargs):
            """Fake chat.

            Args:
                **kwargs: Request.

            Returns:
                Response dict.
            """
            self.kwargs = kwargs
            return {
                "message": {"content": "<think>hmm</think>" + GOOD},
                "prompt_eval_count": 50,
                "eval_count": 20,
                "eval_duration": 1_000_000_000,
            }

    fake = FakeOllama()
    b = OllamaBackend("qwen3:32b", cfg.models["qwen3:32b"], cfg, prompts, client=fake)
    res = b.translate_one(REQ)
    assert res.ok and (res.input_tokens, res.output_tokens, res.total_tokens) == (50, 20, 70)
    assert fake.kwargs["think"] is False and fake.kwargs["format"] == TRANSLATION_JSON_SCHEMA
    assert fake.kwargs["messages"][-1]["content"].endswith("/no_think")


def test_schema_and_render() -> None:
    """Strict schema, fenced JSON parsing, brace-safe rendering."""
    assert TRANSLATION_JSON_SCHEMA["additionalProperties"] is False
    assert set(TRANSLATION_JSON_SCHEMA["required"]) == {
        "translation",
        "word_segmentation",
        "word_meanings",
        "translator_notes",
    }
    parsed = parse_structured("```json\n" + GOOD + "\n```", TranslationOutput)
    assert parsed.translation == "Hello"
    assert render("{a} {b} {x}", a="1", b="{}") == "1 {} {x}"


def test_retry_helper_non_retryable() -> None:
    """Non-retryable errors propagate immediately."""
    calls = []

    def boom():
        """Raise.

        Raises:
            ValueError: Always.
        """
        calls.append(1)
        raise ValueError("x")

    with contextlib.suppress(ValueError):
        call_with_retries(boom, is_retryable=lambda e: False, policy=RetryPolicy(3, 0.001, 0.001))
    assert len(calls) == 1


def test_chunking_and_transliteration() -> None:
    """Long text is split at dandas; Telugu Sanskrit maps to Devanagari."""
    chunks = split_for_length("अ आ । इ ई ॥ उ ऊ", lambda s: len(s.split()), 3)
    assert chunks == ["अ आ ।", "इ ई ॥", "उ ऊ"]
    text, method = transliterate_to_devanagari("ధర్మ", "Telugu")
    assert text == "धर्म" and method


def test_indictrans2_batching_logic(monkeypatch) -> None:
    """Chunking, transliteration remarks, per-record token sums and error isolation (generation faked).

    Args:
        monkeypatch: Pytest monkeypatch.
    """
    import sys
    import types

    from shloka_translator.models.indictrans2_backend import IndicTrans2Backend

    if "torch" not in sys.modules:
        try:
            import torch  # noqa: F401
        except ImportError:
            fake = types.ModuleType("torch")
            fake.cuda = types.SimpleNamespace(OutOfMemoryError=MemoryError, empty_cache=lambda: None)
            monkeypatch.setitem(sys.modules, "torch", fake)
    cfg = load_config(overrides={"generation": {"hf_max_length": 24}})
    b = IndicTrans2Backend("IndicTrans2", "fake/id", cfg)
    b.model = object()  # mark as loaded
    b.tokenizer = lambda text, add_special_tokens=True: {"input_ids": text.split()}

    def fake_generate(chunks) -> None:
        """Pretend to translate.

        Args:
            chunks: Chunks to fill.

        Raises:
            RuntimeError: For the poisoned chunk.
        """
        for c in chunks:
            if "विषम्" in c.text:
                raise RuntimeError("bad chunk")
            c.translation, c.in_tokens, c.out_tokens = f"T({c.text[:4]})", len(c.text.split()), 3

    monkeypatch.setattr(b, "_generate", fake_generate)
    long_text = " ".join(["राम वनं गच्छति ।"] * 12)
    reqs = [
        TranslationRequest(0, "a", long_text, language_name="Sanskrit", language_code="san_Deva", script="Devanagari"),
        TranslationRequest(1, "b", "ధర్మ క్షేత్రే", language_name="Sanskrit", language_code="san_Deva", script="Telugu"),
        TranslationRequest(2, "c", "hello", language_name="Hindi", language_code="hin_Deva", script="Latin"),
        TranslationRequest(
            3, "d", "विषम् अस्ति", language_name="Sanskrit", language_code="san_Deva", script="Devanagari"
        ),
    ]
    res = b.translate_batch(reqs)
    assert res[0].ok and any("Chunked into" in r for r in res[0].remarks) and res[0].input_tokens == 48
    assert res[1].ok and any("Transliterated Telugu" in r for r in res[1].remarks)
    assert not res[2].ok and "Romanised" in res[2].error
    assert not res[3].ok and "bad chunk" in res[3].error


def test_auto_setup_badges() -> None:
    """Ollama / IndicTransToolkit gaps are fixed on Run (⬇); keys, GPUs and training are not (⚠)."""
    from shloka_translator.models.registry import ModelStatus, auto_setup_reason

    assert auto_setup_reason("ollama", "ollama Python package not installed — re-run the Install cell")
    assert auto_setup_reason("ollama", "Ollama server not running (ConnectError)")
    assert auto_setup_reason("hf_seq2seq", "IndicTransToolkit not installed — re-run the Install cell")
    assert auto_setup_reason("hf_seq2seq", "No GPU — switch to a GPU runtime") is None
    assert auto_setup_reason("openai", "OPENAI_API_KEY not set") is None
    assert ModelStatus("qwen3:32b", "ollama", False, "x", setup_on_run=True).badge.startswith("⬇")
    assert ModelStatus("gpt-5", "openai", False, "x").badge.startswith("⚠")


def test_installer_parsing(tmp_path) -> None:
    """The fault-tolerant installer follows -r includes and extracts package names.

    Args:
        tmp_path: Temp dir.
    """
    import importlib.util

    from tests.helpers import REPO

    spec = importlib.util.spec_from_file_location("install_requirements", REPO / "scripts" / "install_requirements.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (tmp_path / "a.txt").write_text("ollama>=0.4,<1  # comment\n\n-r b.txt\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("IndicTransToolkit>=1.0\n", encoding="utf-8")
    reqs = mod.read_requirements(tmp_path / "a.txt")
    assert reqs == ["ollama>=0.4,<1", "IndicTransToolkit>=1.0"]
    assert [mod.package_name(r) for r in reqs] == ["ollama", "IndicTransToolkit"]
    assert "IndicTransToolkit" in mod.FALLBACKS


def test_itihasa_parquet_api_fallback(monkeypatch) -> None:
    """Script-based datasets load via the Hub Parquet API when ``datasets`` cannot run the script.

    Args:
        monkeypatch: Pytest monkeypatch.
    """
    import io
    import sys
    import urllib.request

    import pandas as pd

    from shloka_translator.training import data_prep

    frame = pd.DataFrame({"translation": [{"en": "There is in the north…", "sn": "अस्त्युत्तरस्यां दिशि"}]})
    listing = json.dumps({"Itihasa": {"train": ["https://hub/train.parquet"], "test": ["https://hub/test.parquet"]}})
    requested: list[str] = []

    class Resp(io.BytesIO):
        """Context-manager response."""

        def __enter__(self):
            """Enter.

            Returns:
                Self.
            """
            return self

        def __exit__(self, *exc) -> None:
            """Exit.

            Args:
                *exc: Exception info.
            """

    def fake_urlopen(req, timeout=0):
        """Serve the listing, then (fake) parquet bytes.

        Args:
            req: Request.
            timeout: Ignored.

        Returns:
            Response.
        """
        requested.append(req.full_url)
        return Resp(listing.encode() if req.full_url.endswith("/parquet") else b"PAR1")

    class FakeDatasets:
        """``datasets`` stand-in that fails like datasets>=4 on a script dataset."""

        @staticmethod
        def load_dataset(*args, **kwargs):
            """Fail.

            Args:
                *args: Ignored.
                **kwargs: Ignored.

            Raises:
                RuntimeError: Always.
            """
            raise RuntimeError("Dataset scripts are no longer supported, but found itihasa.py")

    monkeypatch.setitem(sys.modules, "datasets", FakeDatasets)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(pd, "read_parquet", lambda _buf: frame)
    pairs, report = data_prep.load_itihasa()
    assert report.status == "loaded" and "parquet API" in report.note, report.note
    assert len(pairs) == 2 and pairs[0].src == "अस्त्युत्तरस्यां दिशि" and pairs[0].lang == "san_Deva"
    assert requested[0] == "https://huggingface.co/api/datasets/rahular/itihasa/parquet"
