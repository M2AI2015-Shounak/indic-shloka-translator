"""End-to-end run: ingest → detect → translate → analyse → write Excel.

Flow (per Section 10 of the specification): load config → resolve secrets →
ingest / validate / merge → detect languages (once per run) → split into
supported / Skipped → for each selected model: start GPU monitor → translate
(parallel or batched, checkpointed) → stop monitor → analysis helper where
needed → assemble rows → write the six-sheet workbook. "ALL MODELS" also
writes a side-by-side comparison workbook.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from shloka_translator.analysis.word_meanings import AnalysisHelper, make_llm_tiebreaker
from shloka_translator.config import AppConfig
from shloka_translator.io.loader import load_folder
from shloka_translator.io.merger import AcceptedTable, merge_tables
from shloka_translator.io.validator import validate_table
from shloka_translator.langid.detector import DetectionResult, LanguageDetector
from shloka_translator.langid.languages import LanguageTable
from shloka_translator.models.base import TranslationRequest, TranslationResult, TranslatorBackend
from shloka_translator.models.registry import ALL_MODELS, build_backend
from shloka_translator.monitoring.gpu_monitor import GPUMonitor, RunStats
from shloka_translator.output.excel_writer import WorkbookData, write_comparison, write_workbook
from shloka_translator.output.naming import build_stem, unique_path
from shloka_translator.pipeline.checkpoint import CheckpointStore, checkpoint_key
from shloka_translator.pipeline.executor import run_parallel
from shloka_translator.pipeline.progress import MultiModelProgress, ProgressReporter
from shloka_translator.prompting.templates import PromptSet
from shloka_translator.secrets import get_secret, install_redaction, redact
from shloka_translator.utils.text import truncate
from shloka_translator.utils.timing import now_in_tz

logger = logging.getLogger(__name__)

SUCCESS, SKIPPED, FAILED = "Success", "Skipped", "Failed"


# --------------------------------------------------------------------------- data classes


@dataclass
class RunOptions:
    """Per-run choices (from the widget or the CLI).

    Attributes:
        model: Registry name or ``"ALL MODELS"``.
        max_records: Process only the first N merged records (smoke tests).
        resume: Override ``runtime.resume_from_checkpoint``.
        concurrency: Override worker count for the selected provider.
        use_widgets: Force ipywidgets progress on/off.
        interactive_secrets: Allow ``getpass`` prompts for missing secrets.
        confirm: Callback ``(message) -> bool`` for risky choices (VRAM spill). Default: proceed.
        backend_factory: Test hook ``(name) -> TranslatorBackend``.
        openai_key: Pre-resolved OpenAI key (skips secret lookup).
        hf_token: Pre-resolved HF token.
    """

    model: str
    max_records: int | None = None
    resume: bool | None = None
    concurrency: int | None = None
    use_widgets: bool | None = None
    interactive_secrets: bool = False
    confirm: Callable[[str], bool] | None = None
    backend_factory: Callable[[str], TranslatorBackend] | None = None
    openai_key: str | None = None
    hf_token: str | None = None


@dataclass
class RecordState:
    """Mutable per-record state while a model runs.

    Attributes:
        status: Final status.
        result: Translation result (if any).
        remarks: Remarks collected for this record.
        helper_in: Analysis-helper input tokens.
        helper_out: Analysis-helper output tokens.
        cached: Whether the result came from the checkpoint.
    """

    status: str = ""
    result: TranslationResult | None = None
    remarks: list[str] = field(default_factory=list)
    helper_in: int = 0
    helper_out: int = 0
    cached: bool = False


@dataclass
class ModelRunResult:
    """Outcome of one model.

    Attributes:
        model: Model name.
        output_path: Workbook written.
        log_path: Log file.
        counts: Status counts.
        tokens: Token totals.
        rows: ``Translations`` rows (for the comparison workbook / notebook display).
        interrupted: Whether the run was interrupted.
    """

    model: str
    output_path: Path | None = None
    log_path: Path | None = None
    counts: dict[str, int] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)
    interrupted: bool = False


@dataclass
class RunResult:
    """Outcome of the whole run.

    Attributes:
        ok: ``False`` if nothing could be processed.
        message: Human-readable summary / reason.
        models: Per-model results.
        comparison_path: ALL-MODELS comparison workbook.
        validation: Validation report rows.
    """

    ok: bool
    message: str
    models: list[ModelRunResult] = field(default_factory=list)
    comparison_path: Path | None = None
    validation: list[list[Any]] = field(default_factory=list)


@dataclass
class Prepared:
    """Ingested + detected input shared by all models of a run.

    Attributes:
        records: Merged records frame.
        validation: Rows for ``Input_Validation``.
        detections: One detection per record.
        detector_info: Description of detection tiers in use.
        input_files: Accepted source names.
    """

    records: pd.DataFrame
    validation: list[list[Any]]
    detections: list[DetectionResult]
    detector_info: str
    input_files: list[str]


# --------------------------------------------------------------------------- logging


def setup_logging(cfg: AppConfig) -> None:
    """Configure console logging once and install secret redaction.

    Args:
        cfg: Config (``runtime.log_level``).
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, cfg.runtime.log_level.upper(), logging.INFO))
    if not any(getattr(h, "_shloka_console", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        handler._shloka_console = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    install_redaction(root)


def attach_file_log(path: Path) -> logging.Handler:
    """Add a file handler for one model's log (``logs/<stem>.log``).

    Args:
        path: Log file path.

    Returns:
        The handler (remove it with :func:`detach_file_log`).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    install_redaction(logging.getLogger())
    return handler


def detach_file_log(handler: logging.Handler) -> None:
    """Remove and close a file handler.

    Args:
        handler: Handler from :func:`attach_file_log`.
    """
    logging.getLogger().removeHandler(handler)
    handler.close()


# --------------------------------------------------------------------------- environment


def gpu_present() -> tuple[bool, str]:
    """Detect a CUDA GPU (torch first, NVML second).

    Returns:
        ``(present, name)``.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        import pynvml

        pynvml.nvmlInit()
        if pynvml.nvmlDeviceGetCount() > 0:
            name = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
            return True, name.decode() if isinstance(name, bytes) else str(name)
    except Exception:
        logger.debug("NVML GPU probe failed", exc_info=True)
    return False, ""


def library_versions() -> dict[str, str]:
    """Versions of key libraries (for ``Run_Summary``).

    Returns:
        Package → version (``"not installed"`` if missing).
    """
    from importlib import metadata

    out = {"python": platform.python_version()}
    for pkg in (
        "pandas",
        "openpyxl",
        "pydantic",
        "openai",
        "ollama",
        "torch",
        "transformers",
        "IndicTransToolkit",
        "indictranstoolkit",
        "nvidia-ml-py",
        "tenacity",
        "fasttext-wheel",
        "sanskrit_parser",
    ):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue
    return out


def runtime_info() -> str:
    """Describe the runtime (Colab or not, CPU, RAM).

    Returns:
        One-line description.
    """
    colab = "google.colab" in sys.modules or bool(os.environ.get("COLAB_RELEASE_TAG"))
    parts = [f"Colab={'yes' if colab else 'no'}"]
    if os.environ.get("COLAB_RELEASE_TAG"):
        parts.append(f"release={os.environ['COLAB_RELEASE_TAG']}")
    parts.append(f"platform={platform.platform()}")
    parts.append(f"cpus={os.cpu_count()}")
    try:
        import psutil

        parts.append(f"ram={psutil.virtual_memory().total / 1024**3:.1f} GB")
    except ImportError:
        pass
    return "; ".join(parts)


# --------------------------------------------------------------------------- stage 1: ingest + detect


def ingest(cfg: AppConfig) -> tuple[pd.DataFrame, list[list[Any]], list[str]]:
    """Load, validate and merge all input files.

    Args:
        cfg: Config.

    Returns:
        ``(records, validation_rows, accepted_source_names)``.
    """
    folder = cfg.path(cfg.paths.input_folder)
    tables = load_folder(folder, cfg.input.extensions, cfg.input.recursive)
    accepted: list[AcceptedTable] = []
    validation: list[list[Any]] = []
    for table in tables:
        frame, report = validate_table(table, cfg.schema_.expected_columns, cfg.schema_.allow_extra_columns)
        validation.append([report.file, report.sheet, report.status, report.rows, report.reason])
        if frame is not None:
            accepted.append(AcceptedTable(frame, table))
    merged = merge_tables(accepted, cfg.schema_.expected_columns)
    for note in merged.dropped_duplicates:
        validation.append(["(merge)", "", "Info", 0, note])
    return merged.records, validation, [a.table.source_name for a in accepted]


def build_detector(cfg: AppConfig, prompts: PromptSet, openai_key: str | None) -> tuple[LanguageDetector, str]:
    """Create the tiered language detector from config.

    Args:
        cfg: Config.
        prompts: Prompts (for the tiebreaker).
        openai_key: OpenAI key.

    Returns:
        ``(detector, description)``.
    """
    ld = cfg.language_detection
    tiebreaker, tb_name = make_llm_tiebreaker(cfg, prompts, openai_key)
    detector = LanguageDetector(
        LanguageTable.from_config(ld.supported_languages),
        min_confidence=ld.min_confidence,
        use_indiclid=ld.use_indiclid,
        indiclid_dir=cfg.path(ld.indiclid_dir),
        use_fasttext=ld.use_fasttext_fallback,
        tiebreaker=tiebreaker,
        tiebreaker_name=tb_name,
        cache_file=cfg.path(ld.cache_file),
    )
    return detector, tb_name


def prepare_inputs(
    cfg: AppConfig, prompts: PromptSet, openai_key: str | None, max_records: int | None
) -> Prepared | None:
    """Ingest and detect languages once for the whole run.

    Args:
        cfg: Config.
        prompts: Prompts.
        openai_key: OpenAI key.
        max_records: Optional record limit.

    Returns:
        :class:`Prepared`, or ``None`` when no input file passed validation.
    """
    records, validation, accepted = ingest(cfg)
    if records.empty:
        return Prepared(records, validation, [], "", accepted) if accepted else None
    if max_records:
        records = records.head(int(max_records)).reset_index(drop=True)
    detector, tb_name = build_detector(cfg, prompts, openai_key)
    texts = records["Shloka"].tolist()
    started = time.perf_counter()
    detections = detector.detect_many(texts)
    detector.save_cache()
    tiers = ["script profile", "domain rules"]
    if detector.indiclid is not None and detector.indiclid.available:
        tiers.insert(1, "IndicLID")
    elif detector.indiclid is not None:
        tiers.append(f"IndicLID unavailable ({detector.indiclid.error})")
    if detector.fasttext is not None and detector.fasttext._tried:
        tiers.append("fastText NLLB LID" if detector.fasttext.available else "fastText unavailable")
    tiers.append(f"LLM tiebreaker: {tb_name}" if tb_name else "LLM tiebreaker: none available")
    logger.info("Language detection for %d records in %.1fs", len(texts), time.perf_counter() - started)
    return Prepared(records, validation, detections, "; ".join(tiers), accepted)


# --------------------------------------------------------------------------- stage 2: per model


def _language_label(det: DetectionResult) -> str:
    """Language label for the prompt, mentioning a non-native script.

    Args:
        det: Detection result.

    Returns:
        e.g. ``"Sanskrit (written in Bengali script)"``.
    """
    native = {"Sanskrit": "Devanagari"}
    if det.script and det.script != native.get(det.language_name, det.script):
        return f"{det.language_name} (written in {det.script} script)"
    return det.language_name


def _initial_states(prep: Prepared) -> tuple[list[RecordState], list[TranslationRequest]]:
    """Classify records into Skipped/Failed-by-detection vs. to-translate.

    Args:
        prep: Prepared input.

    Returns:
        ``(states, requests)``.
    """
    states: list[RecordState] = []
    requests: list[TranslationRequest] = []
    for idx, row in prep.records.iterrows():
        det = prep.detections[idx]
        st = RecordState(remarks=list(row["_remarks"]))
        text = str(row["Shloka"])
        if not text.strip():
            st.status, st.remarks = SKIPPED, [*st.remarks, "Empty Shloka — skipped"]
        elif det.method == "error":
            st.status, st.remarks = FAILED, [*st.remarks, det.reason]
        elif not det.is_supported:
            st.status, st.remarks = SKIPPED, [*st.remarks, det.reason]
        else:
            requests.append(
                TranslationRequest(
                    int(idx),
                    str(row["ID"]),
                    text,
                    str(row["Source"]),
                    str(row["Verse"]),
                    det.language_name,
                    det.code,
                    det.script,
                )
            )
        states.append(st)
    return states, requests


def _check_backend(backend: TranslatorBackend, cfg: AppConfig, options: RunOptions) -> str | None:
    """Pre-flight checks; return a failure remark if the model cannot run.

    Args:
        backend: Backend.
        cfg: Config.
        options: Run options.

    Returns:
        ``None`` if OK, else the remark to put on every record.
    """
    has_gpu, _gpu_name = gpu_present()
    if not has_gpu:
        logger.warning(
            "⚠ NO GPU DETECTED — Runtime → Change runtime type → GPU. %s",
            "OpenAI models still work." if backend.provider == "openai" else "",
        )
        if backend.runs_on_local_gpu and not cfg.runtime.allow_cpu_fallback:
            return "No GPU detected — local models are blocked (set runtime.allow_cpu_fallback: true to force CPU)"
    ok, reason = backend.is_available()
    if not ok:
        ok, reason = _auto_setup(backend, cfg, reason)
    if not ok:
        if backend.provider == "openai" and "not available for this key" in reason:
            return "Model not available to this API key"
        return f"Model unavailable: {reason}"
    if backend.provider == "ollama":
        from shloka_translator.models.ollama_backend import vram_preflight

        fits, message = vram_preflight(cfg.models[backend.name], cfg.ollama.context_headroom_gb)
        logger.info("VRAM pre-flight: %s", message)
        if not fits:
            logger.warning(message)
            confirm = options.confirm or (lambda _m: True)
            if not confirm(message):
                return f"Cancelled by user: {message}"
    return None


def _auto_setup(backend: TranslatorBackend, cfg: AppConfig, reason: str) -> tuple[bool, str]:
    """Install/download what a local model is missing, then re-check availability.

    * Ollama: installs the ``ollama`` Python client and the Ollama server inside this VM,
      starts it and pulls the model (``scripts/setup_ollama.sh``).
    * IndicTrans2 / custom: pip-installs ``IndicTransToolkit`` if that is what is missing.

    Args:
        backend: Backend that reported it is not available.
        cfg: Config.
        reason: Original ``is_available`` reason.

    Returns:
        ``(ok, reason)`` after the attempt (unchanged when nothing could be done).
    """
    from shloka_translator.models.registry import auto_setup_reason

    if auto_setup_reason(backend.provider, reason) is None:
        return False, reason
    logger.info("%s is not set up yet (%s) — setting it up automatically", backend.name, reason)
    if backend.provider == "ollama":
        from shloka_translator.models.ollama_backend import run_setup_script

        code = run_setup_script(cfg.repo_root, [backend.name], cfg)
        if code != 0:
            return False, f"Ollama setup failed (exit code {code}); see the setup output / logs/ollama_serve.log"
    else:
        from shloka_translator.utils.packages import ensure_packages

        missing = ensure_packages(["IndicTransToolkit"])
        if missing:
            return False, f"IndicTransToolkit could not be installed: {missing['IndicTransToolkit']}"
    return backend.is_available()


def _translate_threads(
    backend: TranslatorBackend,
    todo: list[TranslationRequest],
    workers: int,
    finish: Callable[[TranslationRequest, TranslationResult], None],
) -> None:
    """Translate with a thread pool (API backends).

    Args:
        backend: Backend.
        todo: Requests.
        workers: Thread count.
        finish: Callback per finished record (main thread).
    """

    def worker(req: TranslationRequest) -> TranslationResult:
        """Translate one request.

        Args:
            req: Request.

        Returns:
            Result.
        """
        return backend.translate_batch([req])[0]

    def on_result(_pos: int, req: TranslationRequest, res: TranslationResult | None, exc: BaseException | None) -> None:
        """Convert exceptions into failed results.

        Args:
            _pos: Position.
            req: Request.
            res: Result.
            exc: Exception.
        """
        if exc is not None or res is None:
            res = TranslationResult(error=redact(f"{type(exc).__name__}: {truncate(str(exc), 300)}"))
        finish(req, res)

    run_parallel(todo, worker, max_workers=workers, on_result=on_result)


def _translate_batched(
    backend: TranslatorBackend,
    todo: list[TranslationRequest],
    batch_size: int,
    finish: Callable[[TranslationRequest, TranslationResult], None],
) -> None:
    """Translate with length-sorted batches in a single process (GPU seq2seq).

    Args:
        backend: Backend.
        todo: Requests.
        batch_size: Records per call.
        finish: Callback per finished record.
    """
    ordered = sorted(todo, key=lambda r: len(r.text))
    for start in range(0, len(ordered), max(1, batch_size)):
        group = ordered[start : start + batch_size]
        try:
            results = backend.translate_batch(group)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            results = [
                TranslationResult(error=redact(f"{type(exc).__name__}: {truncate(str(exc), 300)}")) for _ in group
            ]
        for req, res in zip(group, results, strict=True):
            finish(req, res)


class ModelRunner:
    """Runs one model over the prepared records and writes its workbook."""

    def __init__(
        self,
        cfg: AppConfig,
        prompts: PromptSet,
        prep: Prepared,
        model: str,
        options: RunOptions,
        openai_key: str | None,
        hf_token: str | None,
    ) -> None:
        """Set up the runner.

        Args:
            cfg: Config.
            prompts: Prompts.
            prep: Prepared input.
            model: Model name.
            options: Run options.
            openai_key: OpenAI key.
            hf_token: HF token.
        """
        self.cfg, self.prompts, self.prep, self.model, self.options = cfg, prompts, prep, model, options
        self.openai_key, self.hf_token = openai_key, hf_token
        self.spec = cfg.models.get(model)
        self.provider = self.spec.provider if self.spec else "test"
        self.stem = build_stem(model, cfg.runtime.timezone)
        self.started_at = now_in_tz(cfg.runtime.timezone)
        self.states, self.requests = _initial_states(prep)
        self.latencies: list[float] = []
        self.live_count = 0
        self.interrupted = False
        self.translation_seconds = 0.0
        self.monitor = GPUMonitor(cfg.gpu_monitor.sample_interval_s)
        self.backend: TranslatorBackend | None = None
        self.store: CheckpointStore | None = None
        self.progress: ProgressReporter | None = None
        self.analysis_desc = ""

    # ------------------------------------------------------------------ helpers
    def _workers(self) -> int:
        """Worker count for thread-mode backends.

        Returns:
            Workers (option override wins).
        """
        if self.options.concurrency:
            return max(1, int(self.options.concurrency))
        return self.backend.max_workers() if self.backend else 1

    def _finish(self, req: TranslationRequest, res: TranslationResult) -> None:
        """Record a finished translation (main thread).

        Args:
            req: Request.
            res: Result.
        """
        st = self.states[req.index]
        st.result = res
        st.status = SUCCESS if res.ok else FAILED
        self.latencies.append(res.latency_s)
        self.live_count += 1
        if self.progress:
            self.progress.update(st.status)
        if self.backend is not None and self.backend.produces_analysis and self.store is not None:
            self._checkpoint(req, st)

    def _checkpoint(self, req: TranslationRequest, st: RecordState) -> None:
        """Append a finished record to the checkpoint file.

        Args:
            req: Request.
            st: State.
        """
        if self.store is None or st.result is None:
            return
        payload = st.result.to_cache()
        payload["helper_in"], payload["helper_out"] = st.helper_in, st.helper_out
        payload["analysis_remarks"] = [r for r in st.remarks if r.startswith(("Analysis by", "Partial:"))]
        key = checkpoint_key(self.model, self.cfg.runtime.prompt_version, req.text)
        try:
            self.store.append(key, req.record_id, st.status, payload)
        except OSError as exc:
            logger.warning("Checkpoint write failed: %s", exc)

    def _use_cache(self, todo: list[TranslationRequest]) -> list[TranslationRequest]:
        """Fill states from the checkpoint; return records still to process.

        Args:
            todo: Requests.

        Returns:
            Requests not found in the cache.
        """
        resume = self.cfg.runtime.resume_from_checkpoint if self.options.resume is None else self.options.resume
        self.store = CheckpointStore(
            self.cfg.path(self.cfg.paths.cache_folder), self.model, self.cfg.runtime.prompt_version
        )
        if not resume:
            return todo
        self.store.load()
        remaining = []
        for req in todo:
            cached = self.store.get(checkpoint_key(self.model, self.cfg.runtime.prompt_version, req.text))
            if cached is None:
                remaining.append(req)
                continue
            st = self.states[req.index]
            st.result = TranslationResult.from_cache(cached)
            st.status, st.cached = SUCCESS, True
            st.helper_in, st.helper_out = int(cached.get("helper_in", 0)), int(cached.get("helper_out", 0))
            st.remarks.extend(cached.get("analysis_remarks", []))
            st.remarks.append("Loaded from checkpoint (not re-billed)")
        return remaining

    # ------------------------------------------------------------------ main
    def run(self) -> ModelRunResult:
        """Execute the model run and write its workbook.

        Returns:
            :class:`ModelRunResult`.
        """
        log_path = unique_path(self.cfg.path(self.cfg.paths.log_folder), self.stem, "log")
        handler = attach_file_log(log_path)
        try:
            logger.info("=== Model %s (%s) — %d records ===", self.model, self.provider, len(self.states))
            self._translate_all()
            out = self._write()
        finally:
            if self.backend is not None:
                try:
                    self.backend.unload()
                except Exception:
                    logger.debug("unload failed", exc_info=True)
            detach_file_log(handler)
        out.log_path = log_path
        return out

    def _translate_all(self) -> None:
        """Run pre-flight, cache lookup, translation and analysis."""
        factory = self.options.backend_factory or (
            lambda n: build_backend(n, self.cfg, self.prompts, openai_key=self.openai_key, hf_token=self.hf_token)
        )
        todo = list(self.requests)
        try:
            self.backend = factory(self.model)
            blocker = None if self.options.backend_factory else _check_backend(self.backend, self.cfg, self.options)
        except Exception as exc:  # noqa: BLE001
            blocker = f"Could not initialise model: {type(exc).__name__}: {truncate(str(exc), 200)}"
        if blocker:
            logger.error(blocker)
            for req in todo:
                st = self.states[req.index]
                st.status, st.remarks = FAILED, [*st.remarks, blocker]
            return
        todo = self._use_cache(todo)
        self.progress = ProgressReporter(len(self.states), self.model, use_widgets=self.options.use_widgets)
        for st in self.states:
            if st.status:
                self.progress.update(st.status, live=False)
        if todo:
            self._translate_phase(todo)
        self.progress.close()
        if not self.interrupted:
            self._analysis_phase()

    def _translate_phase(self, todo: list[TranslationRequest]) -> None:
        """Translate pending records with the GPU monitor running.

        Args:
            todo: Pending requests.
        """
        assert self.backend is not None
        self.monitor.start()
        started = time.perf_counter()
        try:
            self.backend.load()
            if self.backend.parallel_mode == "batch":
                _translate_batched(self.backend, todo, self.cfg.concurrency.hf_batch_size, self._finish)
            else:
                _translate_threads(self.backend, todo, self._workers(), self._finish)
        except KeyboardInterrupt:
            self.interrupted = True
            logger.warning("Interrupted — writing partial output")
        except Exception as exc:  # noqa: BLE001 - e.g. load() failure
            remark = redact(f"{type(exc).__name__}: {truncate(str(exc), 300)}")
            logger.error("Model failed before translating: %s", remark)
            for req in todo:
                st = self.states[req.index]
                if not st.status:
                    st.status, st.result = FAILED, TranslationResult(error=remark)
        finally:
            self.translation_seconds = time.perf_counter() - started
            self.monitor.stop()
        for req in todo:
            st = self.states[req.index]
            if not st.status:
                st.status = FAILED
                st.remarks.append("Interrupted by user before this record was processed (partial output)")

    def _analysis_phase(self) -> None:
        """Run the analysis helper for translation-only backends."""
        if self.backend is None or self.backend.produces_analysis:
            for req in self.requests:
                st = self.states[req.index]
                res = st.result
                if st.status == SUCCESS and res is not None and not st.cached and not res.has_analysis:
                    st.remarks.append("Partial: word segmentation unavailable (model returned none)")
            return
        word_translator = getattr(self.backend, "translate_words", None)
        helper = AnalysisHelper(self.cfg, self.prompts, self.openai_key, word_translator)
        self.analysis_desc = helper.describe()
        pending = [
            r for r in self.requests if self.states[r.index].status == SUCCESS and not self.states[r.index].cached
        ]
        if not pending:
            return
        bar = ProgressReporter(len(pending), f"{self.model} — word analysis", use_widgets=self.options.use_widgets)

        def work(req: TranslationRequest) -> Any:
            """Analyse one record.

            Args:
                req: Request.

            Returns:
                Analysis result.
            """
            return helper.analyse(req, self.states[req.index].result.translation)  # type: ignore[union-attr]

        def done(_pos: int, req: TranslationRequest, res: Any, exc: BaseException | None) -> None:
            """Apply one analysis result.

            Args:
                _pos: Position.
                req: Request.
                res: Analysis result.
                exc: Exception.
            """
            st = self.states[req.index]
            if exc is not None or res is None or res.error:
                reason = res.error if res is not None else f"{type(exc).__name__}: {exc}"
                st.remarks.append(f"Partial: word segmentation unavailable ({truncate(str(reason), 200)})")
            else:
                st.result.segmentation = res.segmentation  # type: ignore[union-attr]
                st.result.word_meanings = res.word_meanings  # type: ignore[union-attr]
                st.remarks.append(f"Analysis by {res.method}")
                st.remarks.extend(res.notes)
            if res is not None:
                st.helper_in, st.helper_out = res.input_tokens, res.output_tokens
            bar.update(SUCCESS if exc is None else FAILED)
            self._checkpoint(req, st)

        try:
            run_parallel(pending, work, max_workers=max(1, self.cfg.concurrency.openai_max_workers), on_result=done)
        except KeyboardInterrupt:
            self.interrupted = True
        bar.close()

    # ------------------------------------------------------------------ output
    def _rows(self) -> tuple[list[dict[str, Any]], list[list[Any]], list[list[Any]]]:
        """Assemble Translations, Detection_Audit and Token_Breakdown rows.

        Returns:
            ``(translation_rows, detection_rows, token_rows)``.
        """
        rows, det_rows, tok_rows = [], [], []
        for idx, rec in self.prep.records.iterrows():
            st, det = self.states[idx], self.prep.detections[idx]
            res = st.result or TranslationResult()
            remarks = list(st.remarks) + list(res.remarks)
            if res.error:
                remarks.append(res.error)
            if res.translator_notes:
                remarks.append(f"Note: {res.translator_notes}")
            ok = st.status == SUCCESS
            rows.append(
                {
                    "ID": rec["ID"],
                    "Source": rec["Source"],
                    "Verse": rec["Verse"],
                    "Shloka": rec["Shloka"],
                    "Detected_Language": det.language_name,
                    "Generated_Meaning": res.translation if ok else "",
                    "Word_Segmentation": " | ".join(res.segmentation) if ok else "",
                    "Word_Meanings": "\n".join(f"{w.word}: {w.meaning}" for w in res.word_meanings) if ok else "",
                    "Input_Tokens": res.input_tokens,
                    "Output_Tokens": res.output_tokens,
                    "Total_Tokens": res.total_tokens or (res.input_tokens + res.output_tokens),
                    "Processing_Status": st.status or FAILED,
                    "Remarks": "; ".join(dict.fromkeys(r for r in remarks if r)),
                }
            )
            det_rows.append(
                [
                    rec["ID"],
                    det.language_name,
                    det.code,
                    det.script,
                    det.confidence,
                    det.method,
                    rec["_source_file"],
                    rec["_source_sheet"],
                    rec["_row_number"],
                ]
            )
            tok_rows.append(
                [
                    rec["ID"],
                    res.input_tokens,
                    res.output_tokens,
                    rows[-1]["Total_Tokens"],
                    res.reasoning_tokens,
                    st.helper_in,
                    st.helper_out,
                    det.tiebreaker_input_tokens,
                    det.tiebreaker_output_tokens,
                    res.retries,
                    "yes" if st.cached else "",
                ]
            )
        return rows, det_rows, tok_rows

    def _summary(self, rows: list[dict[str, Any]], tok_rows: list[list[Any]], out_name: str) -> list[tuple[str, Any]]:
        """Build ``Run_Summary`` items.

        Args:
            rows: Translation rows.
            tok_rows: Token rows.
            out_name: Output file name.

        Returns:
            ``(item, value)`` pairs.
        """
        ended = now_in_tz(self.cfg.runtime.timezone)
        counts = {s: sum(1 for r in rows if r["Processing_Status"] == s) for s in (SUCCESS, SKIPPED, FAILED)}
        cached = sum(1 for st in self.states if st.cached)
        totals = [sum(int(r[i] or 0) for r in tok_rows) for i in range(1, 9)]
        has_gpu, gpu_name = gpu_present()
        snapshot = json.dumps(self.cfg.snapshot(), ensure_ascii=False)
        return [
            ("Model", self.model),
            ("Provider", self.provider),
            ("Output file", out_name),
            ("Start time", self.started_at.isoformat(timespec="seconds")),
            ("End time", ended.isoformat(timespec="seconds")),
            ("Duration (s)", round((ended - self.started_at).total_seconds(), 1)),
            ("Translation phase (s)", round(self.translation_seconds, 1)),
            ("Records total", len(rows)),
            ("Success", counts[SUCCESS]),
            ("Skipped", counts[SKIPPED]),
            ("Failed", counts[FAILED]),
            ("Cached (from checkpoint, not re-billed)", cached),
            ("Interrupted", "yes (partial output)" if self.interrupted else "no"),
            ("Input tokens (selected model)", totals[0]),
            ("Output tokens (selected model)", totals[1]),
            ("Total tokens (selected model)", totals[2]),
            ("Reasoning tokens", totals[3]),
            ("Analysis-helper tokens (in/out)", f"{totals[4]} / {totals[5]}"),
            ("LID tiebreaker tokens (in/out)", f"{totals[6]} / {totals[7]}"),
            (
                "Token note",
                "Token counts come only from provider usage fields or the model tokenizer; "
                "cached records show the tokens of the original (billed) call.",
            ),
            ("Prompt version", self.cfg.runtime.prompt_version),
            ("Language detection", self.prep.detector_info),
            ("Analysis helpers", self.analysis_desc or "n/a (model produces its own)"),
            ("Accepted inputs", ", ".join(self.prep.input_files)),
            ("GPU", gpu_name if has_gpu else "No GPU detected"),
            ("Runtime", runtime_info()),
            ("Library versions", json.dumps(library_versions())),
            ("Config snapshot (no secrets)", snapshot),
        ]

    def _write(self) -> ModelRunResult:
        """Write the workbook.

        Returns:
            :class:`ModelRunResult`.
        """
        rows, det_rows, tok_rows = self._rows()
        stats = RunStats(
            records_processed=self.live_count,
            output_tokens=sum(int(r[2]) for r, st in zip(tok_rows, self.states, strict=True) if not st.cached),
            total_tokens=sum(int(r[3]) for r, st in zip(tok_rows, self.states, strict=True) if not st.cached),
            latencies=self.latencies,
            duration_s=self.translation_seconds,
            provider=self.provider,
            batch_size_desc=(
                f"{self.cfg.concurrency.hf_batch_size} (length-sorted seq2seq batches)"
                if self.backend is not None and self.backend.parallel_mode == "batch"
                else f"1 per request; {self._workers()} concurrent requests"
            ),
        )
        gpu_rows, gpu_note = self.monitor.metrics_rows(stats, self.cfg.gpu_monitor.memory_data_rate_factor)
        out_path = unique_path(self.cfg.path(self.cfg.paths.output_folder), self.stem, "xlsx")
        data = WorkbookData(
            rows,
            gpu_rows,
            gpu_note,
            self._summary(rows, tok_rows, out_path.name),
            self.prep.validation,
            det_rows,
            tok_rows,
        )
        write_workbook(out_path, data)
        counts = {s: sum(1 for r in rows if r["Processing_Status"] == s) for s in (SUCCESS, SKIPPED, FAILED)}
        counts["Cached"] = sum(1 for st in self.states if st.cached)
        tokens = {
            "input": sum(r["Input_Tokens"] for r in rows),
            "output": sum(r["Output_Tokens"] for r in rows),
            "total": sum(r["Total_Tokens"] for r in rows),
        }
        logger.info("%s: %s → %s", self.model, counts, out_path)
        return ModelRunResult(self.model, out_path, None, counts, tokens, rows, self.interrupted)


# --------------------------------------------------------------------------- entry point


def resolve_models(cfg: AppConfig, choice: str) -> list[str]:
    """Expand the dropdown choice into model names.

    Args:
        cfg: Config.
        choice: Registry name or ``"ALL MODELS"``.

    Returns:
        Model names in registry order.

    Raises:
        KeyError: For an unknown model.
    """
    if choice == ALL_MODELS:
        return list(cfg.models)
    if choice not in cfg.models:
        raise KeyError(f"Unknown model {choice!r}; choose one of {list(cfg.models)} or {ALL_MODELS!r}")
    return [choice]


def run_pipeline(cfg: AppConfig, options: RunOptions) -> RunResult:
    """Run the full pipeline for one model or ALL MODELS.

    Args:
        cfg: Validated config.
        options: Run options.

    Returns:
        :class:`RunResult` (``ok=False`` with a message instead of raising for
        user-facing problems such as "no valid input files").

    Example:
        >>> res = run_pipeline(load_config(), RunOptions(model="gpt-4.1-mini", max_records=5))  # doctest: +SKIP
    """
    setup_logging(cfg)
    if options.max_records is not None:
        cfg.runtime.max_records = options.max_records
    if options.backend_factory is not None and options.model not in cfg.models and options.model != ALL_MODELS:
        models = [options.model]  # test backends need not be in the registry
    else:
        models = resolve_models(cfg, options.model)
    providers = {cfg.models[m].provider for m in models if m in cfg.models}
    openai_key = options.openai_key or get_secret(
        "OPENAI_API_KEY", interactive=options.interactive_secrets and "openai" in providers
    )
    hf_token = options.hf_token or get_secret(
        "HF_TOKEN", interactive=options.interactive_secrets and bool(providers & {"hf_seq2seq"})
    )
    prompts = PromptSet.load(cfg.path(cfg.paths.prompts_folder), cfg.runtime.prompt_version)
    prep = prepare_inputs(cfg, prompts, openai_key, cfg.runtime.max_records)
    if prep is None:
        folder = cfg.path(cfg.paths.input_folder)
        return RunResult(
            False,
            f"No input file passed validation in {folder}. Expected columns exactly "
            f"{cfg.schema_.expected_columns}. See the log for the per-file reasons.",
        )
    if prep.records.empty:
        return RunResult(False, "Accepted input files contain no records.", validation=prep.validation)
    results: list[ModelRunResult] = []
    outer = MultiModelProgress(models, use_widgets=options.use_widgets) if len(models) > 1 else None
    for model in models:
        runner = ModelRunner(cfg, prompts, prep, model, options, openai_key, hf_token)
        res = runner.run()
        results.append(res)
        if outer:
            outer.model_done(not res.interrupted)
        if res.interrupted:
            break
    if outer:
        outer.close()
    comparison = None
    if len(results) > 1:
        stem = build_stem("ALL-MODELS", cfg.runtime.timezone)
        comparison = write_comparison(
            unique_path(cfg.path(cfg.paths.output_folder), stem, "xlsx"),
            results[0].rows,
            {r.model: r.rows for r in results},
        )
    interrupted = any(r.interrupted for r in results)
    msg = "Interrupted — partial output written" if interrupted else "Completed"
    return RunResult(True, msg, results, comparison, prep.validation)
