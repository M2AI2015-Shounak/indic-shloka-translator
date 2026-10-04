"""YAML configuration loading and pydantic validation.

Every runtime setting of the translator lives in ``config/config.yaml``. This
module turns that file into a typed, validated :class:`AppConfig` object and
resolves all relative paths against the repository root.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

REPO_ROOT_MARKER = Path("shloka_translator") / "__init__.py"
DEFAULT_CONFIG_RELATIVE = Path("config") / "config.yaml"


class _Strict(BaseModel):
    """Base model that rejects unknown keys so typos in YAML fail loudly."""

    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Strict):
    """Folder locations, relative to the repository root unless absolute."""

    repo_root: str = "auto"
    input_folder: str = "Input_Folder"
    output_folder: str = "Output_Folder"
    log_folder: str = "logs"
    custom_model_dir: str = "models/custom"
    cache_folder: str = "Output_Folder/.cache"
    prompts_folder: str = "config/prompts"
    reference_folder: str = "reference"
    training_data_folder: str = "Training_Data"


class SchemaConfig(_Strict):
    """Expected input schema."""

    expected_columns: list[str] = Field(default_factory=lambda: ["ID", "Source", "Verse", "Shloka"])
    allow_extra_columns: bool = False
    column_name_match: Literal["exact_after_strip"] = "exact_after_strip"


class InputConfig(_Strict):
    """Input-folder scanning options."""

    recursive: bool = False
    extensions: list[str] = Field(default_factory=lambda: [".csv", ".xlsx", ".json", ".jsonl"])


class RuntimeConfig(_Strict):
    """General runtime behaviour."""

    timezone: str = "Asia/Kolkata"
    device: str = "cuda"
    allow_cpu_fallback: bool = False
    resume_from_checkpoint: bool = True
    max_records: int | None = None
    prompt_version: str = "v1"
    log_level: str = "INFO"


class LanguageDetectionConfig(_Strict):
    """Language-identification settings and the supported-language table."""

    min_confidence: float = 0.60
    use_llm_tiebreaker: bool = True
    tiebreaker_model: str = "gpt-4.1-mini"
    use_indiclid: bool = True
    indiclid_dir: str = "Output_Folder/.cache/IndicLID"
    use_fasttext_fallback: bool = True
    cache_file: str = "Output_Folder/.cache/langid_cache.json"
    supported_languages: dict[str, list[str]] = Field(default_factory=dict)


class ConcurrencyConfig(_Strict):
    """Parallelism, retry and timeout settings."""

    openai_max_workers: int = 8
    ollama_max_workers: int = 2
    hf_batch_size: int = 16
    max_retries: int = 5
    request_timeout_s: float = 180
    openai_requests_per_minute: int = 500


class GenerationConfig(_Strict):
    """Decoding parameters shared by all backends."""

    temperature: float = 0.0
    seed: int = 42
    max_output_tokens: int = 2048
    hf_num_beams: int = 5
    hf_max_length: int = 512
    ollama_num_ctx: int = 4096


class OpenAIConfig(_Strict):
    """OpenAI-specific switches."""

    use_responses_api: bool = False


class OllamaConfig(_Strict):
    """Local Ollama server settings."""

    host: str = "http://localhost:11434"
    keep_alive: str = "30m"
    context_headroom_gb: float = 1.5


class AnalysisConfig(_Strict):
    """Helper used for segmentation/word meanings of seq2seq models."""

    helper_model: str = "gpt-4.1-mini"
    helper_fallback_order: list[str] = Field(default_factory=lambda: ["gpt-4.1-mini", "qwen3:30b", "rule_based"])
    sanskrit_parser_timeout_s: float = 20


class GPUMonitorConfig(_Strict):
    """Background NVML sampler settings."""

    sample_interval_s: float = 1.0
    memory_data_rate_factor: float = 2


class TrainingConfig(_Strict):
    """Custom-model fine-tuning settings (see shloka_translator/training)."""

    base_model: str = "ai4bharat/indictrans2-indic-en-dist-200M"
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_pairs_per_language: int = 20000
    max_itihasa_pairs: int | None = None
    learning_rate: float = 3e-5
    warmup_steps: int = 500
    num_epochs: float = 3
    label_smoothing: float = 0.1
    per_device_batch_size: int = 16
    effective_batch_size: int = 64
    max_source_tokens: int = 512
    save_steps: int = 500
    eval_steps: int = 500
    early_stopping_patience: int = 3
    checkpoint_dir: str = "models/checkpoints"
    leakage_similarity: float = 0.9
    length_ratio_max: float = 3.0
    seed: int = 42
    use_bpcc: bool = True
    extra_hf_datasets: list[dict[str, Any]] = Field(default_factory=list)


class ModelSpec(_Strict):
    """One entry of the model registry (drives the dropdown)."""

    provider: Literal["openai", "ollama", "hf_seq2seq", "custom"]
    hf_id: str | None = None
    gated: bool = False
    approx_vram_gb: float | None = None
    disable_thinking: bool = False
    checkpoint: str | None = None


class AppConfig(_Strict):
    """Root configuration object.

    Attributes:
        repo_root: Absolute repository root (resolved, not from YAML directly).
    """

    paths: PathsConfig = Field(default_factory=PathsConfig)
    schema_: SchemaConfig = Field(default_factory=SchemaConfig, alias="schema")
    input: InputConfig = Field(default_factory=InputConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    language_detection: LanguageDetectionConfig = Field(default_factory=LanguageDetectionConfig)
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    gpu_monitor: GPUMonitorConfig = Field(default_factory=GPUMonitorConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    models: dict[str, ModelSpec] = Field(default_factory=dict)
    repo_root: Path = Field(default=Path("."), exclude=True)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("models")
    @classmethod
    def _models_not_empty(cls, value: dict[str, ModelSpec]) -> dict[str, ModelSpec]:
        """Ensure the model registry has at least one entry.

        Args:
            value: Parsed ``models`` mapping.

        Returns:
            The unchanged mapping.

        Raises:
            ValueError: If the registry is empty.
        """
        if not value:
            raise ValueError("config.models must list at least one model")
        return value

    def path(self, relative: str | Path) -> Path:
        """Resolve a configured path against the repository root.

        Args:
            relative: A path from the config (relative or absolute).

        Returns:
            Absolute :class:`Path`.

        Example:
            >>> cfg.path(cfg.paths.output_folder)  # doctest: +SKIP
            PosixPath('<repo_root>/Output_Folder')
        """
        p = Path(relative).expanduser()
        return p if p.is_absolute() else (self.repo_root / p).resolve()

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-friendly dump of the config (contains no secrets by design).

        Returns:
            Nested dict suitable for writing into the ``Run_Summary`` sheet.
        """
        data = self.model_dump(mode="json", by_alias=True, exclude={"repo_root"})
        data["repo_root"] = str(self.repo_root)
        return data


def find_repo_root(start: Path | None = None) -> Path:
    """Locate the repository root by walking up from ``start``.

    Args:
        start: Directory (or file) to start from; defaults to this module's folder.

    Returns:
        The first ancestor containing ``shloka_translator/__init__.py``.

    Raises:
        FileNotFoundError: If no ancestor contains the package.
    """
    here = (start or Path(__file__)).resolve()
    candidates = [here, *here.parents]
    for candidate in candidates:
        if (candidate / REPO_ROOT_MARKER).is_file():
            return candidate
    raise FileNotFoundError(f"Could not find repository root above {here}")


def load_config(config_path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> AppConfig:
    """Load, merge overrides into, and validate ``config.yaml``.

    Args:
        config_path: Path to the YAML file. Defaults to ``<repo>/config/config.yaml``.
        overrides: Optional nested dict merged on top of the YAML (e.g.
            ``{"runtime": {"max_records": 5}}``) before validation.

    Returns:
        A validated :class:`AppConfig` with ``repo_root`` resolved.

    Raises:
        FileNotFoundError: If the config file does not exist.
        pydantic.ValidationError: If the YAML contains invalid or unknown keys.

    Example:
        >>> cfg = load_config(overrides={"runtime": {"max_records": 5}})  # doctest: +SKIP
        >>> cfg.runtime.max_records  # doctest: +SKIP
        5
    """
    if config_path is None:
        root = find_repo_root()
        config_path = root / DEFAULT_CONFIG_RELATIVE
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if overrides:
        raw = deep_merge(raw, overrides)
    cfg = AppConfig.model_validate(raw)
    root_setting = cfg.paths.repo_root
    if root_setting in ("auto", "", None):
        cfg.repo_root = find_repo_root(config_path.parent)
    else:
        cfg.repo_root = Path(root_setting).expanduser().resolve()
    logger.debug("Loaded config from %s (repo_root=%s)", config_path, cfg.repo_root)
    return cfg


def deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``extra`` into a copy of ``base``.

    Args:
        base: Original mapping (not modified).
        extra: Mapping whose values win on conflict; nested dicts are merged.

    Returns:
        A new merged dict.

    Example:
        >>> deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}})
        {'a': {'b': 1, 'c': 3}}
    """
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
