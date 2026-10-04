"""Colab control panel: model dropdown (grouped by provider, with availability
badges), concurrency slider, max-records box, resume checkbox and Run button."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from shloka_translator.config import AppConfig, load_config
from shloka_translator.models.registry import ALL_MODELS, PROVIDER_LABELS, ModelStatus, model_statuses
from shloka_translator.pipeline.orchestrator import RunOptions, RunResult, run_pipeline
from shloka_translator.prompting.templates import PromptSet
from shloka_translator.secrets import get_secret

logger = logging.getLogger(__name__)

LAST_RESULT: dict[str, RunResult | None] = {"result": None}
PROVIDER_ORDER = ["openai", "hf_seq2seq", "ollama", "custom"]


def dropdown_options(statuses: list[ModelStatus]) -> list[tuple[str, str]]:
    """Build dropdown ``(label, value)`` pairs grouped by provider.

    ipywidgets has no option groups, so each label is prefixed with its
    provider group and ordered group by group.

    Args:
        statuses: Availability of each registry model.

    Returns:
        Options, ending with ``ALL MODELS``.

    Example:
        >>> dropdown_options([ModelStatus("gpt-4.1", "openai", True, "ready")])[0]
        ('OpenAI (API) · ✓ gpt-4.1', 'gpt-4.1')
    """
    options: list[tuple[str, str]] = []
    for provider in PROVIDER_ORDER + sorted({s.provider for s in statuses} - set(PROVIDER_ORDER)):
        for s in statuses:
            if s.provider == provider:
                options.append((f"{PROVIDER_LABELS.get(provider, provider)} · {s.badge}", s.name))
    options.append((f"★ {ALL_MODELS} (run every model sequentially + comparison workbook)", ALL_MODELS))
    return options


def build_panel(config_path: str | Path | None = None) -> Any:
    """Create and display the control panel.

    Args:
        config_path: Optional path to ``config.yaml``.

    Returns:
        The ipywidgets container (already displayed).

    Raises:
        ImportError: If ipywidgets / IPython are not installed.
    """
    import ipywidgets as w
    from IPython.display import display

    cfg = load_config(config_path)
    prompts = PromptSet.load(cfg.path(cfg.paths.prompts_folder), cfg.runtime.prompt_version)
    openai_key = get_secret("OPENAI_API_KEY")
    hf_token = get_secret("HF_TOKEN")
    status_html = w.HTML("<i>Checking model availability…</i>")
    model_dd = w.Dropdown(description="Model", layout=w.Layout(width="95%"), style={"description_width": "80px"})
    concurrency = w.IntSlider(
        value=cfg.concurrency.openai_max_workers,
        min=1,
        max=32,
        description="Concurrency",
        style={"description_width": "80px"},
        tooltip="Parallel requests for OpenAI/Ollama (seq2seq models always batch on GPU)",
    )
    max_records = w.IntText(
        value=0,
        description="Max records",
        style={"description_width": "80px"},
        tooltip="0 = all records; e.g. 5 for a smoke test",
    )
    resume = w.Checkbox(value=cfg.runtime.resume_from_checkpoint, description="Resume from checkpoint")
    accept_spill = w.Checkbox(value=False, description="Run Ollama model even if VRAM is insufficient (very slow)")
    run_btn = w.Button(description="Run", button_style="success", icon="play")
    refresh_btn = w.Button(description="Refresh availability", icon="refresh")
    ollama_btn = w.Button(description="Set up Ollama for selection", icon="download")
    out = w.Output(layout=w.Layout(border="1px solid #ccc", padding="6px"))
    statuses: dict[str, ModelStatus] = {}

    def refresh(_btn: Any = None) -> None:
        """Re-check availability and rebuild the dropdown.

        Args:
            _btn: Button (unused).
        """
        sts = model_statuses(cfg, prompts, openai_key=openai_key, hf_token=hf_token)
        statuses.clear()
        statuses.update({s.name: s for s in sts})
        current = model_dd.value
        model_dd.options = dropdown_options(sts)
        if current in statuses or current == ALL_MODELS:
            model_dd.value = current
        ready = sum(1 for s in sts if s.available)
        auto = sum(1 for s in sts if s.setup_on_run)
        status_html.value = (
            f"<b>{ready}/{len(sts)}</b> models ready · <b>{auto}</b> set up automatically when you click Run (⬇) · "
            "⚠ entries need an action from you (key, GPU, licence, training)."
        )

    def setup_ollama(_btn: Any = None) -> None:
        """Install/start Ollama and pull the selected model(s).

        Args:
            _btn: Button (unused).
        """
        from shloka_translator.models.ollama_backend import run_setup_script

        chosen = model_dd.value
        names = [n for n, s in cfg.models.items() if s.provider == "ollama"] if chosen == ALL_MODELS else [chosen]
        names = [n for n in names if cfg.models.get(n) and cfg.models[n].provider == "ollama"]
        with out:
            if not names:
                print("Selected model is not an Ollama model — nothing to set up.")
                return
            print(f"Setting up Ollama and pulling: {', '.join(names)} (this can take several minutes)…")
            code = run_setup_script(cfg.repo_root, names, cfg)
            print("Ollama ready." if code == 0 else f"Ollama setup exited with code {code} — see output above.")
        refresh()

    def run(_btn: Any = None) -> None:
        """Run the pipeline for the selected model.

        Args:
            _btn: Button (unused).
        """
        run_btn.disabled = True
        out.clear_output()
        with out:
            try:
                choice = model_dd.value
                needs_ollama = (choice == ALL_MODELS and any(s.provider == "ollama" for s in cfg.models.values())) or (
                    choice in cfg.models
                    and cfg.models[choice].provider == "ollama"
                    and not statuses.get(choice, ModelStatus("", "", False, "")).available
                )
                providers = (
                    {s.provider for s in cfg.models.values()}
                    if choice == ALL_MODELS
                    else {cfg.models[choice].provider}
                    if choice in cfg.models
                    else set()
                )
                if providers & {"hf_seq2seq", "custom"}:
                    from shloka_translator.utils.packages import ensure_packages

                    missing = ensure_packages(["IndicTransToolkit"])
                    if missing:
                        print(f"✗ IndicTransToolkit could not be installed: {missing['IndicTransToolkit']}")
                if needs_ollama:
                    setup_ollama()
                options = RunOptions(
                    model=choice,
                    max_records=max_records.value or None,
                    resume=resume.value,
                    concurrency=concurrency.value,
                    use_widgets=True,
                    confirm=lambda msg: bool(accept_spill.value) or _refuse(msg),
                    openai_key=openai_key,
                    hf_token=hf_token,
                )
                result = run_pipeline(load_config(config_path), options)
                LAST_RESULT["result"] = result
                _print_result(result)
            except Exception as exc:
                logger.exception("Run failed")
                print(f"❌ Run failed: {type(exc).__name__}: {exc}")
            finally:
                run_btn.disabled = False

    refresh_btn.on_click(refresh)
    ollama_btn.on_click(setup_ollama)
    run_btn.on_click(run)
    refresh()
    panel = w.VBox(
        [
            w.HTML("<h3>Indic Shloka → English translator</h3>"),
            status_html,
            model_dd,
            w.HBox([concurrency, max_records]),
            w.HBox([resume, accept_spill]),
            w.HBox([run_btn, refresh_btn, ollama_btn]),
            out,
        ]
    )
    display(panel)
    return panel


def _refuse(message: str) -> bool:
    """Print why an Ollama run was not started and return ``False``.

    Args:
        message: VRAM pre-flight message.

    Returns:
        Always ``False``.
    """
    print(f"⚠ {message}\nTick 'Run Ollama model even if VRAM is insufficient' to proceed anyway.")
    return False


def _print_result(result: RunResult) -> None:
    """Print a short summary of a run in the notebook.

    Args:
        result: Pipeline result.
    """
    if not result.ok:
        print(f"⚠ {result.message}")
        for row in result.validation:
            print("   ", row)
        return
    print(f"✅ {result.message}")
    for m in result.models:
        print(f"• {m.model}: {m.counts} | tokens {m.tokens} \n  → {m.output_path}\n  log: {m.log_path}")
    if result.comparison_path:
        print(f"• Comparison workbook → {result.comparison_path}")


def latest_result() -> RunResult | None:
    """Return the result of the last Run click.

    Returns:
        :class:`RunResult` or ``None``.
    """
    return LAST_RESULT["result"]


def current_config(config_path: str | Path | None = None) -> AppConfig:
    """Convenience loader for notebook cells.

    Args:
        config_path: Optional config path.

    Returns:
        Loaded config.
    """
    return load_config(config_path)
