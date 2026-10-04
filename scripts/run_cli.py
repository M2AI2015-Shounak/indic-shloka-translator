"""Thin command-line wrapper around the pipeline.

Usage::

    python scripts/run_cli.py --model gpt-4.1 [--max-records 5] [--no-resume]
    python scripts/run_cli.py --model "ALL MODELS"
    python scripts/run_cli.py --list-models
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shloka_translator.config import load_config  # noqa: E402
from shloka_translator.pipeline.orchestrator import RunOptions, run_pipeline  # noqa: E402

logger = logging.getLogger("run_cli")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed namespace.
    """
    p = argparse.ArgumentParser(description="Translate shlokas in Input_Folder/ to English.")
    p.add_argument("--model", help='Registry model name, or "ALL MODELS"')
    p.add_argument("--max-records", type=int, default=None, help="Only the first N records (smoke test)")
    p.add_argument("--no-resume", action="store_true", help="Ignore the checkpoint cache")
    p.add_argument("--concurrency", type=int, default=None, help="Override worker count")
    p.add_argument("--config", default=None, help="Path to config.yaml")
    p.add_argument("--list-models", action="store_true", help="Show registry models with availability")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list.

    Returns:
        Process exit code (0 ok, 1 nothing processed, 2 usage error, 130 interrupted).
    """
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.list_models:
        from shloka_translator.models.registry import model_statuses
        from shloka_translator.prompting.templates import PromptSet
        from shloka_translator.secrets import get_secret

        prompts = PromptSet.load(cfg.path(cfg.paths.prompts_folder), cfg.runtime.prompt_version)
        for s in model_statuses(cfg, prompts, openai_key=get_secret("OPENAI_API_KEY"), hf_token=get_secret("HF_TOKEN")):
            print(f"{s.provider:11s} {s.badge}")
        return 0
    if not args.model:
        print("error: --model is required (or use --list-models)", file=sys.stderr)
        return 2
    options = RunOptions(
        model=args.model,
        max_records=args.max_records,
        resume=not args.no_resume,
        concurrency=args.concurrency,
        use_widgets=False,
        interactive_secrets=sys.stdin.isatty(),
    )
    result = run_pipeline(cfg, options)
    print(result.message)
    for m in result.models:
        print(f"{m.model}: {m.counts} tokens={m.tokens}\n  output: {m.output_path}\n  log:    {m.log_path}")
    if result.comparison_path:
        print(f"comparison: {result.comparison_path}")
    if not result.ok:
        return 1
    return 130 if any(m.interrupted for m in result.models) else 0


if __name__ == "__main__":
    sys.exit(main())
