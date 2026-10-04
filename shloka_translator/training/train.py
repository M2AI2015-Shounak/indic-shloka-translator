"""Fine-tune IndicTrans2 Indic→English for shlokas (``custom-shloka-mt``).

* Base: ``ai4bharat/indictrans2-indic-en-dist-200M`` full fine-tune (fits a T4
  with fp16 + gradient accumulation), or ``indictrans2-indic-en-1B`` + LoRA
  (``training.use_lora: true``) on L4/A100.
* ``Seq2SeqTrainer``: lr 3e-5, warmup 500, 2–3 epochs, label smoothing 0.1,
  gradient accumulation to an effective batch of 64, fp16/bf16 by GPU,
  early stopping on validation chrF++.
* Checkpoints go to Drive every ``save_steps`` and training resumes from the
  latest one after a Colab disconnect.
* The best model is saved to ``models/custom/custom-shloka-mt_<date>_<time>/``
  with ``model_card.md``, ``metrics.json``, ``training_config.yaml`` and
  ``data_manifest.csv``.
"""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from typing import Any

import yaml

from shloka_translator.output.naming import build_stem
from shloka_translator.training.data_prep import Pair, PreparedData, prepare_training_data, write_manifest

logger = logging.getLogger(__name__)
MODEL_NAME = "custom-shloka-mt"


def _processor() -> Any:
    """Instantiate IndicProcessor for training (non-inference mode).

    Returns:
        ``IndicProcessor(inference=False)``.
    """
    from shloka_translator.models.indictrans2_backend import IndicTrans2Backend

    return IndicTrans2Backend._import_processor()(inference=False)


def _preprocess(pairs: list[Pair], processor: Any) -> tuple[list[str], list[str]]:
    """Apply IndicProcessor tagging to sources and targets, language by language.

    Args:
        pairs: Pairs.
        processor: IndicProcessor.

    Returns:
        ``(sources, targets)`` aligned with ``pairs``.
    """
    src_out: list[str] = [""] * len(pairs)
    tgt_out: list[str] = [""] * len(pairs)
    by_lang: dict[str, list[int]] = {}
    for i, p in enumerate(pairs):
        by_lang.setdefault(p.lang, []).append(i)
    for lang, idxs in by_lang.items():
        srcs = processor.preprocess_batch(
            [pairs[i].src for i in idxs], src_lang=lang, tgt_lang="eng_Latn", is_target=False
        )
        tgts = processor.preprocess_batch(
            [pairs[i].tgt for i in idxs], src_lang="eng_Latn", tgt_lang=lang, is_target=True
        )
        for j, i in enumerate(idxs):
            src_out[i], tgt_out[i] = srcs[j], tgts[j]
    return src_out, tgt_out


def _tokenize(pairs: list[Pair], tokenizer: Any, processor: Any, max_len: int) -> tuple[Any, int]:
    """Tokenise pairs into a ``datasets.Dataset``, dropping pairs over ``max_len`` tokens.

    Args:
        pairs: Pairs.
        tokenizer: IndicTrans2 tokenizer.
        processor: IndicProcessor.
        max_len: Token limit for source and target.

    Returns:
        ``(dataset, dropped_count)``.
    """
    from datasets import Dataset

    srcs, tgts = _preprocess(pairs, processor)
    enc = tokenizer(srcs, truncation=False, padding=False)
    try:
        lab = tokenizer(text_target=tgts, truncation=False, padding=False)
    except TypeError:
        with tokenizer.as_target_tokenizer():
            lab = tokenizer(tgts, truncation=False, padding=False)
    rows = {"input_ids": [], "attention_mask": [], "labels": []}
    dropped = 0
    for ids, mask, labels in zip(enc["input_ids"], enc["attention_mask"], lab["input_ids"], strict=True):
        if len(ids) > max_len or len(labels) > max_len:
            dropped += 1
            continue
        rows["input_ids"].append(ids)
        rows["attention_mask"].append(mask)
        rows["labels"].append(labels)
    return Dataset.from_dict(rows), dropped


def _training_args(cfg: Any, output_dir: Path, max_steps: int | None) -> Any:
    """Build ``Seq2SeqTrainingArguments`` compatible with the installed transformers.

    Args:
        cfg: App config.
        output_dir: Checkpoint directory (on Drive).
        max_steps: Optional step cap (quick runs).

    Returns:
        Training arguments.
    """
    import torch
    from transformers import Seq2SeqTrainingArguments

    t = cfg.training
    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    fp16 = torch.cuda.is_available() and not bf16
    accum = max(1, t.effective_batch_size // max(1, t.per_device_batch_size))
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": t.learning_rate,
        "warmup_steps": t.warmup_steps,
        "num_train_epochs": t.num_epochs,
        "label_smoothing_factor": t.label_smoothing,
        "per_device_train_batch_size": t.per_device_batch_size,
        "per_device_eval_batch_size": t.per_device_batch_size,
        "gradient_accumulation_steps": accum,
        "fp16": fp16,
        "bf16": bf16,
        "save_steps": t.save_steps,
        "eval_steps": t.eval_steps,
        "save_strategy": "steps",
        "logging_steps": 50,
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "chrf",
        "greater_is_better": True,
        "predict_with_generate": True,
        "generation_max_length": 256,
        "generation_num_beams": 1,
        "seed": t.seed,
        "report_to": [],
        "dataloader_num_workers": 2,
        "max_steps": max_steps or -1,
    }
    params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    kwargs["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "steps"
    return Seq2SeqTrainingArguments(**{k: v for k, v in kwargs.items() if k in params})


def _metrics_fn(tokenizer: Any) -> Any:
    """Create ``compute_metrics`` returning chrF++ and BLEU.

    Args:
        tokenizer: Tokenizer for decoding.

    Returns:
        Callable for the trainer.
    """
    import numpy as np
    import sacrebleu

    def compute(eval_pred: Any) -> dict[str, float]:
        """Decode predictions and score them.

        Args:
            eval_pred: ``(predictions, label_ids)``.

        Returns:
            ``{"chrf": ..., "bleu": ...}``.
        """
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
        labels = np.where(labels < 0, tokenizer.pad_token_id, labels)
        hyp = [s.strip() for s in tokenizer.batch_decode(preds, skip_special_tokens=True)]
        ref = [s.strip() for s in tokenizer.batch_decode(labels, skip_special_tokens=True)]
        return {
            "chrf": sacrebleu.corpus_chrf(hyp, [ref], word_order=2).score,
            "bleu": sacrebleu.corpus_bleu(hyp, [ref]).score,
        }

    return compute


def _latest_checkpoint(output_dir: Path) -> str | None:
    """Find the newest ``checkpoint-*`` folder for resuming.

    Args:
        output_dir: Trainer output dir.

    Returns:
        Path string or ``None``.
    """
    if not output_dir.is_dir():
        return None
    cps = [d for d in output_dir.glob("checkpoint-*") if d.is_dir()]
    if not cps:
        return None
    return str(max(cps, key=lambda d: int(d.name.split("-")[-1]) if d.name.split("-")[-1].isdigit() else -1))


def _load_model(cfg: Any, hf_token: str | None) -> tuple[Any, Any]:
    """Load base model + tokenizer (optionally wrapped with LoRA).

    Args:
        cfg: App config.
        hf_token: HF token.

    Returns:
        ``(model, tokenizer)``.
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    t = cfg.training
    tokenizer = AutoTokenizer.from_pretrained(t.base_model, trust_remote_code=True, token=hf_token)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        t.base_model, trust_remote_code=True, token=hf_token, attn_implementation="eager"
    )
    if t.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        lora = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=t.lora_r,
            lora_alpha=t.lora_alpha,
            lora_dropout=t.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()
    return model, tokenizer


def write_model_card(out_dir: Path, cfg: Any, data: PreparedData, metrics: dict[str, Any]) -> None:
    """Write ``model_card.md``.

    Args:
        out_dir: Model directory.
        cfg: App config.
        data: Prepared data (sources and stats).
        metrics: Evaluation results.
    """
    lines = [
        f"# {MODEL_NAME}",
        "",
        f"Fine-tuned from `{cfg.training.base_model}`"
        + (" with LoRA (merged)" if cfg.training.use_lora else " (full fine-tune)")
        + ".",
        "",
        "## Training data",
        "",
        "| Source | Licence | Status | Pairs | Note |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {s.name} | {s.licence} | {s.status} | {s.pairs} | {s.note[:120]} |" for s in data.sources]
    lines += ["", "## Data statistics", ""] + [f"- {k}: {v}" for k, v in data.stats.items()]
    lines += [
        "",
        "Leakage guard: training pairs matching (exact or ≥ "
        f"{cfg.training.leakage_similarity:.0%} character similarity) any shloka in `reference/` or "
        "`Input_Folder/` were removed.",
        "",
        "## Evaluation (held-out test split, sacreBLEU)",
        "",
    ]
    rows = metrics.get("comparison", [])
    if rows:
        lines += [
            "| Lang | n | BLEU base | BLEU tuned | ΔBLEU | chrF++ base | chrF++ tuned | ΔchrF++ |",
            "|---|---|---|---|---|---|---|---|",
        ]
        lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    lines += [
        "",
        "These scores compare the base and fine-tuned IndicTrans2 models only. No claim is made "
        "about quality relative to GPT or other LLM backends unless measured on the same test set.",
    ]
    (out_dir / "model_card.md").write_text("\n".join(lines), encoding="utf-8")


def train(
    cfg: Any,
    hf_token: str | None = None,
    *,
    subset: int | None = None,
    max_steps: int | None = None,
    evaluate_base: bool = True,
) -> Path:
    """Run data prep, fine-tuning, evaluation and export.

    Args:
        cfg: App config.
        hf_token: HF token (gated base model / BPCC).
        subset: Cap on total pairs (quick runs).
        max_steps: Cap on optimisation steps (quick runs).
        evaluate_base: Also score the base model for comparison.

    Returns:
        Path of the exported model directory in ``models/custom``.

    Raises:
        RuntimeError: If no training data could be collected.
    """
    from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback, Seq2SeqTrainer

    from shloka_translator.training.evaluate import comparison_table, evaluate_model

    data = prepare_training_data(cfg, hf_token, subset=subset)
    if not data.train:
        raise RuntimeError(f"No training data collected: {[(s.name, s.note) for s in data.sources]}")
    model, tokenizer = _load_model(cfg, hf_token)
    processor = _processor()
    max_len = cfg.training.max_source_tokens
    train_ds, d1 = _tokenize(data.train, tokenizer, processor, max_len)
    valid_ds, d2 = _tokenize(data.valid or data.train[:200], tokenizer, processor, max_len)
    data.stats["dropped_over_max_tokens"] = d1 + d2
    run_dir = cfg.path(cfg.training.checkpoint_dir) / "trainer"
    run_dir.mkdir(parents=True, exist_ok=True)
    trainer = Seq2SeqTrainer(
        model=model,
        args=_training_args(cfg, run_dir, max_steps),
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest"),
        compute_metrics=_metrics_fn(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.training.early_stopping_patience)],
    )
    resume = _latest_checkpoint(run_dir)
    if resume:
        logger.info("Resuming from %s", resume)
    trainer.train(resume_from_checkpoint=resume)
    best = trainer.model
    if cfg.training.use_lora:
        best = best.merge_and_unload()
    out_dir = cfg.path(cfg.paths.custom_model_dir) / build_stem(MODEL_NAME, cfg.runtime.timezone)
    out_dir.mkdir(parents=True, exist_ok=True)
    best.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    test = data.test or data.valid
    tuned = evaluate_model(str(out_dir), test, cfg, hf_token, "fine-tuned") if test else {}
    base = evaluate_model(cfg.training.base_model, test, cfg, hf_token, "base") if (test and evaluate_base) else {}
    metrics = {
        "fine_tuned": tuned,
        "base": base,
        "comparison": comparison_table(base, tuned) if base else [],
        "stats": data.stats,
        "best_checkpoint": trainer.state.best_model_checkpoint,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "training_config.yaml").write_text(
        yaml.safe_dump(cfg.training.model_dump(), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    write_manifest(out_dir / "data_manifest.csv", data)
    write_model_card(out_dir, cfg, data, metrics)
    logger.info("Saved %s", out_dir)
    return out_dir
