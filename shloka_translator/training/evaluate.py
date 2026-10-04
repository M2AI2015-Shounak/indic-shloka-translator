"""sacreBLEU / chrF++ evaluation of seq2seq models on the held-out split."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from shloka_translator.models.base import TranslationRequest
from shloka_translator.training.data_prep import Pair

logger = logging.getLogger(__name__)


def corpus_scores(hypotheses: list[str], references: list[str]) -> dict[str, float]:
    """Compute corpus BLEU and chrF++.

    Args:
        hypotheses: System outputs.
        references: References (same length).

    Returns:
        ``{"bleu": ..., "chrf++": ..., "n": ...}``.

    Raises:
        ImportError: If sacrebleu is not installed.
    """
    import sacrebleu

    if not hypotheses:
        return {"bleu": 0.0, "chrf++": 0.0, "n": 0}
    bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2).score
    return {"bleu": round(bleu, 2), "chrf++": round(chrf, 2), "n": len(hypotheses)}


def evaluate_model(
    model_id_or_path: str, pairs: list[Pair], cfg: Any, hf_token: str | None = None, name: str = "eval"
) -> dict[str, dict[str, float]]:
    """Translate the test pairs and score them per language and overall.

    Args:
        model_id_or_path: HF id or local checkpoint.
        pairs: Held-out pairs.
        cfg: App config (generation + batch size).
        hf_token: HF token.
        name: Label for logs.

    Returns:
        ``{lang: {"bleu", "chrf++", "n"}, "ALL": {...}}``.
    """
    from shloka_translator.models.indictrans2_backend import IndicTrans2Backend

    backend = IndicTrans2Backend(name, model_id_or_path, cfg, hf_token=hf_token, gated="1B" in model_id_or_path)
    backend.load()
    reqs = [
        TranslationRequest(i, str(i), p.src, language_code=p.lang, language_name="", script="")
        for i, p in enumerate(pairs)
    ]
    hyps: list[str] = []
    bs = max(1, cfg.concurrency.hf_batch_size) * 4
    for start in range(0, len(reqs), bs):
        hyps.extend(r.translation for r in backend.translate_batch(reqs[start : start + bs]))
    backend.unload()
    by_lang: dict[str, tuple[list[str], list[str]]] = defaultdict(lambda: ([], []))
    for p, h in zip(pairs, hyps, strict=True):
        by_lang[p.lang][0].append(h)
        by_lang[p.lang][1].append(p.tgt)
    result = {lang: corpus_scores(h, r) for lang, (h, r) in sorted(by_lang.items())}
    result["ALL"] = corpus_scores(hyps, [p.tgt for p in pairs])
    logger.info("%s: %s", name, result["ALL"])
    return result


def comparison_table(base: dict[str, dict[str, float]], tuned: dict[str, dict[str, float]]) -> list[list[Any]]:
    """Side-by-side base vs fine-tuned scores.

    Args:
        base: Scores of the base model.
        tuned: Scores of the fine-tuned model.

    Returns:
        Rows ``[lang, n, base BLEU, tuned BLEU, ΔBLEU, base chrF++, tuned chrF++, ΔchrF++]``.

    Example:
        >>> comparison_table({"ALL": {"bleu": 1, "chrf++": 2, "n": 3}}, {"ALL": {"bleu": 2, "chrf++": 3, "n": 3}})[0][4]
        1
    """
    rows = []
    for lang in sorted(set(base) | set(tuned), key=lambda k: (k == "ALL", k)):
        b, t = base.get(lang, {}), tuned.get(lang, {})
        rows.append(
            [
                lang,
                t.get("n", b.get("n", 0)),
                b.get("bleu"),
                t.get("bleu"),
                round((t.get("bleu") or 0) - (b.get("bleu") or 0), 2),
                b.get("chrf++"),
                t.get("chrf++"),
                round((t.get("chrf++") or 0) - (b.get("chrf++") or 0), 2),
            ]
        )
    return rows
