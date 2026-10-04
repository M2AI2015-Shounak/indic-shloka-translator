"""Training-data collection, cleaning, leakage guard and stratified splitting.

Sources (all optional; each is licence-noted and logged in the model card):

* ``rahular/itihasa`` — ~93k Sanskrit–English pairs (Rāmāyaṇa, Mahābhārata;
  M. N. Dutt translations; Apache-2.0). Core in-domain set.
* ``ai4bharat/BPCC`` — gated; up to ``max_pairs_per_language`` human-annotated
  pairs per language (CC-BY-4.0 per the dataset card) so the model stays
  multilingual. File layout is discovered at run time; unavailable → skipped.
* ``training.extra_hf_datasets`` — any further public parallel sets you have
  verified (``{id, split, src_field, tgt_field, lang, licence}``).
* ``Training_Data/`` — your CSV/XLSX/JSON files with columns
  ``Source_Text, Language, English``.
"""

from __future__ import annotations

import csv
import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd

from shloka_translator.langid.languages import DEFAULT_SUPPORTED, LanguageTable
from shloka_translator.utils.text import normalize_text

logger = logging.getLogger(__name__)


@dataclass
class Pair:
    """One parallel training example.

    Attributes:
        src: Source text (Indic).
        lang: IndicTrans2 source code, e.g. ``san_Deva``.
        tgt: English reference.
        origin: Dataset of origin.
    """

    src: str
    lang: str
    tgt: str
    origin: str


@dataclass
class SourceReport:
    """Bookkeeping for one data source (goes into the manifest/model card).

    Attributes:
        name: Source name.
        licence: Licence note.
        pairs: Pairs loaded.
        status: ``loaded`` / ``skipped``.
        note: Details or skip reason.
    """

    name: str
    licence: str
    pairs: int = 0
    status: str = "skipped"
    note: str = ""


@dataclass
class PreparedData:
    """Output of :func:`prepare_training_data`.

    Attributes:
        train: Training pairs.
        valid: Validation pairs.
        test: Held-out test pairs.
        sources: Per-source reports.
        stats: Cleaning / leakage statistics.
    """

    train: list[Pair] = field(default_factory=list)
    valid: list[Pair] = field(default_factory=list)
    test: list[Pair] = field(default_factory=list)
    sources: list[SourceReport] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)


# --------------------------------------------------------------------------- loaders


def _rows_from_parquet_api(dataset_id: str, split: str | None, token: str | None) -> list[dict[str, Any]]:
    """Read a dataset through the Hub's auto-converted Parquet files (no loading script needed).

    Uses ``GET https://huggingface.co/api/datasets/<id>/parquet``, which lists the Parquet
    files the Hub generated for every config/split, downloads them and reads them with pandas.

    Args:
        dataset_id: Hub dataset id, e.g. ``"rahular/itihasa"``.
        split: Split to read, or ``None`` for all splits.
        token: Optional HF token (gated/private datasets).

    Returns:
        Rows as dicts.

    Raises:
        RuntimeError: If the Hub lists no Parquet files for the dataset/split.
    """
    import io
    import json
    import urllib.request

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    api = f"https://huggingface.co/api/datasets/{dataset_id}/parquet"
    with urllib.request.urlopen(urllib.request.Request(api, headers=headers), timeout=60) as resp:
        listing = json.loads(resp.read().decode("utf-8"))
    urls: list[str] = []
    for config_splits in listing.values():
        for split_name, files in config_splits.items():
            if split is None or split_name == split:
                urls.extend(files)
    if not urls:
        raise RuntimeError(f"No auto-converted Parquet files listed for {dataset_id} (split={split})")
    rows: list[dict[str, Any]] = []
    for url in urls:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=300) as resp:  # noqa: S310
            frame = pd.read_parquet(io.BytesIO(resp.read()))
        rows.extend(frame.to_dict(orient="records"))
    return rows


def load_hf_rows(dataset_id: str, split: str | None = None, token: str | None = None) -> tuple[list[dict], str]:
    """Load a Hub dataset robustly, whatever ``datasets`` version is installed.

    ``datasets`` ≥ 4.0 no longer runs dataset *loading scripts* (``<name>.py``) and raises
    ``RuntimeError: Dataset scripts are no longer supported``. Script-based datasets such as
    ``rahular/itihasa`` are therefore read, in order, via:

    1. ``load_dataset(id)`` — works for script-free datasets and for ``datasets`` < 4;
    2. ``load_dataset(id, revision="refs/convert/parquet")`` — the Hub's Parquet branch;
    3. the Hub Parquet API + pandas (no ``datasets`` involvement at all).

    Args:
        dataset_id: Hub dataset id.
        split: Split name, or ``None`` for all splits.
        token: Optional HF token.

    Returns:
        ``(rows, method)`` where ``method`` says which route worked.

    Raises:
        RuntimeError: If every route fails (message lists each failure).
    """
    errors: list[str] = []
    try:
        from datasets import load_dataset
    except ImportError as exc:
        load_dataset = None
        errors.append(f"datasets not installed ({exc})")
    if load_dataset is not None:
        for kwargs, label in (({}, "datasets"), ({"revision": "refs/convert/parquet"}, "datasets (parquet branch)")):
            try:
                ds = load_dataset(dataset_id, split=split, token=token, **kwargs)
                parts = [ds] if split else [ds[name] for name in ds]
                return [dict(ex) for part in parts for ex in part], label
            except Exception as exc:  # noqa: BLE001 - try the next route
                errors.append(f"{label}: {type(exc).__name__}: {str(exc)[:160]}")
    try:
        return _rows_from_parquet_api(dataset_id, split, token), "Hub parquet API"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"parquet API: {type(exc).__name__}: {str(exc)[:160]}")
    raise RuntimeError(" | ".join(errors))


def _translation_field(row: dict[str, Any], key: str) -> str:
    """Read ``row["translation"][key]`` whether it is a dict or a JSON string.

    Args:
        row: Dataset row.
        key: Language key (``"sn"`` / ``"en"``).

    Returns:
        Text value.
    """
    tr = row["translation"]
    if isinstance(tr, str):
        import json

        tr = json.loads(tr)
    return str(tr[key])


def load_itihasa(max_pairs: int | None = None, token: str | None = None) -> tuple[list[Pair], SourceReport]:
    """Load ``rahular/itihasa`` (all splits; ~93k Sanskrit–English pairs).

    Args:
        max_pairs: Optional cap.
        token: Optional HF token.

    Returns:
        ``(pairs, report)``.
    """
    report = SourceReport("rahular/itihasa", "Apache-2.0")
    try:
        rows, method = load_hf_rows("rahular/itihasa", None, token)
        pairs = [Pair(_translation_field(r, "sn"), "san_Deva", _translation_field(r, "en"), "itihasa") for r in rows]
        if max_pairs:
            pairs = pairs[:max_pairs]
        report.pairs, report.status, report.note = len(pairs), "loaded", f"via {method}"
        logger.info("itihasa: %d pairs via %s", len(pairs), method)
        return pairs, report
    except Exception as exc:  # noqa: BLE001
        report.note = f"{type(exc).__name__}: {exc}"
        logger.warning("itihasa unavailable: %s", report.note)
        return [], report


def _guess_columns(frame: pd.DataFrame, lang: str) -> tuple[str, str] | None:
    """Guess (indic_column, english_column) in a BPCC-like table.

    Args:
        frame: Table.
        lang: Target Indic code.

    Returns:
        Column names or ``None``.
    """
    cols = [str(c) for c in frame.columns]
    eng = next((c for c in cols if re.search(r"eng|^en$|english|src", c, re.I)), None)
    prefix = lang.split("_", maxsplit=1)[0]
    ind = next((c for c in cols if prefix in c.lower() or re.search(r"tgt|indic", c, re.I)), None)
    if eng and ind and eng != ind:
        return ind, eng
    if len(cols) == 2:
        return cols[1], cols[0]
    return None


def load_bpcc(languages: list[str], max_per_language: int, token: str | None) -> tuple[list[Pair], SourceReport]:
    """Subsample human-annotated BPCC pairs per language (best effort).

    The repository layout is discovered with ``list_repo_tree``; for each
    language the smallest human-annotated file (wiki/daily/seed) whose path
    contains the language code is read.

    Args:
        languages: IndicTrans2 codes to sample.
        max_per_language: Cap per language.
        token: HF token (BPCC is gated).

    Returns:
        ``(pairs, report)``.
    """
    report = SourceReport("ai4bharat/BPCC", "CC-BY-4.0 (see dataset card); gated")
    try:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi(token=token)
        files = [
            f
            for f in api.list_repo_tree("ai4bharat/BPCC", repo_type="dataset", recursive=True)
            if getattr(f, "size", None)
        ]
    except Exception as exc:  # noqa: BLE001
        report.note = f"cannot list BPCC ({type(exc).__name__}) — accept the dataset terms on HF and set HF_TOKEN"
        logger.warning(report.note)
        return [], report
    pairs: list[Pair] = []
    notes = []
    for lang in languages:
        cands = [
            f
            for f in files
            if lang in f.path
            and f.path.endswith((".csv", ".tsv", ".parquet", ".jsonl"))
            and re.search(r"wiki|daily|seed|human|bpcc-h", f.path, re.I)
            and f.size < 2_000_000_000
        ]
        if not cands:
            notes.append(f"{lang}: no file")
            continue
        chosen = sorted(cands, key=lambda f: f.size)[0]
        try:
            local = hf_hub_download("ai4bharat/BPCC", chosen.path, repo_type="dataset", token=token)
            if local.endswith(".parquet"):
                frame = pd.read_parquet(local)
            elif local.endswith(".jsonl"):
                frame = pd.read_json(local, lines=True, nrows=max_per_language * 2)
            else:
                frame = pd.read_csv(
                    local,
                    sep="\t" if local.endswith(".tsv") else ",",
                    nrows=max_per_language * 2,
                    dtype=str,
                    quoting=csv.QUOTE_MINIMAL,
                    on_bad_lines="skip",
                )
            colmap = _guess_columns(frame, lang)
            if colmap is None:
                notes.append(f"{lang}: unknown columns {list(frame.columns)[:4]}")
                continue
            sub = frame[[colmap[0], colmap[1]]].dropna().head(max_per_language)
            pairs.extend(Pair(str(a), lang, str(b), "bpcc") for a, b in sub.itertuples(index=False))
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{lang}: {type(exc).__name__}")
    report.pairs = len(pairs)
    report.status = "loaded" if pairs else "skipped"
    report.note = "; ".join(notes)
    return pairs, report


def load_extra_hf(specs: list[dict[str, Any]], token: str | None = None) -> tuple[list[Pair], list[SourceReport]]:
    """Load user-listed public HF parallel datasets.

    Args:
        specs: ``training.extra_hf_datasets`` entries.
        token: Optional HF token.

    Returns:
        ``(pairs, reports)``.
    """
    pairs: list[Pair] = []
    reports = []
    for spec in specs:
        rep = SourceReport(str(spec.get("id")), str(spec.get("licence", "verify before use")))
        try:
            rows, method = load_hf_rows(spec["id"], spec.get("split", "train"), token)
            src, tgt = spec["src_field"], spec["tgt_field"]
            got = [Pair(str(ex[src]), spec["lang"], str(ex[tgt]), spec["id"]) for ex in rows]
            pairs.extend(got)
            rep.pairs, rep.status, rep.note = len(got), "loaded", f"via {method}"
        except Exception as exc:  # noqa: BLE001
            rep.note = f"{type(exc).__name__}: {exc}"
        reports.append(rep)
    return pairs, reports


def load_user_training_data(folder: Path) -> tuple[list[Pair], SourceReport]:
    """Read ``Training_Data/`` files with ``Source_Text, Language, English``.

    ``Language`` may be a name (``Sanskrit``) or a code (``san_Deva``).

    Args:
        folder: Training data folder.

    Returns:
        ``(pairs, report)``.
    """
    report = SourceReport("Training_Data/ (user material)", "user-provided")
    table = LanguageTable.from_config(DEFAULT_SUPPORTED)
    codes = table.all_codes()
    pairs: list[Pair] = []
    if not folder.is_dir():
        return pairs, report
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in (".csv", ".xlsx", ".json", ".jsonl") or path.name.startswith((".", "~$")):
            continue
        try:
            if path.suffix.lower() == ".csv":
                frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            elif path.suffix.lower() == ".xlsx":
                frame = pd.read_excel(path, dtype=str, keep_default_na=False)
            else:
                frame = pd.read_json(path, lines=path.suffix.lower() == ".jsonl", dtype=str)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cannot read %s: %s", path.name, exc)
            continue
        if not {"Source_Text", "Language", "English"} <= set(frame.columns):
            logger.warning("%s skipped: needs columns Source_Text, Language, English", path.name)
            continue
        for src, lang, tgt in frame[["Source_Text", "Language", "English"]].itertuples(index=False):
            code = lang if lang in codes else table.code_for(str(lang).strip().title())
            if code:
                pairs.append(Pair(str(src), code, str(tgt), f"user:{path.name}"))
    report.pairs, report.status = len(pairs), ("loaded" if pairs else "skipped")
    return pairs, report


# --------------------------------------------------------------------------- cleaning


def clean_pairs(pairs: list[Pair], length_ratio_max: float = 3.0) -> tuple[list[Pair], dict[str, int]]:
    """NFC-normalise, drop empties, dedupe and filter implausible length ratios.

    The English/source character-length ratio must lie in
    ``[1/length_ratio_max, 2*length_ratio_max]`` (English glosses of terse
    Sanskrit verse are routinely 2–4× longer, hence the wider upper bound).

    Args:
        pairs: Raw pairs.
        length_ratio_max: Ratio bound.

    Returns:
        ``(clean_pairs, stats)``.
    """
    stats = {"raw": len(pairs), "empty": 0, "duplicate": 0, "length_ratio": 0}
    seen: set[tuple[str, str]] = set()
    out = []
    for p in pairs:
        src, tgt = normalize_text(p.src), normalize_text(p.tgt)
        if not src or not tgt:
            stats["empty"] += 1
            continue
        key = (src, tgt)
        if key in seen:
            stats["duplicate"] += 1
            continue
        seen.add(key)
        ratio = len(tgt) / max(1, len(src))
        if not (1 / length_ratio_max <= ratio <= 2 * length_ratio_max):
            stats["length_ratio"] += 1
            continue
        out.append(Pair(src, p.lang, tgt, p.origin))
    stats["clean"] = len(out)
    return out, stats


def _grams(text: str, n: int = 4) -> set[str]:
    """Character n-grams of whitespace-free text.

    Args:
        text: Text.
        n: Gram size.

    Returns:
        Set of n-grams.
    """
    t = re.sub(r"\s+", "", text)
    return {t[i : i + n] for i in range(max(1, len(t) - n + 1))}


def leakage_guard(pairs: list[Pair], protected: list[str], threshold: float = 0.9) -> tuple[list[Pair], int]:
    """Remove pairs whose source matches (exact or ≥ ``threshold`` similar) a protected shloka.

    A 4-gram inverted index finds candidates cheaply; ``difflib`` similarity
    confirms them.

    Args:
        pairs: Training pairs.
        protected: Shlokas from ``reference/`` and ``Input_Folder/``.
        threshold: Character-level similarity threshold.

    Returns:
        ``(kept_pairs, removed_count)``.

    Example:
        >>> kept, removed = leakage_guard([Pair("राम वनं गच्छति", "san_Deva", "x", "t")], ["राम वनं गच्छति"])
        >>> removed
        1
    """
    norm_protected = [re.sub(r"\s+", " ", normalize_text(t)) for t in protected if t and t.strip()]
    exact = set(norm_protected)
    index: dict[str, set[int]] = defaultdict(set)
    grams_of = [_grams(t) for t in norm_protected]
    for i, grams in enumerate(grams_of):
        for g in grams:
            index[g].add(i)
    kept, removed = [], 0
    for p in pairs:
        src = re.sub(r"\s+", " ", normalize_text(p.src))
        if src in exact:
            removed += 1
            continue
        grams = _grams(src)
        hits: dict[int, int] = defaultdict(int)
        for g in grams:
            for i in index.get(g, ()):
                hits[i] += 1
        leak = False
        for i, h in hits.items():
            overlap = h / max(1, min(len(grams), len(grams_of[i])))
            if overlap >= 0.5 and SequenceMatcher(None, src, norm_protected[i]).ratio() >= threshold:
                leak = True
                break
        if leak:
            removed += 1
        else:
            kept.append(p)
    return kept, removed


def stratified_split(
    pairs: list[Pair], seed: int = 42, valid: float = 0.025, test: float = 0.025
) -> tuple[list[Pair], list[Pair], list[Pair]]:
    """Split 95/2.5/2.5 per language.

    Args:
        pairs: Clean pairs.
        seed: Shuffle seed.
        valid: Validation share.
        test: Test share.

    Returns:
        ``(train, valid, test)``.
    """
    rng = random.Random(seed)
    by_lang: dict[str, list[Pair]] = defaultdict(list)
    for p in pairs:
        by_lang[p.lang].append(p)
    tr, va, te = [], [], []
    for lang in sorted(by_lang):
        group = by_lang[lang]
        rng.shuffle(group)
        n = len(group)
        n_te = max(1, int(n * test)) if n >= 40 else 0
        n_va = max(1, int(n * valid)) if n >= 40 else 0
        te.extend(group[:n_te])
        va.extend(group[n_te : n_te + n_va])
        tr.extend(group[n_te + n_va :])
    rng.shuffle(tr)
    return tr, va, te


def protected_texts(reference_dir: Path, input_dir: Path) -> list[str]:
    """Collect shlokas that must never appear in training data.

    Args:
        reference_dir: ``reference/``.
        input_dir: ``Input_Folder/``.

    Returns:
        Shloka strings.
    """
    from shloka_translator.io.loader import load_folder

    texts: list[str] = []
    for folder in (reference_dir, input_dir):
        if not folder.is_dir():
            continue
        for table in load_folder(folder, [".csv", ".xlsx", ".json", ".jsonl"]):
            if table.frame is not None and "Shloka" in [str(c).strip() for c in table.frame.columns]:
                col = next(c for c in table.frame.columns if str(c).strip() == "Shloka")
                texts.extend(str(v) for v in table.frame[col] if str(v).strip())
    return texts


def prepare_training_data(cfg: Any, hf_token: str | None, *, subset: int | None = None) -> PreparedData:
    """Collect, clean, leakage-filter and split all training material.

    Args:
        cfg: :class:`~shloka_translator.config.AppConfig`.
        hf_token: HF token (BPCC).
        subset: Optional cap on total pairs (quick runs / smoke tests).

    Returns:
        :class:`PreparedData`.
    """
    t = cfg.training
    data = PreparedData()
    pairs, rep = load_itihasa(t.max_itihasa_pairs, hf_token)
    data.sources.append(rep)
    if t.use_bpcc:
        langs = sorted(
            {c for codes in cfg.language_detection.supported_languages.values() for c in codes}
            or {c for codes in DEFAULT_SUPPORTED.values() for c in codes}
        )
        bp, rep = load_bpcc(langs, t.max_pairs_per_language, hf_token)
        pairs += bp
        data.sources.append(rep)
    extra, reps = load_extra_hf(t.extra_hf_datasets, hf_token)
    pairs += extra
    data.sources.extend(reps)
    user, rep = load_user_training_data(cfg.path(cfg.paths.training_data_folder))
    pairs += user
    data.sources.append(rep)
    clean, stats = clean_pairs(pairs, t.length_ratio_max)
    protected = protected_texts(cfg.path(cfg.paths.reference_folder), cfg.path(cfg.paths.input_folder))
    clean, removed = leakage_guard(clean, protected, t.leakage_similarity)
    stats["leakage_removed"] = removed
    logger.info("Leakage guard removed %d pair(s) overlapping reference/input shlokas", removed)
    if subset:
        random.Random(t.seed).shuffle(clean)
        clean = clean[:subset]
    data.train, data.valid, data.test = stratified_split(clean, t.seed)
    stats.update({"train": len(data.train), "valid": len(data.valid), "test": len(data.test)})
    data.stats = stats
    return data


def write_manifest(path: Path, data: PreparedData) -> None:
    """Write ``data_manifest.csv`` (one row per source + statistics).

    Args:
        path: Output CSV path.
        data: Prepared data.
    """
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "licence", "status", "pairs", "note"])
        for s in data.sources:
            w.writerow([s.name, s.licence, s.status, s.pairs, s.note])
        w.writerow([])
        w.writerow(["statistic", "value"])
        for k, v in data.stats.items():
            w.writerow([k, v])
