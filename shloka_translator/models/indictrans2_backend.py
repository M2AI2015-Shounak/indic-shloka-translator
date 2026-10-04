"""IndicTrans2 (AI4Bharat) Indic→English seq2seq backend.

* ``ai4bharat/indictrans2-indic-en-1B`` is **gated**: accept the licence on its
  Hugging Face page and provide ``HF_TOKEN``.
* Pre/post-processing via ``IndicTransToolkit.IndicProcessor``.
* fp16 on T4/V100, bf16 on Ampere and newer; batched beam search (beam 5),
  ``max_length`` 512; length-sorted batches; single process, no GPU threads.
* Sanskrit in a non-Devanagari or romanised script is transliterated to
  Devanagari first; long inputs are split at ``।``/``॥``/sentence boundaries.
* Token counts are real tokenizer counts (non-padding input ids; generated ids
  excluding padding/special tokens).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shloka_translator.config import AppConfig
from shloka_translator.langid.script_detector import PARALLEL_INDIC, to_devanagari
from shloka_translator.models.base import TranslationRequest, TranslationResult, TranslatorBackend
from shloka_translator.secrets import redact
from shloka_translator.utils.text import truncate

logger = logging.getLogger(__name__)

INDICTRANS2_CODES = {
    "asm_Beng",
    "ben_Beng",
    "brx_Deva",
    "doi_Deva",
    "gom_Deva",
    "guj_Gujr",
    "hin_Deva",
    "kan_Knda",
    "kas_Arab",
    "kas_Deva",
    "mai_Deva",
    "mal_Mlym",
    "mar_Deva",
    "mni_Beng",
    "mni_Mtei",
    "npi_Deva",
    "ory_Orya",
    "pan_Guru",
    "san_Deva",
    "sat_Olck",
    "snd_Arab",
    "snd_Deva",
    "tam_Taml",
    "tel_Telu",
    "urd_Arab",
}
_SPLIT_RE = re.compile(r"(?<=[।॥.?!|])\s+")
_gated_cache: dict[str, tuple[bool, str]] = {}


def transliterate_to_devanagari(text: str, script: str) -> tuple[str, str]:
    """Transliterate Sanskrit from another script into Devanagari.

    Tries ``indic_transliteration`` (sanscript), then ``aksharamukha``, then the
    built-in Unicode-offset mapping (parallel Indic scripts only).

    Args:
        text: Source text.
        script: Script name (``"Bengali"``, ``"Telugu"``, ``"Latin"``...).

    Returns:
        ``(devanagari_text, method)``.

    Raises:
        ValueError: If no method can handle the script.
    """
    sanscript_names = {
        "Bengali": "BENGALI",
        "Gurmukhi": "GURMUKHI",
        "Gujarati": "GUJARATI",
        "Odia": "ORIYA",
        "Tamil": "TAMIL",
        "Telugu": "TELUGU",
        "Kannada": "KANNADA",
        "Malayalam": "MALAYALAM",
        "Latin": "IAST",
        "Grantha": "GRANTHA",
    }
    try:
        from indic_transliteration import sanscript  # type: ignore[import-not-found]

        scheme_name = sanscript_names.get(script)
        if script == "Latin" and not re.search(r"[āīūṛṝḷṃṁḥśṣṇṭḍñṅ]", text, flags=re.IGNORECASE):
            scheme_name = "ITRANS"
        if scheme_name and hasattr(sanscript, scheme_name):
            return sanscript.transliterate(
                text, getattr(sanscript, scheme_name), sanscript.DEVANAGARI
            ), f"indic_transliteration ({scheme_name}→Devanagari)"
    except Exception:
        logger.debug("indic_transliteration unavailable/failed", exc_info=True)
    try:
        from aksharamukha import transliterate as ak  # type: ignore[import-not-found]

        src = {"Latin": "IAST", "Odia": "Oriya"}.get(script, script)
        return ak.process(src, "Devanagari", text), f"aksharamukha ({src}→Devanagari)"
    except Exception:
        logger.debug("aksharamukha unavailable/failed", exc_info=True)
    if script in PARALLEL_INDIC:
        return to_devanagari(text, script), "Unicode offset mapping (approximate)"
    raise ValueError(f"Cannot transliterate {script} to Devanagari (install indic-transliteration)")


def split_for_length(text: str, count_tokens: Any, max_tokens: int) -> list[str]:
    """Split text at danda / sentence boundaries so each chunk fits ``max_tokens``.

    Args:
        text: Input text.
        count_tokens: Callable returning a token count for a string.
        max_tokens: Budget per chunk.

    Returns:
        List of chunks (``[text]`` when it already fits).

    Example:
        >>> split_for_length("a । b ॥ c", lambda s: len(s.split()), 2)
        ['a ।', 'b ॥', 'c']
    """
    if count_tokens(text) <= max_tokens:
        return [text]
    pieces = [p for p in _SPLIT_RE.split(text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip() if current else piece
        if count_tokens(candidate) <= max_tokens:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if count_tokens(piece) <= max_tokens:
            current = piece
        else:  # a single sentence is too long: split on words
            words, buf = piece.split(), ""
            for w in words:
                cand = f"{buf} {w}".strip()
                if buf and count_tokens(cand) > max_tokens:
                    chunks.append(buf)
                    buf = w
                else:
                    buf = cand
            current = buf
    if current:
        chunks.append(current)
    return chunks


@dataclass
class _Chunk:
    """One unit of seq2seq work.

    Attributes:
        item: Index into the batch's items.
        order: Chunk order within the item.
        text: Text to translate.
        src_lang: IndicTrans2 source code.
        translation: Output (filled later).
        in_tokens: Input token count.
        out_tokens: Output token count.
    """

    item: int
    order: int
    text: str
    src_lang: str
    translation: str = ""
    in_tokens: int = 0
    out_tokens: int = 0


@dataclass
class _Prepared:
    """Per-item preparation outcome.

    Attributes:
        chunks: Work units.
        remarks: Notes (transliteration, chunking).
        error: Preparation error.
    """

    chunks: list[_Chunk] = field(default_factory=list)
    remarks: list[str] = field(default_factory=list)
    error: str | None = None


class IndicTrans2Backend(TranslatorBackend):
    """Hugging Face seq2seq backend for IndicTrans2 (and its fine-tunes)."""

    provider = "hf_seq2seq"
    produces_analysis = False
    parallel_mode = "batch"
    runs_on_local_gpu = True

    def __init__(
        self, name: str, model_id_or_path: str, cfg: AppConfig, hf_token: str | None = None, gated: bool = False
    ) -> None:
        """Create the backend (weights are loaded in :meth:`load`).

        Args:
            name: Registry name.
            model_id_or_path: HF repo id or local checkpoint directory.
            cfg: Config.
            hf_token: Hugging Face token (gated models).
            gated: Whether the HF repo is gated.
        """
        self.name = name
        self.model_id = model_id_or_path
        self.cfg = cfg
        self.hf_token = hf_token
        self.gated = gated
        self.model: Any = None
        self.tokenizer: Any = None
        self.processor: Any = None
        self.device = "cpu"
        self.dtype_name = "float32"

    # ---------------------------------------------------------------- environment
    @staticmethod
    def _import_processor() -> Any:
        """Import ``IndicProcessor`` from whichever module path the installed toolkit uses.

        Returns:
            The ``IndicProcessor`` class.

        Raises:
            ImportError: If IndicTransToolkit is not installed.
        """
        errors = []
        for mod in ("IndicTransToolkit.processor", "IndicTransToolkit", "IndicTransToolkit.IndicTransToolkit"):
            try:
                module = __import__(mod, fromlist=["IndicProcessor"])
                return module.IndicProcessor
            except (ImportError, AttributeError) as exc:
                errors.append(f"{mod}: {exc}")
        raise ImportError("IndicTransToolkit not installed (" + "; ".join(errors) + ")")

    def _gpu_ok(self) -> tuple[bool, str]:
        """Check CUDA availability against ``allow_cpu_fallback``.

        Returns:
            ``(ok, reason)``.
        """
        try:
            import torch
        except ImportError:
            return False, "torch not installed"
        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
        if self.cfg.runtime.allow_cpu_fallback:
            return True, "CPU fallback (slow)"
        return False, "No GPU — switch to a GPU runtime or set runtime.allow_cpu_fallback: true"

    def _gated_access(self) -> tuple[bool, str]:
        """Check HF access to a gated repo (cached per model id).

        Returns:
            ``(ok, reason)``.
        """
        if not self.gated or Path(self.model_id).exists():
            return True, ""
        if self.model_id in _gated_cache:
            return _gated_cache[self.model_id]
        if not self.hf_token:
            outcome = (False, "HF_TOKEN not set (gated model — accept the licence on Hugging Face first)")
        else:
            try:
                from huggingface_hub import auth_check

                auth_check(self.model_id, token=self.hf_token)
                outcome = (True, "")
            except ImportError:
                outcome = (True, "")
            except Exception as exc:  # noqa: BLE001
                outcome = (
                    False,
                    f"No access to {self.model_id} — accept the licence on its HF page ({type(exc).__name__})",
                )
        _gated_cache[self.model_id] = outcome
        return outcome

    def is_available(self) -> tuple[bool, str]:
        """Check packages, GPU and gated-model access.

        Returns:
            ``(ok, reason)``.
        """
        try:
            import transformers  # noqa: F401
        except ImportError:
            return False, "transformers not installed — re-run the Install cell"
        try:
            self._import_processor()
        except ImportError:
            return False, "IndicTransToolkit not installed — re-run the Install cell (pip install IndicTransToolkit)"
        ok, reason = self._gpu_ok()
        if not ok:
            return False, reason
        ok, why = self._gated_access()
        if not ok:
            return False, why
        return True, f"ready ({reason})"

    # ---------------------------------------------------------------- lifecycle
    def load(self) -> None:
        """Load tokenizer, model and processor onto the GPU.

        Raises:
            RuntimeError: If no GPU is present and CPU fallback is disabled.
        """
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        ok, reason = self._gpu_ok()
        if not ok:
            raise RuntimeError(reason)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            major, _ = torch.cuda.get_device_capability(0)
            dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            dtype = torch.float32
        self.dtype_name = str(dtype).replace("torch.", "")
        token = self.hf_token or None
        logger.info("Loading %s (%s) on %s", self.model_id, self.dtype_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True, token=token)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id, trust_remote_code=True, torch_dtype=dtype, token=token
        ).to(self.device)
        self.model.eval()
        self.processor = self._import_processor()(inference=True)

    def unload(self) -> None:
        """Free model weights and VRAM."""
        self.model = None
        self.tokenizer = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def max_workers(self) -> int:
        """Seq2seq models never use threads on the GPU.

        Returns:
            Always 1.
        """
        return 1

    # ---------------------------------------------------------------- work
    def _count_tokens(self, text: str) -> int:
        """Count tokenizer tokens for raw text.

        Args:
            text: Input text.

        Returns:
            Token count.
        """
        return len(self.tokenizer(text, add_special_tokens=True)["input_ids"])

    def prepare(self, req: TranslationRequest) -> _Prepared:
        """Transliterate / chunk one request.

        Args:
            req: Request.

        Returns:
            :class:`_Prepared`.
        """
        prep = _Prepared()
        text, code = req.text, req.language_code
        if req.language_name == "Sanskrit" and req.script not in ("", "Devanagari"):
            try:
                text, method = transliterate_to_devanagari(text, req.script)
                prep.remarks.append(f"Transliterated {req.script}→Devanagari via {method}")
                code = "san_Deva"
            except ValueError as exc:
                prep.error = str(exc)
                return prep
        elif req.script == "Latin":
            prep.error = f"Romanised {req.language_name} is not supported by IndicTrans2 — use an LLM model"
            return prep
        if code not in INDICTRANS2_CODES:
            prep.error = f"Language code {code or '?'} not supported by IndicTrans2"
            return prep
        budget = self.cfg.generation.hf_max_length - 16
        chunks = split_for_length(text, self._count_tokens, budget)
        if len(chunks) > 1:
            prep.remarks.append(f"Chunked into {len(chunks)} segments")
        prep.chunks = [_Chunk(-1, i, c, code) for i, c in enumerate(chunks)]
        return prep

    def _generate(self, chunks: list[_Chunk]) -> None:
        """Translate chunks of one source language in one batch (fills them in place).

        Args:
            chunks: Chunks sharing ``src_lang``.
        """
        import torch

        src_lang = chunks[0].src_lang
        gen = self.cfg.generation
        batch = self.processor.preprocess_batch([c.text for c in chunks], src_lang=src_lang, tgt_lang="eng_Latn")
        enc = self.tokenizer(
            batch,
            truncation=True,
            padding="longest",
            max_length=gen.hf_max_length,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)
        with torch.inference_mode():
            out = self.model.generate(
                **enc,
                use_cache=True,
                min_length=0,
                max_length=gen.hf_max_length,
                num_beams=gen.hf_num_beams,
                num_return_sequences=1,
            )
        special = {
            i
            for i in (
                self.tokenizer.pad_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.bos_token_id,
                getattr(self.model.config, "decoder_start_token_id", None),
            )
            if i is not None
        }
        in_counts = enc["attention_mask"].sum(dim=1).tolist()
        out_counts = [sum(1 for t in row.tolist() if t not in special) for row in out]
        decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded = self.processor.postprocess_batch(decoded, lang="eng_Latn")
        for c, text, n_in, n_out in zip(chunks, decoded, in_counts, out_counts, strict=True):
            c.translation, c.in_tokens, c.out_tokens = text.strip(), int(n_in), int(n_out)

    def _run_batches(self, chunks: list[_Chunk]) -> dict[int, str]:
        """Length-sorted batching per language with OOM back-off.

        Args:
            chunks: All chunks of this call.

        Returns:
            Mapping chunk-id → error for chunks that failed.
        """
        import torch

        errors: dict[int, str] = {}
        by_lang: dict[str, list[_Chunk]] = {}
        for c in chunks:
            by_lang.setdefault(c.src_lang, []).append(c)
        for group in by_lang.values():
            group.sort(key=lambda c: len(c.text))
            size = max(1, self.cfg.concurrency.hf_batch_size)
            pos = 0
            while pos < len(group):
                part = group[pos : pos + size]
                try:
                    self._generate(part)
                    pos += len(part)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if size == 1:
                        errors[id(part[0])] = "CUDA out of memory"
                        pos += 1
                    else:
                        size = max(1, size // 2)
                        logger.warning("OOM — reducing batch size to %d", size)
                except Exception as exc:  # noqa: BLE001
                    if len(part) > 1:  # isolate the bad record(s): retry this batch one chunk at a time
                        logger.warning(
                            "Batch failed (%s); retrying its %d items individually", type(exc).__name__, len(part)
                        )
                        for c in part:
                            try:
                                self._generate([c])
                            except Exception as one_exc:  # noqa: BLE001
                                errors[id(c)] = redact(f"{type(one_exc).__name__}: {truncate(str(one_exc), 300)}")
                    else:
                        errors[id(part[0])] = redact(f"{type(exc).__name__}: {truncate(str(exc), 300)}")
                    pos += len(part)
        return errors

    def translate_batch(self, items: list[TranslationRequest]) -> list[TranslationResult]:
        """Translate a batch of records with length-sorted GPU batching.

        Per-record latency is the wall time of this call divided equally among
        its records (batched inference has no per-record timing).

        Args:
            items: Requests.

        Returns:
            Results aligned with ``items``.
        """
        start = time.perf_counter()
        results = [TranslationResult() for _ in items]
        try:
            self.load()
        except Exception as exc:  # noqa: BLE001
            for r in results:
                r.error = redact(f"{type(exc).__name__}: {truncate(str(exc), 300)}")
            return results
        all_chunks: list[_Chunk] = []
        for idx, req in enumerate(items):
            try:
                prep = self.prepare(req)
            except Exception as exc:  # noqa: BLE001
                prep = _Prepared(error=f"{type(exc).__name__}: {truncate(str(exc), 300)}")
            results[idx].remarks.extend(prep.remarks)
            if prep.error:
                results[idx].error = prep.error
                continue
            for c in prep.chunks:
                c.item = idx
                all_chunks.append(c)
        errors = self._run_batches(all_chunks) if all_chunks else {}
        per_item: dict[int, list[_Chunk]] = {}
        for c in all_chunks:
            per_item.setdefault(c.item, []).append(c)
        for idx, chunks in per_item.items():
            res = results[idx]
            failed = [errors[id(c)] for c in chunks if id(c) in errors]
            chunks.sort(key=lambda c: c.order)
            res.input_tokens = sum(c.in_tokens for c in chunks)
            res.output_tokens = sum(c.out_tokens for c in chunks)
            res.total_tokens = res.input_tokens + res.output_tokens
            if failed:
                res.error = failed[0]
            else:
                res.translation = " ".join(c.translation for c in chunks).strip()
                if not res.translation:
                    res.error = "Empty translation from model"
        elapsed = time.perf_counter() - start
        for r in results:
            r.latency_s = elapsed / max(1, len(items))
        return results

    def translate_words(self, words: list[str], src_lang: str) -> list[str]:
        """Word-level translation (used by the rule-based analysis path).

        Args:
            words: Segments to gloss.
            src_lang: IndicTrans2 source code.

        Returns:
            One English gloss per word (``""`` on failure).
        """
        if not words:
            return []
        self.load()
        chunks = [_Chunk(i, 0, w, src_lang) for i, w in enumerate(words)]
        errors = self._run_batches(chunks)
        return ["" if id(c) in errors else c.translation for c in chunks]
