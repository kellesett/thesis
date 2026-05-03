#!/usr/bin/env python3
"""
metrics/factuality/main.py
Factuality metrics for generated surveys (group B, metric B.1).

Requires pre-computed Claimify output:
  results/scores/<dataset_id>_<model_id>_claims/

Fails with a clear error if the claims cache is absent.

For each survey:
  1. Load atomic claims from Claimify cache.
  2. LLM judge classifies each claim as category A / B / C / D.
  3. AlignScore checks citation support for each claim against the
     source paper text (abstract fetched from SurGE corpus.json).
  4. Compute conditional citation correctness per category:
       CitCorrect_k = |{a : φ(a)=k ∧ Support(a)=1}| / |{a : φ(a)=k}|

Output per survey: results/scores/<dataset_id>_<model_id>_factuality_<run_id>/<survey_id>.json
Summary:           .../summary.csv

Usage (inside Docker):
    python metrics/factuality/main.py --dataset SurGE --model perplexity_dr
"""
import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import ijson
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

logger = logging.getLogger(__name__)

ROOT   = Path(__file__).parent.parent.parent
CONFIG = Path(__file__).parent / "config.yaml"
sys.path.insert(0, str(ROOT))

from src.log_setup import setup_logging
from metrics.utils import (
    make_client, load_config, TokenCounter,
    resolve_claims_dir, load_generation_files,
    check_and_load_cache, write_summary_csv, llm_json_call,
)
from metrics.factuality.prompts import CATEGORY_SYSTEM, CATEGORY_PROMPT
from metrics.factuality.claim_scope import (
    resolve_claim_scope,
    _find_source_offset, _paragraph_span, _section_scope_spans,
)
from metrics.factuality.evidence_fetcher import (
    _ref_key as _evidence_key,
    fetch_evidence, load_abstract_cache, save_abstract_cache,
    prepare_key_evidence,
)
from metrics.factuality.sources_io import load_gen_sources


# ── Corpus loading ────────────────────────────────────────────────────────────

def build_corpus_index(dataset: str) -> dict[str, str]:
    """
    Stream corpus.json once and build {paper_id → abstract} index.
    Returns dict keyed by paper_id (string).
    """
    corpus_path = ROOT / "datasets" / dataset / "corpus.json"
    if not corpus_path.exists():
        logger.warning("corpus.json not found at %s", corpus_path)
        return {}

    logger.info("Indexing corpus.json (streaming) ...")
    index: dict[str, str] = {}
    with open(corpus_path, "rb") as f:
        for paper in ijson.items(f, "item"):
            pid   = str(paper.get("doc_id", paper.get("paperId", "")))
            title = paper.get("title", "")
            abstr = paper.get("abstract", "")
            index[pid] = f"{title}\n\n{abstr}".strip()
    logger.info("Corpus indexed: %d papers", len(index))
    return index


# ── AlignScore cost meter ─────────────────────────────────────────────────────
# Populated during process_survey to feed the top-k / abstract-cascade
# calibration appendix (see theory/main/main_full.tex §app:topk_calibration).
# Reset per survey at the start of process_survey; written into the survey
# result JSON as the `cost` block. Resume-from-cache keeps the meter at zero —
# that's by design: cost is meaningful only for fresh runs.

_cost_meter: dict[str, int] = {
    "n_alignscore_calls":  0,  # number of alignscore_model.score() invocations
    "n_alignscore_pairs":  0,  # number of (premise, claim) pairs scored
    "n_chunks_evaluated":  0,  # full-text chunks sent to AlignScore (top-k path)
    "n_abstract_skip":     0,  # (claim, ref) units closed by abstract cascade
}


def _reset_cost_meter() -> None:
    for k in _cost_meter:
        _cost_meter[k] = 0


def _add_cost(key: str, value: int) -> None:
    _cost_meter[key] = _cost_meter.get(key, 0) + int(value)


def _get_cost_meter() -> dict[str, int]:
    return dict(_cost_meter)


# ── AlignScore loader ─────────────────────────────────────────────────────────

def load_alignscore(cfg: dict):
    """Load the AlignScore NLI model.

    Preflight-checks that the checkpoint file actually exists so we fail fast
    with a clear message instead of crashing inside the AlignScore constructor
    (which surfaces a less readable error deep in torch-lightning loading).

    ``alignscore_device`` is configurable (default ``cpu``) to make a future
    GPU switch a one-line config change — CLAUDE.md §11 explicitly flags the
    old hardcoded ``cpu`` as a scaling wart.
    """
    ckpt = cfg["alignscore_ckpt"]
    if not Path(ckpt).is_file():
        raise FileNotFoundError(
            f"AlignScore checkpoint not found: {ckpt}\n"
            f"Run `make download-metric-models` to fetch it, or set "
            f"`alignscore_enabled: false` in metrics/factuality/config.yaml "
            f"to skip the support-verification step."
        )
    from alignscore import AlignScore
    device = cfg.get("alignscore_device", "cpu")
    logger.info("Loading AlignScore: %s (device=%s)", ckpt, device)
    return AlignScore(
        model="roberta-large",
        batch_size=cfg.get("alignscore_batch_size", 16),
        device=device,
        ckpt_path=ckpt,
        evaluation_mode="nli_sp",
        # Mute AlignScore's internal "Evaluating" tqdm — our own align_bar
        # already tracks progress at the pair level (per_ref) / claim level
        # (concat), so the nested bar is redundant noise.
        verbose=False,
    )


# ── Claim categorization ──────────────────────────────────────────────────────


def classify_claim(
    claim: str,
    source_context: str,
    client: OpenAI,
    model: str,
    max_retries: int,
    disable_reasoning: bool = False,
    provider: str | None = None,
    token_counter: TokenCounter | None = None,
    reasoning_effort: str | None = None,
) -> dict:
    """Classify a claim into one of four categories (A/B/C/D).

    Args:
        claim: Atomic claim text.
        source_context: Paper abstract or source context.
        client: OpenAI client instance.
        model: Judge model name.
        max_retries: Maximum retry attempts for API calls.
        disable_reasoning: Disable thinking tokens (Qwen3, DeepSeek-R1).
        provider: Optional OpenRouter provider override.
        token_counter: Optional thread-safe accumulator for tracking API token usage.
        reasoning_effort: "low"|"medium"|"high" for reasoning-capable models (takes priority over disable_reasoning).

    Returns:
        Dict with "category" (A/B/C/D), "confidence", and optional "error".
    """
    prompt = CATEGORY_PROMPT.format(
        claim=claim[:600],
        source_context=source_context[:800] if source_context else "Not available",
    )

    def _fallback(exc):
        return {"category": "A", "confidence": "low", "error": str(exc)}

    parsed = llm_json_call(
        client, model,
        messages=[
            {"role": "system", "content": CATEGORY_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        max_retries=max_retries,
        provider=provider,
        disable_reasoning=disable_reasoning,
        reasoning_effort=reasoning_effort,
        token_counter=token_counter,
        on_failure=_fallback,
    )
    if "error" in parsed:
        return parsed  # fallback dict from on_failure
    cat = parsed.get("category", "").upper()
    if cat not in {"A", "B", "C", "D"}:
        return {"category": "A", "confidence": "low", "error": f"Unknown category: {cat}"}
    return {"category": cat, "confidence": parsed.get("confidence", "low")}


# ── Support verification via AlignScore ───────────────────────────────────────


_ALIGN_CHUNK_SIZE = 50   # pairs per score() call for visible tqdm progress


def _alignscore_concat(
    categorized: list[dict],
    per_claim_evidence: list[list[tuple[dict, str]]],
    alignscore_model,
    threshold: float,
    bar = None,
) -> None:
    """Concat-mode AlignScore: one premise per claim, one score per claim.

    All evidence texts for a claim are joined by ``\\n\\n\\n`` into a single
    premise. AlignScore's ``nli_sp`` chunks the premise internally and
    max-pools over chunks, so concatenation is mathematically ~equivalent
    to ``max`` of per-ref scores — while being 3-5× faster because the
    batch overhead drops from N calls to 1. Attribution (which ref "won")
    is lost, which is the acknowledged trade-off.

    Mutates ``categorized`` in place: each entry gets ``supported``,
    ``alignscore``, ``evidence_refs`` (list of ref idx'es that contributed).
    """
    # Collect only the claims that actually have evidence; skip the rest.
    batch_claims:    list[str] = []
    batch_premises:  list[str] = []
    indices:         list[int] = []

    for i, evs in enumerate(per_claim_evidence):
        if not evs:
            categorized[i]["supported"]     = None
            categorized[i]["alignscore"]    = None
            categorized[i]["evidence_refs"] = []
            continue
        # Triple-newline separator — nli_sp sentence-tokenizer treats this
        # as a hard break, so chunks won't bridge across reference abstracts.
        batch_claims.append(categorized[i]["claim"])
        batch_premises.append("\n\n\n".join(text for _, text in evs))
        indices.append(i)

    # Chunked scoring so the outer tqdm bar moves as work progresses —
    # calling score() once with all N pairs would freeze the bar until
    # the whole batch returns. 50 pairs per chunk is a middle ground
    # between progress granularity and per-call overhead.
    scores: list[float] = []
    for start in range(0, len(batch_premises), _ALIGN_CHUNK_SIZE):
        chunk_p = batch_premises[start:start + _ALIGN_CHUNK_SIZE]
        chunk_c = batch_claims[start:start + _ALIGN_CHUNK_SIZE]
        chunk_scores = alignscore_model.score(contexts=chunk_p, claims=chunk_c)
        scores.extend(chunk_scores)
        if bar is not None:
            bar.update(len(chunk_c))

    for i, score in zip(indices, scores):
        evs = per_claim_evidence[i]
        categorized[i]["supported"]     = bool(score >= threshold)
        categorized[i]["alignscore"]    = round(float(score), 4)
        categorized[i]["evidence_refs"] = [r["idx"] for r, _ in evs]
    # Claims without evidence still advance the bar so the total reads right.
    if bar is not None:
        no_ev = sum(1 for evs in per_claim_evidence if not evs)
        if no_ev:
            bar.update(no_ev)


def _alignscore_per_ref(
    categorized: list[dict],
    per_claim_evidence: list[list[tuple[dict, str]]],
    alignscore_model,
    threshold: float,
    bar = None,
) -> None:
    """Per-ref AlignScore: one score per (claim, ref) pair.

    Each claim's Support is ``max`` over its per-ref scores; the argmax ref
    is recorded for attribution (``alignscore_best``). Full per-ref break-
    down lands in ``alignscore_per_ref`` — useful for debugging
    "why did this claim pass / fail" in the viewer.

    Cost: N_claims × avg_refs_per_claim AlignScore pairs. On SurGE_reference
    ~9.6k claims × ~11 median refs = ~100k pairs — 3-5× the concat mode.
    Use when attribution is worth the extra compute (research debugging,
    not bulk runs).
    """
    # Flatten all (claim, premise) pairs into one batch so AlignScore's
    # internal batching amortises per-call overhead.
    flat_claims:   list[str] = []
    flat_premises: list[str] = []
    slices:        list[tuple[int, int]] = []
    for i, evs in enumerate(per_claim_evidence):
        s = len(flat_claims)
        for r, text in evs:
            flat_claims.append(categorized[i]["claim"])
            flat_premises.append(text)
        slices.append((s, len(flat_claims)))

    # Chunked scoring for visible progress (see _alignscore_concat).
    flat_scores: list[float] = []
    for start in range(0, len(flat_premises), _ALIGN_CHUNK_SIZE):
        chunk_p = flat_premises[start:start + _ALIGN_CHUNK_SIZE]
        chunk_c = flat_claims[start:start + _ALIGN_CHUNK_SIZE]
        chunk_scores = alignscore_model.score(contexts=chunk_p, claims=chunk_c)
        flat_scores.extend(chunk_scores)
        if bar is not None:
            bar.update(len(chunk_c))

    for i, (s_start, s_end) in enumerate(slices):
        evs = per_claim_evidence[i]
        if s_start == s_end:
            categorized[i]["supported"]          = None
            categorized[i]["alignscore"]         = None
            categorized[i]["evidence_refs"]      = []
            categorized[i]["alignscore_per_ref"] = []
            categorized[i]["alignscore_best"]    = None
            continue

        scores_i = list(flat_scores[s_start:s_end])
        max_score = max(scores_i)
        best_local = scores_i.index(max_score)
        best_ref_idx = evs[best_local][0]["idx"]

        categorized[i]["supported"]     = bool(max_score >= threshold)
        categorized[i]["alignscore"]    = round(float(max_score), 4)
        categorized[i]["evidence_refs"] = [r["idx"] for r, _ in evs]
        categorized[i]["alignscore_per_ref"] = [
            {"ref_idx": r["idx"], "score": round(float(s), 4)}
            for (r, _), s in zip(evs, scores_i)
        ]
        categorized[i]["alignscore_best"] = {
            "ref_idx": best_ref_idx,
            "score":   round(float(max_score), 4),
        }


def _score_alignscore_pairs(
    premises: list[str],
    claims: list[str],
    alignscore_model,
    bar = None,
) -> list[float]:
    """Score premise/claim pairs with chunked calls for visible progress."""
    scores: list[float] = []
    for start in range(0, len(premises), _ALIGN_CHUNK_SIZE):
        chunk_p = premises[start:start + _ALIGN_CHUNK_SIZE]
        chunk_c = claims[start:start + _ALIGN_CHUNK_SIZE]
        chunk_scores = alignscore_model.score(contexts=chunk_p, claims=chunk_c)
        _add_cost("n_alignscore_calls", 1)
        _add_cost("n_alignscore_pairs", len(chunk_c))
        scores.extend(float(s) for s in chunk_scores)
        if bar is not None:
            bar.update(len(chunk_c))
    return scores


def _set_topk_bar_postfix(bar, chunks_to_align: int, abstract_skipped: int) -> None:
    if bar is not None:
        bar.set_postfix_str(
            f"chunks={chunks_to_align}  abs-skip={abstract_skipped}",
            refresh=False,
        )


def _alignscore_style_chunks(text: str, chunk_words: int) -> list[str]:
    """Split text into ~AlignScore-sized chunks (sentence groups, ~350 words)."""
    from nltk.tokenize import sent_tokenize

    sents = sent_tokenize(text)
    sents = sents or [""]
    n_chunk = len(text.strip().split()) // max(chunk_words, 1) + 1
    n_chunk = max(len(sents) // n_chunk, 1)
    chunks: list[str] = []
    for i in range(0, len(sents), n_chunk):
        chunk = " ".join(sents[i:i + n_chunk]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _load_chunk_embedder(cfg: dict):
    """Load the semantic embedder used for cheap full-text chunk preselection."""
    from sentence_transformers import SentenceTransformer

    model_name = str(cfg.get("full_text_chunk_embedder") or "datasets/gte-large-en-v1.5")
    model_path = Path(model_name)
    if not model_path.is_absolute() and (ROOT / model_path).exists():
        model_name = str(ROOT / model_path)
    device = cfg.get("full_text_chunk_embedder_device", "cpu")
    logger.info("Loading full-text chunk embedder: %s (device=%s)", model_name, device)
    try:
        return SentenceTransformer(model_name, device=device, trust_remote_code=True)
    except TypeError:
        return SentenceTransformer(model_name, device=device)


def _select_top_chunks_by_similarity(
    *,
    categorized: list[dict],
    fallback_pairs: list[dict],
    chunk_embedder,
    top_k: int,
    chunk_words: int,
    batch_size: int,
) -> dict[int, str]:
    """Return selected full-text premises keyed by fallback-pair index."""
    import numpy as np

    ref_chunks: dict[str, list[str]] = {}
    for pair in fallback_pairs:
        key = pair["ref_key"]
        if key not in ref_chunks:
            ref_chunks[key] = _alignscore_style_chunks(pair["full_text"], chunk_words)

    flat_chunks: list[str] = []
    chunk_ranges: dict[str, tuple[int, int]] = {}
    for key, chunks in ref_chunks.items():
        start = len(flat_chunks)
        flat_chunks.extend(chunks)
        chunk_ranges[key] = (start, len(flat_chunks))

    if not flat_chunks:
        return {}

    chunk_emb = chunk_embedder.encode(
        flat_chunks,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    claim_indices = sorted({pair["claim_i"] for pair in fallback_pairs})
    claim_texts = [categorized[i]["claim"] for i in claim_indices]
    claim_emb = chunk_embedder.encode(
        claim_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    claim_vec_by_i = {
        claim_i: claim_emb[pos]
        for pos, claim_i in enumerate(claim_indices)
    }

    selected: dict[int, str] = {}
    for pair_i, pair in enumerate(fallback_pairs):
        start, end = chunk_ranges[pair["ref_key"]]
        if start == end:
            continue
        chunks = ref_chunks[pair["ref_key"]]
        scores = np.asarray(chunk_emb[start:end]) @ np.asarray(claim_vec_by_i[pair["claim_i"]])
        k = min(top_k, len(chunks))
        if k <= 0:
            continue
        best = np.argsort(scores)[-k:][::-1]
        # Restore document order after selecting by similarity; AlignScore then
        # sees a coherent local evidence slice rather than score-sorted scraps.
        best_doc_order = sorted(int(i) for i in best)
        selected[pair_i] = "\n\n".join(chunks[i] for i in best_doc_order)
        pair["selected_chunk_count"] = len(best_doc_order)
        pair["total_chunk_count"] = len(chunks)
    return selected


def _alignscore_per_ref_fulltext_topk(
    categorized: list[dict],
    per_claim_evidence: list[list[tuple[dict, str | None, str | None]]],
    alignscore_model,
    threshold: float,
    abstract_skip_threshold: float,
    top_k: int,
    chunk_words: int,
    chunk_embedder,
    embed_batch_size: int,
    bar = None,
) -> None:
    """Per-ref AlignScore with abstract cascade and top-k full-text chunks.

    For each (claim, ref) pair, first score the abstract when available. If
    that score is high enough, keep it and skip full text. Otherwise select
    top-k full-text chunks by semantic similarity to the claim and score only
    those chunks. Final per-ref score is max(abstract_score, selected_text_score).
    """
    records_by_claim: list[list[dict]] = []
    abstract_claims: list[str] = []
    abstract_premises: list[str] = []
    abstract_meta: list[dict] = []
    fallback_pairs: list[dict] = []
    extra_fallback_scores = 0
    abstract_skipped = 0
    chunks_to_align = 0
    _set_topk_bar_postfix(bar, chunks_to_align, abstract_skipped)

    for claim_i, evs in enumerate(per_claim_evidence):
        records: list[dict] = []
        for ref, abstract, full_text in evs:
            rec = {
                "ref": ref,
                "ref_key": _evidence_key(ref),
                "abstract_score": None,
                "full_text_score": None,
                "score": None,
                "selected_chunk_count": 0,
                "total_chunk_count": 0,
                "evidence_source": None,
            }
            records.append(rec)
            if abstract:
                abstract_meta.append(rec)
                abstract_claims.append(categorized[claim_i]["claim"])
                abstract_premises.append(abstract)
            elif full_text:
                fallback_pairs.append({
                    "claim_i": claim_i,
                    "record": rec,
                    "ref_key": rec["ref_key"],
                    "abstract": abstract,
                    "full_text": full_text,
                    "selected_chunk_count": 0,
                    "total_chunk_count": 0,
                })
        records_by_claim.append(records)

    abstract_scores = _score_alignscore_pairs(
        abstract_premises, abstract_claims, alignscore_model, bar=bar,
    )
    for rec, score in zip(abstract_meta, abstract_scores):
        rec["abstract_score"] = score

    # Decide which abstract-scored pairs still need selected full text.
    for claim_i, evs in enumerate(per_claim_evidence):
        for rec, (_, abstract, full_text) in zip(records_by_claim[claim_i], evs):
            abs_score = rec["abstract_score"]
            if abs_score is not None:
                if not full_text or abs_score >= abstract_skip_threshold:
                    rec["score"] = abs_score
                    rec["evidence_source"] = "abstract"
                    if full_text:
                        abstract_skipped += 1
                    continue
                rec["score"] = abs_score
                rec["evidence_source"] = "abstract"
                extra_fallback_scores += 1
                fallback_pairs.append({
                    "claim_i": claim_i,
                    "record": rec,
                    "ref_key": rec["ref_key"],
                    "abstract": abstract,
                    "full_text": full_text,
                    "selected_chunk_count": 0,
                    "total_chunk_count": 0,
                })

    _set_topk_bar_postfix(bar, chunks_to_align, abstract_skipped)
    if bar is not None and extra_fallback_scores:
        bar.total = (bar.total or 0) + extra_fallback_scores
        bar.refresh()

    selected = _select_top_chunks_by_similarity(
        categorized=categorized,
        fallback_pairs=fallback_pairs,
        chunk_embedder=chunk_embedder,
        top_k=top_k,
        chunk_words=chunk_words,
        batch_size=embed_batch_size,
    ) if fallback_pairs else {}

    fallback_claims: list[str] = []
    fallback_premises: list[str] = []
    fallback_meta: list[dict] = []
    for pair_i, pair in enumerate(fallback_pairs):
        selected_full_text = selected.get(pair_i)
        if not selected_full_text:
            continue
        premise_parts = []
        if pair.get("abstract"):
            premise_parts.append(f"Abstract:\n{pair['abstract']}")
        premise_parts.append(f"Full text:\n{selected_full_text}")
        fallback_meta.append(pair)
        fallback_claims.append(categorized[pair["claim_i"]]["claim"])
        fallback_premises.append("\n\n\n".join(premise_parts))

    chunks_to_align = sum(pair.get("selected_chunk_count", 0) for pair in fallback_meta)
    _add_cost("n_chunks_evaluated", chunks_to_align)
    _add_cost("n_abstract_skip", abstract_skipped)
    _set_topk_bar_postfix(bar, chunks_to_align, abstract_skipped)

    if bar is not None:
        skipped_fallback = len(fallback_pairs) - len(fallback_meta)
        if skipped_fallback:
            bar.update(skipped_fallback)

    fallback_scores = _score_alignscore_pairs(
        fallback_premises, fallback_claims, alignscore_model, bar=bar,
    )
    for pair, score in zip(fallback_meta, fallback_scores):
        rec = pair["record"]
        rec["full_text_score"] = score
        rec["selected_chunk_count"] = pair.get("selected_chunk_count", 0)
        rec["total_chunk_count"] = pair.get("total_chunk_count", 0)
        if rec["abstract_score"] is None or score > rec["abstract_score"]:
            rec["score"] = score
            rec["evidence_source"] = (
                "full_text_or_abstract_top_k"
                if pair.get("abstract") else "full_text_top_k"
            )
        else:
            rec["score"] = rec["abstract_score"]
            rec["evidence_source"] = "abstract"

    for claim_i, records in enumerate(records_by_claim):
        scored = [rec for rec in records if rec["score"] is not None]
        if not scored:
            categorized[claim_i]["supported"]          = None
            categorized[claim_i]["alignscore"]         = None
            categorized[claim_i]["evidence_refs"]      = []
            categorized[claim_i]["alignscore_per_ref"] = []
            categorized[claim_i]["alignscore_best"]    = None
            continue

        best = max(scored, key=lambda rec: rec["score"])
        categorized[claim_i]["supported"]     = bool(best["score"] >= threshold)
        categorized[claim_i]["alignscore"]    = round(float(best["score"]), 4)
        categorized[claim_i]["evidence_refs"] = [rec["ref"]["idx"] for rec in scored]
        categorized[claim_i]["alignscore_per_ref"] = [
            {
                "ref_idx": rec["ref"]["idx"],
                "score": round(float(rec["score"]), 4),
                "abstract_score": (
                    round(float(rec["abstract_score"]), 4)
                    if rec["abstract_score"] is not None else None
                ),
                "full_text_score": (
                    round(float(rec["full_text_score"]), 4)
                    if rec["full_text_score"] is not None else None
                ),
                "evidence_source": rec["evidence_source"],
                "selected_chunk_count": rec["selected_chunk_count"],
                "total_chunk_count": rec["total_chunk_count"],
            }
            for rec in scored
        ]
        categorized[claim_i]["alignscore_best"] = {
            "ref_idx": best["ref"]["idx"],
            "score": round(float(best["score"]), 4),
            "evidence_source": best["evidence_source"],
        }


def _alignscore_concat_fulltext_topk(
    categorized: list[dict],
    per_claim_evidence: list[list[tuple[dict, str | None, str | None]]],
    alignscore_model,
    threshold: float,
    abstract_skip_threshold: float,
    top_k: int,
    chunk_words: int,
    chunk_embedder,
    embed_batch_size: int,
    bar = None,
) -> None:
    """Concat AlignScore with top-k full-text chunks selected per claim+ref.

    Abstracts, when present, are scored as a cheap claim-level cascade. If the
    concatenated abstracts already clear ``abstract_skip_threshold``, full
    texts are skipped for that claim. Otherwise every full-text ref contributes
    up to ``top_k`` semantically nearest AlignScore-style chunks, and those
    selected slices are concatenated into one claim-level premise.
    """
    abstract_claims: list[str] = []
    abstract_premises: list[str] = []
    abstract_meta: list[tuple[int, list[dict]]] = []
    no_evidence = 0
    abstract_skipped = 0
    chunks_to_align = 0
    _set_topk_bar_postfix(bar, chunks_to_align, abstract_skipped)

    for claim_i, evs in enumerate(per_claim_evidence):
        abstract_refs = [(ref, abstract) for ref, abstract, _ in evs if abstract]
        if abstract_refs:
            abstract_meta.append((claim_i, [ref for ref, _ in abstract_refs]))
            abstract_claims.append(categorized[claim_i]["claim"])
            abstract_premises.append(
                "\n\n\n".join(f"Abstract:\n{abstract}" for _, abstract in abstract_refs)
            )
        elif not any(full_text for _, _, full_text in evs):
            categorized[claim_i]["supported"]     = None
            categorized[claim_i]["alignscore"]    = None
            categorized[claim_i]["evidence_refs"] = []
            categorized[claim_i]["alignscore_topk_refs"] = []
            no_evidence += 1

    if bar is not None and no_evidence:
        bar.update(no_evidence)

    abstract_scores = _score_alignscore_pairs(
        abstract_premises, abstract_claims, alignscore_model, bar=bar,
    )

    abstract_score_by_claim: dict[int, float] = {}
    abstract_refs_by_claim: dict[int, list[dict]] = {}
    for (claim_i, refs), score in zip(abstract_meta, abstract_scores):
        abstract_score_by_claim[claim_i] = score
        abstract_refs_by_claim[claim_i] = refs

    fallback_pairs: list[dict] = []
    fallback_claims: set[int] = set()
    for claim_i, evs in enumerate(per_claim_evidence):
        if not evs:
            continue
        abs_score = abstract_score_by_claim.get(claim_i)
        has_full_text = any(full_text for _, _, full_text in evs)
        if abs_score is not None and (not has_full_text or abs_score >= abstract_skip_threshold):
            if has_full_text:
                abstract_skipped += 1
            categorized[claim_i]["supported"]     = bool(abs_score >= threshold)
            categorized[claim_i]["alignscore"]    = round(float(abs_score), 4)
            categorized[claim_i]["evidence_refs"] = [
                ref["idx"] for ref in abstract_refs_by_claim.get(claim_i, [])
            ]
            categorized[claim_i]["alignscore_topk_refs"] = []
            categorized[claim_i]["alignscore_best"] = {
                "score": round(float(abs_score), 4),
                "evidence_source": "abstract",
            }
            continue
        for ref, abstract, full_text in evs:
            if not full_text:
                continue
            fallback_pairs.append({
                "claim_i": claim_i,
                "ref": ref,
                "ref_key": _evidence_key(ref),
                "abstract": abstract,
                "full_text": full_text,
                "selected_chunk_count": 0,
                "total_chunk_count": 0,
            })
            fallback_claims.add(claim_i)

    _set_topk_bar_postfix(bar, chunks_to_align, abstract_skipped)
    extra_fallback_scores = sum(
        1 for claim_i in fallback_claims
        if claim_i in abstract_score_by_claim
    )
    if bar is not None and extra_fallback_scores:
        bar.total = (bar.total or 0) + extra_fallback_scores
        bar.refresh()

    selected = _select_top_chunks_by_similarity(
        categorized=categorized,
        fallback_pairs=fallback_pairs,
        chunk_embedder=chunk_embedder,
        top_k=top_k,
        chunk_words=chunk_words,
        batch_size=embed_batch_size,
    ) if fallback_pairs else {}

    fallback_by_claim: dict[int, list[tuple[dict, str]]] = {}
    for pair_i, pair in enumerate(fallback_pairs):
        selected_full_text = selected.get(pair_i)
        if not selected_full_text:
            continue
        fallback_by_claim.setdefault(pair["claim_i"], []).append((pair, selected_full_text))

    chunks_to_align = sum(
        pair.get("selected_chunk_count", 0)
        for pairs in fallback_by_claim.values()
        for pair, _ in pairs
    )
    _add_cost("n_chunks_evaluated", chunks_to_align)
    _add_cost("n_abstract_skip", abstract_skipped)
    _set_topk_bar_postfix(bar, chunks_to_align, abstract_skipped)

    fallback_claim_indices = sorted(fallback_by_claim)
    fallback_premises: list[str] = []
    fallback_claim_texts: list[str] = []
    for claim_i in fallback_claim_indices:
        parts: list[str] = []
        seen_abstract_refs: set[str] = set()
        for ref, abstract, _ in per_claim_evidence[claim_i]:
            if not abstract:
                continue
            key = _evidence_key(ref)
            if key in seen_abstract_refs:
                continue
            seen_abstract_refs.add(key)
            parts.append(f"Abstract:\n{abstract}")
        for pair, selected_full_text in fallback_by_claim[claim_i]:
            parts.append(f"Full text:\n{selected_full_text}")
        fallback_premises.append("\n\n\n".join(parts))
        fallback_claim_texts.append(categorized[claim_i]["claim"])

    skipped_fallback_claims = len(fallback_claims) - len(fallback_claim_indices)
    if bar is not None and skipped_fallback_claims:
        bar.update(skipped_fallback_claims)

    fallback_scores = _score_alignscore_pairs(
        fallback_premises, fallback_claim_texts, alignscore_model, bar=bar,
    )
    fallback_score_by_claim = dict(zip(fallback_claim_indices, fallback_scores))

    for claim_i, evs in enumerate(per_claim_evidence):
        if not evs:
            continue
        abs_score = abstract_score_by_claim.get(claim_i)
        fallback_score = fallback_score_by_claim.get(claim_i)
        if abs_score is None and fallback_score is None:
            continue
        if fallback_score is not None and (abs_score is None or fallback_score > abs_score):
            score = fallback_score
            evidence_source = "full_text_or_abstract_top_k" if abs_score is not None else "full_text_top_k"
        else:
            score = abs_score
            evidence_source = "abstract"

        topk_pairs = fallback_by_claim.get(claim_i, [])
        evidence_ref_ids = {ref["idx"] for ref, abstract, full_text in evs if abstract or full_text}
        categorized[claim_i]["supported"]     = bool(score >= threshold)
        categorized[claim_i]["alignscore"]    = round(float(score), 4)
        categorized[claim_i]["evidence_refs"] = sorted(evidence_ref_ids)
        categorized[claim_i]["alignscore_topk_refs"] = [
            {
                "ref_idx": pair["ref"]["idx"],
                "selected_chunk_count": pair.get("selected_chunk_count", 0),
                "total_chunk_count": pair.get("total_chunk_count", 0),
            }
            for pair, _ in topk_pairs
        ]
        categorized[claim_i]["alignscore_best"] = {
            "score": round(float(score), 4),
            "abstract_score": (
                round(float(abs_score), 4) if abs_score is not None else None
            ),
            "full_text_score": (
                round(float(fallback_score), 4)
                if fallback_score is not None else None
            ),
            "evidence_source": evidence_source,
        }


# ── Per-survey processing ─────────────────────────────────────────────────────


def _legacy_or_new_sources(claim: dict) -> list[dict]:
    """Return claim's sources list in new schema, synthesising a single-source
    entry from legacy ``source_sentence[_idx]`` fields if needed.

    Old veriscore/claimify outputs have ``source_sentence: ""`` and the new
    ``sources[]`` list may be absent. The new patch writes ``sources[]``
    with full occurrences. This shim keeps compatibility with old caches —
    though those cases will mostly be empty-string → no scope resolution.
    """
    sources = claim.get("sources") or []
    if sources:
        return sources
    legacy = claim.get("source_sentence")
    if legacy:
        return [{
            "sentence":     legacy,
            "sentence_idx": claim.get("source_sentence_idx", 0),
        }]
    return []


# ── Stage checkpoints (tmp/factuality/{classify,align}/.../<sid>.json) ───────
#
# Two costly stages in process_survey (classification = LLM money, AlignScore =
# CPU-minutes) each persist a checkpoint after completion. A crash between them
# — or a re-run after e.g. an AlignScore/NLTK environment fix — then resumes
# from the last completed stage instead of re-spending.
#
# File schema:
#   {"stage":         "classified" | "aligned",
#    "categorized":   [...],                    # as built by stage 2/3
#    "n_no_evidence": int,                      # only present when stage="aligned"
#    "n_scope_failed": int}                     # idem
#
# Classification does not depend on scope / evidence_source / aggregation, so
# that cache is shared across factuality variants. AlignScore does depend on
# those knobs, so its cache remains variant-specific.

# Checkpoint root. Default — repo-relative `tmp/factuality/`, which is
# persistent (survives reboots) and already gitignored. The default is
# what we want on bare-metal (server, macOS without docker).
#
# When running inside our own thesis docker image, the canonical `tmp/`
# mount lives at `/tmp` (not `/app/tmp`), so the Makefile sets the env
# override `FACTUALITY_CHECKPOINT_ROOT=/tmp/factuality` to point the code
# at the real mounted path. Any other container environment (e.g. a
# shared SSH-over-docker host) leaves the env unset and uses the default
# repo-relative path.
def _default_checkpoint_root() -> Path:
    override = os.environ.get("FACTUALITY_CHECKPOINT_ROOT")
    if override:
        return Path(override)
    return ROOT / "tmp" / "factuality"

_CHECKPOINT_ROOT = _default_checkpoint_root()


def _run_id(cfg: dict) -> str:
    return f"{cfg['judge_id']}_{cfg['judge_comment']}"


def _full_text_topk_enabled(cfg: dict) -> bool:
    return (
        cfg.get("evidence_source", "abstract") in {"full_text", "full_text_or_abstract"}
        and cfg.get("evidence_aggregation", "concat") in {"concat", "per_ref"}
        and int(cfg.get("full_text_top_k_chunks", 0) or 0) > 0
    )


def _variant_id(cfg: dict) -> str:
    variant = (
        f"{cfg.get('claim_scope', 'section')}"
        f"_{cfg.get('evidence_source', 'abstract')}"
        f"_{cfg.get('evidence_aggregation', 'concat')}"
    )
    if _full_text_topk_enabled(cfg):
        top_k = int(cfg.get("full_text_top_k_chunks", 0))
        variant += f"_topk{top_k}"
        if cfg.get("evidence_source", "abstract") == "full_text_or_abstract":
            threshold = float(cfg.get("full_text_abstract_skip_threshold", 0.75))
            threshold_tag = str(threshold).replace(".", "")
            variant += f"_abs{threshold_tag}"
    return variant


def _classification_checkpoint_dir(cfg: dict, dataset_id: str, model_id: str) -> Path:
    """Per-(dataset, model, judge) directory for LLM classification checkpoints."""
    return _CHECKPOINT_ROOT / "classify" / f"{dataset_id}_{model_id}_{_run_id(cfg)}"


def _align_checkpoint_dir(cfg: dict, dataset_id: str, model_id: str) -> Path:
    """Per-(dataset, model, factuality variant) directory for AlignScore checkpoints."""
    return (
        _CHECKPOINT_ROOT / "align"
        / f"{dataset_id}_{model_id}_{_run_id(cfg)}_{_variant_id(cfg)}"
    )


def _save_state(state_file: Path, state: dict) -> None:
    """Atomic write of a stage checkpoint via tmp-rename."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_file.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    tmp.replace(state_file)


def _load_state(state_file: Path) -> dict | None:
    """Load a stage checkpoint, or None when absent / corrupt / malformed."""
    if not state_file.exists():
        return None
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("corrupt state file %s — ignoring", state_file)
        return None
    if data.get("stage") not in {"classified", "aligned"} \
            or not isinstance(data.get("categorized"), list):
        logger.warning("state file %s has unexpected schema — ignoring", state_file)
        return None
    return data


_CLASSIFICATION_CACHE_DROP_KEYS = {
    "supported",
    "alignscore",
    "alignscore_per_ref",
    "alignscore_best",
    "evidence_refs",
    "alignscore_topk_refs",
    "scope_resolution",
    "n_sources_resolved",
    "n_sources_total",
    "scope_citations",
}


def _classification_cache_payload(categorized: list[dict]) -> list[dict]:
    """Keep only claim + category fields for the evidence-agnostic cache."""
    return [
        {k: v for k, v in c.items() if k not in _CLASSIFICATION_CACHE_DROP_KEYS}
        for c in categorized
    ]


def _save_classification_state_if_missing(state_file: Path, categorized: list[dict]) -> None:
    """Seed classification cache from an existing result without overwriting it."""
    if state_file.exists() or not categorized:
        return
    _save_state(state_file, {
        "stage":       "classified",
        "categorized": _classification_cache_payload(categorized),
    })


def process_survey(
    gen: dict,
    claims_dir: Path,
    out_path: Path,
    cfg: dict,
    client: OpenAI,
    alignscore_model,
    corpus_index: dict,
    sources_dir: Path,
    classification_checkpoint_dir: Path,
    align_checkpoint_dir: Path,
    chunk_embedder = None,
    *,
    global_tokens: TokenCounter | None = None,
    abstract_cache: dict | None = None,
) -> dict | None:
    """Evaluate factuality for one survey.

    Pipeline:
        1. Resolve per-claim scope (paragraph/section + citation list) via
           :func:`metrics.factuality.claim_scope.resolve_claim_scope`.
        2. Fetch evidence (abstract or full-text) for every unique ref in
           scope across all claims — dedup first, then one batched fetch.
        3. LLM-classify each claim A/B/C/D (parallel). Classification uses
           paragraph-scope's first fetched evidence as a light context hint.
        4. AlignScore-based Support verification per claim. Two modes:
           ``concat`` (default, fast — one score per claim) or ``per_ref``
           (slower, preserves per-ref attribution for debugging).
        5. CitCorrect_k per category = supported / total in that category.

    Args:
        gen: Generation dict with success, text, query, meta.references.
        claims_dir: Path to pre-computed claims cache (veriscore/claimify).
        out_path: Output directory for scores.
        cfg: Config dict. Factcheck-specific knobs:
            ``claim_scope`` (paragraph|section),
            ``section_max_ancestor_depth`` (None=unlimited, 0=own-only),
            ``evidence_source`` (abstract|full_text|full_text_or_abstract),
            ``evidence_aggregation`` (concat|per_ref),
            ``alignscore_threshold`` (float, default 0.5).
        client: OpenAI client instance.
        alignscore_model: Initialized AlignScore model (or None to skip).
        corpus_index: ``{doc_id_str: "title\\n\\nabstract"}`` map for the
            tier-0 cache path in evidence fetching.
        global_tokens: Run-wide TokenCounter; per-survey tokens roll up
            after classification so the outer tqdm can show cumulative cost.
        abstract_cache: Shared abstract-cache dict. Caller loads once in
            :func:`main` and passes through so multiple surveys share hits.

    Returns:
        Score dict with citation correctness per category, or None on skip.
    """
    survey_id = gen["id"]
    out_file  = out_path / f"{survey_id}.json"
    classification_state_file = classification_checkpoint_dir / f"{survey_id}.json"
    align_state_file          = align_checkpoint_dir / f"{survey_id}.json"

    _reset_cost_meter()

    cached = check_and_load_cache(out_file, cfg, survey_id)
    if cached is not None:
        cached_claims = cached.get("claims", [])
        if isinstance(cached_claims, list):
            _save_classification_state_if_missing(
                classification_state_file, cached_claims,
            )

        # Promote-from-classify-only workflow: if the cached file was
        # produced with alignscore disabled but this run has alignscore
        # on, the cached result is stale (supported=None everywhere).
        # Ignore the cache and re-run — the state file (saved when the
        # previous run finished with alignscore off) will let us skip
        # classify and jump straight into fetch + align.
        currently_want_align = alignscore_model is not None
        cached_had_align     = cached.get("alignscore_enabled", True)
        if currently_want_align and not cached_had_align:
            logger.info(
                "[PROC] sid=%s — cached file was produced with alignscore "
                "off; re-running fetch + align (classify will be loaded "
                "from checkpoint if present).",
                survey_id,
            )
        else:
            logger.info("[SKIP] sid=%s — already scored", survey_id)
            return cached

    if not gen.get("success", False):
        logger.info("[SKIP] sid=%s — generation not successful", survey_id)
        return None

    # Load claimify cache
    claims_file = claims_dir / f"{survey_id}.json"
    if not claims_file.exists():
        logger.info("[SKIP] sid=%s — no claims file (run claimify first)", survey_id)
        return None

    with open(claims_file) as f:
        claims_data = json.load(f)

    claims = claims_data.get("claims", [])
    if not claims:
        logger.info("[SKIP] sid=%s — empty claims", survey_id)
        return None

    # Config knobs (factcheck-specific + classical)
    max_retries        = cfg.get("max_retries", 3)
    judge_workers      = cfg.get("judge_workers", 8)
    disable_reasoning  = not cfg.get("judge_reasoning", True)
    provider           = cfg.get("judge_provider") or None
    reasoning_effort   = cfg.get("judge_reasoning_effort") or None
    claim_scope_mode   = cfg.get("claim_scope", "section")
    max_ancestor_depth = cfg.get("section_max_ancestor_depth")   # None = unlimited
    evidence_source    = cfg.get("evidence_source", "abstract")
    evidence_aggregation = cfg.get("evidence_aggregation", "concat")
    alignscore_threshold = cfg.get("alignscore_threshold", 0.5)
    full_text_top_k_chunks = int(cfg.get("full_text_top_k_chunks", 0) or 0)
    full_text_chunk_words = int(cfg.get("full_text_chunk_words", 350) or 350)
    full_text_abstract_skip_threshold = float(
        cfg.get("full_text_abstract_skip_threshold", 0.75)
    )
    full_text_chunk_embedder_batch_size = int(
        cfg.get("full_text_chunk_embedder_batch_size", 32) or 32
    )
    ss_api_key         = os.environ.get(cfg.get("ss_api_key_env", "SEMANTIC_SCHOLAR_API_KEY"))
    ss_enabled         = bool(cfg.get("ss_enabled", False))
    evidence_mode      = cfg.get("evidence_mode", "internal")
    use_full_text_topk = _full_text_topk_enabled(cfg)
    alignscore_enabled = alignscore_model is not None
    if alignscore_enabled and use_full_text_topk and chunk_embedder is None:
        raise RuntimeError("full_text_top_k_chunks > 0 requires a loaded chunk embedder")

    logger.info(
        "[PROC] sid=%s | %s | %d claims | scope=%s src=%s agg=%s alignscore=%s",
        survey_id, gen.get("query", "")[:50], len(claims),
        claim_scope_mode, evidence_source, evidence_aggregation,
        "on" if alignscore_enabled else "off",
    )
    t0 = time.time()
    per_tokens = TokenCounter()

    # Stage checkpoints — classification is shared across factuality variants;
    # AlignScore remains variant-specific because it consumes evidence.
    classification_state      = _load_state(classification_state_file)
    align_state               = _load_state(align_state_file)
    skip_align                = align_state is not None and align_state["stage"] == "aligned"
    skip_classify             = skip_align or classification_state is not None
    if classification_state is not None:
        logger.info(
            "[RESUME] sid=%s — loaded classification state stage=%s",
            survey_id, classification_state["stage"],
        )
    if align_state is not None:
        logger.info(
            "[RESUME] sid=%s — loaded align state stage=%s (skip_align=%s)",
            survey_id, align_state["stage"], skip_align,
        )

    text     = gen.get("text", "")
    gen_refs = gen.get("meta", {}).get("references", [])

    # Debug hook — when ``--debug-claim-idx N`` is set, dump detailed pipeline
    # state for the Nth claim at each stage. Writes go through logger.info →
    # results/logs/factuality.log (stderr is owned by tqdm).
    debug_idx = cfg.get("_debug_claim_idx")
    do_debug  = debug_idx is not None and 0 <= debug_idx < len(claims)

    # ── Step 1: Resolve scope for each claim ──────────────────────────────
    # For every claim get (paragraph_refs, section_refs) per the canonical
    # claim_scope rule; we then pick one per config.claim_scope.
    claim_scopes: list[dict] = []
    for c in claims:
        claim_scopes.append(resolve_claim_scope(
            text, _legacy_or_new_sources(c), gen_refs,
            max_ancestor_depth=max_ancestor_depth,
        ))

    def _scope_refs_for(scope: dict) -> list[dict]:
        return (scope["paragraph_refs"] if claim_scope_mode == "paragraph"
                else scope["section_refs"])

    # DEBUG DUMP: scope
    if do_debug:
        dc     = claims[debug_idx]
        scope  = claim_scopes[debug_idx]
        refs_s = _scope_refs_for(scope)
        logger.info("==== DEBUG sid=%s claim_idx=%d ====", survey_id, debug_idx)
        logger.info("[DEBUG] claim: %s", dc["claim"])
        logger.info("[DEBUG] scope mode: %s  max_ancestor_depth: %s",
                    claim_scope_mode, max_ancestor_depth)
        logger.info(
            "[DEBUG] scope citations: %s  (resolution=%s, %d/%d sources resolved)",
            [r["idx"] for r in refs_s],
            scope["resolution"],
            scope["n_sources_resolved"], scope["n_sources_total"],
        )
        for r in refs_s:
            logger.info(
                "[DEBUG]   ref=[%s] arxiv_id=%s  ss_id=%s  doc_id=%s",
                r["idx"], r.get("arxiv_id"), r.get("semantic_scholar_id"), r.get("doc_id"),
            )

        # Dump the actual text spans that were used as scope. Recomputed here
        # (instead of returned from resolve_claim_scope) because we don't want
        # the public API to carry 1604 text-blob copies for the sake of 1
        # debug claim. Per-source → union already reflects what the resolver
        # did internally.
        for src_i, src in enumerate(_legacy_or_new_sources(dc)):
            sent = src.get("sentence") or ""
            off, res = _find_source_offset(text, sent)
            logger.info(
                "[DEBUG] --- source %d/%d: resolution=%s  offset=%s  sentence=%r",
                src_i + 1, len(_legacy_or_new_sources(dc)),
                res, off, sent[:160],
            )
            if off is None:
                continue

            if claim_scope_mode == "paragraph":
                p_s, p_e = _paragraph_span(text, off)
                logger.info(
                    "[DEBUG]   paragraph span [%d:%d]  len=%d chars",
                    p_s, p_e, p_e - p_s,
                )
                logger.info("[DEBUG]   paragraph text:\n%s", text[p_s:p_e])
            else:
                spans = _section_scope_spans(
                    text, off, max_ancestor_depth=max_ancestor_depth,
                )
                total = sum(e - s for s, e in spans)
                logger.info(
                    "[DEBUG]   section scope: %d span(s), total %d chars",
                    len(spans), total,
                )
                for sp_i, (s, e) in enumerate(spans):
                    tag = "own" if sp_i == len(spans) - 1 else f"ancestor {sp_i+1}"
                    logger.info(
                        "[DEBUG]   --- span %d (%s) [%d:%d]  len=%d chars",
                        sp_i + 1, tag, s, e, e - s,
                    )
                    logger.info("[DEBUG]   %s text:\n%s", tag, text[s:e])

    # ── Step 2: Fetch evidence ───────────────────────────────────────────
    # Fetch goes through the unified per-survey sources file (see
    # metrics/factuality/sources_io.py for schema) rather than dispatching
    # ad-hoc per-ref. The fetch_bar tracks fetches across ALL refs in the
    # generation (not only those present in claim scopes) because the
    # unified file is claim-agnostic — it describes evidence for the whole
    # generation and is cache-reusable across claim-extractor variants.

    # Three inner progress bars — created fresh per survey (no position=,
    # leave=False), closed in a try/finally below. This avoids the reset()-
    # on-persistent-bar duplication we hit in the veriscore pass; the small
    # cost is a brief clear between surveys.
    # Creation order mirrors pipeline execution order (classify → fetch →
    # align) so the tqdm display reads top-to-bottom in the order work
    # actually happens.
    classify_bar = tqdm(
        total=len(claims),
        desc="  classify", unit="claim",
        leave=False, dynamic_ncols=True, mininterval=0.2,
    )
    fetch_bar = tqdm(
        total=max(len(gen_refs), 1),
        desc="  fetch",    unit="ref",
        leave=False, dynamic_ncols=True, mininterval=0.2,
    )
    align_bar = tqdm(
        total=len(claims),
        desc="  align",    unit="claim",
        leave=False, dynamic_ncols=True, mininterval=0.2,
    )

    key_to_evidence: dict[str, tuple[str | None, str]] = {}
    raw_key_to_evidence: dict[str, tuple[str | None, str | None]] = {}
    try:
        # Classification context — the local survey prose around the claim
        # (first resolved source's paragraph span from `resolve_claim_scope`).
        # NOT the cited paper's abstract: that would require a hit against
        # key_to_evidence, locking classify to the fetch stage and baking in
        # an arbitrary "first ref's abstract" signal (the claim might actually
        # be grounded in ref #3 of the scope, not ref #1). Local survey prose
        # is free (already in memory), always-available, and discriminates
        # A/B/C/D on how the survey author uses the citation — which is what
        # the categorical scheme actually measures.
        def _classify_context(i: int) -> str:
            spans = claim_scopes[i].get("paragraph_spans") or []
            if not spans:
                return ""
            s, e = spans[0]   # first resolved source's paragraph
            return text[s:e]

        # ── Step 2: LLM classify each claim A/B/C/D (parallel) ────────────
        # Moved BEFORE evidence fetch so that:
        #   * classify stage doesn't depend on any network,
        #   * on resume with stage=="classified", the fetch below runs only
        #     if align still needs to run.
        if skip_classify:
            # Load categorized from checkpoint; tokens are NOT re-added to
            # global_tokens (they belong to the original run).
            classify_source_state = align_state if skip_align else classification_state
            categorized = classify_source_state["categorized"]   # type: ignore[index]
            classify_bar.update(classify_bar.total)
            logger.info(
                "[RESUME] sid=%s — skipped classify, %d claims loaded",
                survey_id, len(categorized),
            )
        else:
            results: list[dict | None] = [None] * len(claims)
            futures: dict = {}
            with ThreadPoolExecutor(max_workers=judge_workers) as pool:
                for i, c in enumerate(claims):
                    fut = pool.submit(
                        classify_claim,
                        c["claim"], _classify_context(i),
                        client, cfg["judge_model"], max_retries,
                        disable_reasoning, provider, per_tokens, reasoning_effort,
                    )
                    futures[fut] = i

                for fut in as_completed(futures):
                    i = futures[fut]
                    results[i] = {**claims[i], **fut.result()}
                    if global_tokens is not None:
                        classify_bar.set_postfix_str(
                            f"surv: {per_tokens.fmt()}  total: {global_tokens.fmt()}",
                            refresh=False,
                        )
                    else:
                        classify_bar.set_postfix_str(per_tokens.fmt(), refresh=False)
                    classify_bar.update(1)

            if global_tokens is not None:
                global_tokens.add(
                    per_tokens.in_tokens, per_tokens.out_tokens, per_tokens.cost_usd,
                )

            categorized = [r for r in results if r is not None]
            # Persist classify output BEFORE running AlignScore — if align
            # then crashes, a re-run only repeats align, not the LLM calls.
            _save_state(classification_state_file, {
                "stage":       "classified",
                "categorized": _classification_cache_payload(categorized),
            })

        # DEBUG DUMP: category for the debug claim.
        if do_debug:
            dc = categorized[debug_idx]
            logger.info(
                "[DEBUG] category: %s  confidence: %s",
                dc.get("category"), dc.get("confidence"),
            )

        # ── Step 3: Fetch evidence (via unified per-survey sources file) ──
        # Goes through the unified entry point: in internal mode runs the
        # waterfall (cache → corpus → arxiv API → opt. SS), saves the
        # <sid>_sources.json artefact, and returns the key_to_evidence dict;
        # in external mode just loads a pre-built file. Only runs when
        # AlignScore will actually consume its output.
        if alignscore_enabled and gen_refs and not skip_align:
            key_to_evidence = prepare_key_evidence(
                gen,
                evidence_mode=evidence_mode,
                evidence_source=evidence_source,
                sources_dir=sources_dir,
                abstract_cache=abstract_cache,
                corpus_index=corpus_index,
                ss_api_key=ss_api_key,
                ss_enabled=ss_enabled,
                progress_bar=fetch_bar,
            )
            if use_full_text_topk:
                sources = load_gen_sources(sources_dir, survey_id)
                if sources is None:
                    sources_path = sources_dir / f"{survey_id}_sources.json"
                    raise FileNotFoundError(
                        f"sources file disappeared while preparing top-k evidence: "
                        f"{sources_path}"
                    )
                raw_key_to_evidence = {
                    _evidence_key(entry): (entry.get("abstract"), entry.get("text"))
                    for entry in sources.get("refs", {}).values()
                }
        else:
            # AlignScore off / no refs / align already done — sweep the bar
            # to completion so the display stays honest.
            fetch_bar.update(max(len(gen_refs), 1))

        # DEBUG DUMP: extracted evidence for the debug claim.
        if do_debug and not skip_align:
            logger.info("[DEBUG] extracted evidence for debug claim:")
            for r in _scope_refs_for(claim_scopes[debug_idx]):
                key = _evidence_key(r)
                txt, src = key_to_evidence.get(key, (None, "—"))
                snippet = (txt[:300] + "...") if (txt and len(txt) > 300) else (txt or "—")
                logger.info(
                    "[DEBUG]   ref=[%s] source=%s  text: %s",
                    r["idx"], src, snippet,
                )

        # ── Step 4: AlignScore Support via evidence pool (batched) ────────
        # per_claim_evidence[i] = [(ref_dict, evidence_text), ...] for claim i
        if skip_align:
            # State already contains `supported`, `alignscore`, `evidence_refs`
            # for every entry in `categorized`. Pull the aggregate counters
            # from the checkpoint if they were saved there; otherwise
            # recompute from categorized (only ``n_scope_failed`` needs
            # claim_scopes, which is cheap enough to always hold).
            n_no_evidence  = align_state.get("n_no_evidence", sum(     # type: ignore[union-attr]
                1 for c in categorized if not c.get("evidence_refs")
            ))
            n_scope_failed = align_state.get("n_scope_failed", sum(    # type: ignore[union-attr]
                1 for s in claim_scopes if s["resolution"] == "failed"
            ))
            align_bar.update(align_bar.total)
            logger.info("[RESUME] sid=%s — skipped align", survey_id)
        elif alignscore_enabled:
            n_no_evidence  = 0
            n_scope_failed = 0
            per_claim_evidence: list[list[tuple[dict, str]]] = []
            per_claim_raw_evidence: list[list[tuple[dict, str | None, str | None]]] = []
            for i in range(len(categorized)):
                refs_i = _scope_refs_for(claim_scopes[i])
                if use_full_text_topk:
                    raw_evs = []
                    for r in refs_i:
                        abstract, full_text = raw_key_to_evidence.get(
                            _evidence_key(r), (None, None)
                        )
                        if evidence_source == "full_text":
                            abstract = None
                        raw_evs.append((r, abstract, full_text))
                    raw_evs = [
                        (r, abstract, full_text)
                        for r, abstract, full_text in raw_evs
                        if (full_text if evidence_source == "full_text" else abstract or full_text)
                    ]
                    per_claim_raw_evidence.append(raw_evs)
                    if not raw_evs:
                        n_no_evidence += 1
                else:
                    evs = [
                        (r, key_to_evidence.get(_evidence_key(r), (None, None))[0])
                        for r in refs_i
                    ]
                    evs = [(r, t) for r, t in evs if t]   # drop misses
                    per_claim_evidence.append(evs)
                    if not evs:
                        n_no_evidence += 1
                if claim_scopes[i]["resolution"] == "failed":
                    n_scope_failed += 1

            if use_full_text_topk:
                if evidence_aggregation == "concat":
                    align_bar.reset(total=max(len(categorized), 1))
                    align_bar.unit = "score"
                    _alignscore_concat_fulltext_topk(
                        categorized, per_claim_raw_evidence,
                        alignscore_model, alignscore_threshold,
                        full_text_abstract_skip_threshold,
                        full_text_top_k_chunks,
                        full_text_chunk_words,
                        chunk_embedder,
                        full_text_chunk_embedder_batch_size,
                        bar=align_bar,
                    )
                elif evidence_aggregation == "per_ref":
                    pair_count = sum(len(evs) for evs in per_claim_raw_evidence)
                    align_bar.reset(total=max(pair_count, 1))
                    align_bar.unit = "score"
                    _alignscore_per_ref_fulltext_topk(
                        categorized, per_claim_raw_evidence,
                        alignscore_model, alignscore_threshold,
                        full_text_abstract_skip_threshold,
                        full_text_top_k_chunks,
                        full_text_chunk_words,
                        chunk_embedder,
                        full_text_chunk_embedder_batch_size,
                        bar=align_bar,
                    )
                else:
                    raise ValueError(f"unknown evidence_aggregation: {evidence_aggregation!r}")
            elif evidence_aggregation == "concat":
                # Bar total already == len(claims), which equals the total
                # updates concat mode will emit (one per scored claim + one
                # per no-evidence claim at the end). No reset needed.
                _alignscore_concat(
                    categorized, per_claim_evidence,
                    alignscore_model, alignscore_threshold,
                    bar=align_bar,
                )
            elif evidence_aggregation == "per_ref":
                # In per_ref mode the bar tracks per-pair progress, not per
                # claim — reset its total to the real pair count before
                # kicking off scoring. ``reset`` on a fresh-per-survey bar
                # that hasn't been drawn for other content is safe.
                pair_count = sum(len(evs) for evs in per_claim_evidence)
                align_bar.reset(total=max(pair_count, 1))
                align_bar.unit = "pair"
                _alignscore_per_ref(
                    categorized, per_claim_evidence,
                    alignscore_model, alignscore_threshold,
                    bar=align_bar,
                )
            else:
                raise ValueError(f"unknown evidence_aggregation: {evidence_aggregation!r}")

            # Persist full align result before computing the summary: if we
            # crash between here and the final write, the next re-run picks
            # up here instead of repeating the alignscore pass.
            _save_state(align_state_file, {
                "stage":          "aligned",
                "categorized":    categorized,
                "n_no_evidence":  n_no_evidence,
                "n_scope_failed": n_scope_failed,
            })
        else:
            # AlignScore off — advance align bar to completion and record no Support.
            n_no_evidence  = 0
            n_scope_failed = 0
            align_bar.update(align_bar.total)
            for i, c in enumerate(categorized):
                c["supported"]     = None
                c["alignscore"]    = None
                c["evidence_refs"] = [r["idx"] for r in _scope_refs_for(claim_scopes[i])]
                if claim_scopes[i]["resolution"] == "failed":
                    n_scope_failed += 1

        # DEBUG DUMP: support for the debug claim.
        if do_debug:
            dc = categorized[debug_idx]
            logger.info(
                "[DEBUG] supported: %s  alignscore: %s  aggregation: %s",
                dc.get("supported"), dc.get("alignscore"), evidence_aggregation,
            )
            per_ref = dc.get("alignscore_per_ref")
            if per_ref:
                for rec in per_ref:
                    logger.info(
                        "[DEBUG]   per_ref: [%s] → %.4f",
                        rec["ref_idx"], rec["score"],
                    )
            elif dc.get("evidence_refs"):
                logger.info(
                    "[DEBUG]   concat over refs: %s", dc["evidence_refs"],
                )
            logger.info("==== /DEBUG sid=%s claim_idx=%d ====", survey_id, debug_idx)
    finally:
        fetch_bar.close()
        classify_bar.close()
        align_bar.close()

    # Attach scope diagnostics to each claim for observability in output.
    for i, c in enumerate(categorized):
        scope = claim_scopes[i]
        c["scope_resolution"]     = scope["resolution"]
        c["n_sources_resolved"]   = scope["n_sources_resolved"]
        c["n_sources_total"]      = scope["n_sources_total"]
        c["scope_citations"]      = [r["idx"] for r in _scope_refs_for(scope)]

    # ── Step 5: Compute CitCorrect_k per category ─────────────────────────
    categories = ["A", "B", "C", "D"]
    cit_correct: dict[str, float | None] = {}
    cit_counts:  dict[str, dict]         = {}

    for k in categories:
        subset = [c for c in categorized if c["category"] == k]
        if not subset or not alignscore_enabled:
            cit_correct[k] = None
            cit_counts[k]  = {"n": len(subset), "n_supported": None}
            continue
        n_sup = sum(1 for c in subset if c["supported"])
        cit_correct[k] = round(n_sup / len(subset), 4)
        cit_counts[k]  = {"n": len(subset), "n_supported": n_sup}

    overall_n   = len(categorized)
    if alignscore_enabled:
        overall_sup = sum(1 for c in categorized if c["supported"])
        cit_correct_overall = round(overall_sup / overall_n, 4) if overall_n else None
    else:
        overall_sup         = None
        cit_correct_overall = None

    result = {
        "survey_id":    survey_id,
        "dataset_id":   gen["dataset_id"],
        "model_id":     gen["model_id"],
        "query":        gen.get("query", ""),
        "n_claims":     overall_n,
        "n_supported":  overall_sup,
        "n_no_evidence":  n_no_evidence,
        "n_scope_failed": n_scope_failed,
        # Factcheck configuration snapshot — lets downstream viewer/plots
        # disambiguate runs without reading the dir-name variant tag.
        "claim_scope":                  claim_scope_mode,
        "section_max_ancestor_depth":   max_ancestor_depth,
        "evidence_source":              evidence_source,
        "evidence_aggregation":         evidence_aggregation,
        "full_text_top_k_chunks":       full_text_top_k_chunks if use_full_text_topk else None,
        "full_text_chunk_words":        full_text_chunk_words if use_full_text_topk else None,
        "full_text_abstract_skip_threshold": (
            full_text_abstract_skip_threshold if use_full_text_topk else None
        ),
        "full_text_chunk_embedder": (
            cfg.get("full_text_chunk_embedder") if use_full_text_topk else None
        ),
        "alignscore_enabled":           alignscore_enabled,
        "alignscore_threshold":         alignscore_threshold,
        "cit_correct_overall":          cit_correct_overall,
        "cit_correct_A": cit_correct["A"],
        "cit_correct_B": cit_correct["B"],
        "cit_correct_C": cit_correct["C"],
        "cit_correct_D": cit_correct["D"],
        "category_counts": cit_counts,
        "claims":        categorized,
        "judge_model":   cfg["judge_model"],
        "latency_sec":   round(time.time() - t0, 1),
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "usage": {
            "in_tokens":  per_tokens.in_tokens,
            "out_tokens": per_tokens.out_tokens,
            "cost_usd":   round(per_tokens.cost_usd, 6),
        },
        "cost": _get_cost_meter(),
    }

    with open(out_file, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Survey fully complete — the variant-specific AlignScore checkpoint can
    # be dropped. Keep the classification checkpoint: it is evidence-agnostic
    # and lets another factuality variant reuse the LLM classification pass.
    if alignscore_enabled:
        try:
            align_state_file.unlink(missing_ok=True)
        except OSError as e:
            logger.debug("could not remove state file %s: %s", align_state_file, e)

    if alignscore_enabled:
        logger.info(
            "[DONE] sid=%s — CitCorrect overall=%s A=%s B=%s C=%s D=%s  [%s]  (%.1fs)",
            survey_id,
            cit_correct_overall,
            cit_correct["A"], cit_correct["B"],
            cit_correct["C"], cit_correct["D"],
            per_tokens.fmt(), result["latency_sec"],
        )
    else:
        cat_dist = "  ".join(f"{k}={cit_counts[k]['n']}" for k in ["A", "B", "C", "D"])
        logger.info(
            "[DONE] sid=%s — categories: %s  (AlignScore disabled, %s, %.1fs)",
            survey_id, cat_dist, per_tokens.fmt(), result["latency_sec"],
        )
    return result


# ── Summary CSV ───────────────────────────────────────────────────────────────

def write_summary(results: list[dict], out_path: Path) -> None:
    flat = []
    for r in results:
        cost = r.get("cost") or {}
        flat.append({
            **r,
            "n_alignscore_calls": cost.get("n_alignscore_calls"),
            "n_alignscore_pairs": cost.get("n_alignscore_pairs"),
            "n_chunks_evaluated": cost.get("n_chunks_evaluated"),
            "n_abstract_skip":    cost.get("n_abstract_skip"),
        })
    fields = [
        "survey_id", "query", "n_claims", "n_supported",
        "cit_correct_overall",
        "cit_correct_A", "cit_correct_B", "cit_correct_C", "cit_correct_D",
        "latency_sec",
        "n_alignscore_calls", "n_alignscore_pairs",
        "n_chunks_evaluated", "n_abstract_skip",
    ]
    write_summary_csv(flat, out_path, fields, "factuality")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    setup_logging("factuality")

    # Reserve stderr for tqdm — drop the console StreamHandler that
    # setup_logging attached so nothing stomps the live bars. All log records
    # keep flowing to results/logs/factuality.log via the FileHandler.
    # FileHandler subclasses StreamHandler, so we guard by type.
    log_file: Path | None = None
    for h in list(logging.root.handlers):
        if isinstance(h, logging.FileHandler):
            log_file = Path(getattr(h, "baseFilename", "")) or log_file
        elif isinstance(h, logging.StreamHandler):
            logging.root.removeHandler(h)
    if log_file is not None:
        print(
            f"Logs → {log_file}  (tail -f to follow; stderr reserved for tqdm)",
            file=sys.stderr, flush=True,
        )

    parser = argparse.ArgumentParser(description="Factuality metrics (B.1)")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model",   required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only surveys with survey_id <= LIMIT "
                             "(inclusive, id-based — not positional; handles sparse sets "
                             "like SurGE_reference where ids may skip).")
    parser.add_argument("--id", type=int, default=None,
                        help="Process only this survey_id.")
    # Factcheck knobs — override config.yaml at the CLI.
    parser.add_argument("--claim-scope", choices=("paragraph", "section"), default=None,
                        help="Override cfg.claim_scope.")
    parser.add_argument(
        "--evidence-source",
        choices=("abstract", "full_text", "full_text_or_abstract"),
        default=None,
        help="Override cfg.evidence_source.",
    )
    parser.add_argument("--evidence-aggregation", choices=("concat", "per_ref"), default=None,
                        help="Override cfg.evidence_aggregation.")
    parser.add_argument("--max-ancestor-depth", type=int, default=None,
                        help="Override cfg.section_max_ancestor_depth (None in config = unlimited).")
    parser.add_argument("--debug-claim-idx", type=int, default=None,
                        help="For the claim at this index (0-based, within each survey) "
                             "log detailed scope / evidence / class / support to the log "
                             "file. Useful for auditing one specific claim end-to-end.")
    args = parser.parse_args()

    cfg    = load_config(CONFIG)
    client = make_client(cfg)

    # CLI → config overrides (CLI wins). Only apply when the flag was
    # explicitly passed — None means "keep config value".
    if args.claim_scope is not None:
        cfg["claim_scope"] = args.claim_scope
    if args.evidence_source is not None:
        cfg["evidence_source"] = args.evidence_source
    if args.evidence_aggregation is not None:
        cfg["evidence_aggregation"] = args.evidence_aggregation
    if args.max_ancestor_depth is not None:
        cfg["section_max_ancestor_depth"] = args.max_ancestor_depth
    # Pass --debug-claim-idx through the cfg dict to process_survey. Using
    # an underscore-prefixed key to signal "internal / not from config.yaml".
    if args.debug_claim_idx is not None:
        cfg["_debug_claim_idx"] = args.debug_claim_idx

    claims_dir = resolve_claims_dir(args.dataset, args.model)

    alignscore_enabled = cfg.get("alignscore_enabled", True)

    logger.info("[factuality] Loading models ...")
    alignscore_model = load_alignscore(cfg) if alignscore_enabled else None
    chunk_embedder = (
        _load_chunk_embedder(cfg)
        if alignscore_enabled and _full_text_topk_enabled(cfg)
        else None
    )
    if not alignscore_enabled:
        logger.info("AlignScore disabled — running LLM categorization only")
    corpus_index = build_corpus_index(args.dataset)

    # Load the abstract cache once; share across surveys. save_abstract_cache
    # at the end of the run (and periodically via fetch_evidence save_every).
    abstract_cache = load_abstract_cache()

    # Preflight: report the SS-tier configuration. SS tier is off by default
    # (cfg.ss_enabled=false) — without a key it's unusable (shared anon pool
    # ~100 req/5min/IP means 429 on the very first request). When explicitly
    # enabled we additionally report whether a key is present.
    _ss_key_env = cfg.get("ss_api_key_env", "SEMANTIC_SCHOLAR_API_KEY")
    _ss_key_val = os.environ.get(_ss_key_env) or ""
    _ss_enabled = bool(cfg.get("ss_enabled", False))
    if not _ss_enabled:
        logger.info(
            "SS tier: disabled (cfg.ss_enabled=false). Chain: "
            "cache → corpus → arxiv_api → miss."
        )
    elif _ss_key_val.strip():
        logger.info(
            "SS tier: enabled with API key (env=%s, len=%d). "
            "Chain: cache → corpus → arxiv_api → ss_api.",
            _ss_key_env, len(_ss_key_val),
        )
    else:
        logger.warning(
            "SS tier: enabled BUT no API key (env=%s unset). Anon SS "
            "traffic shares a ~100 req/5min/IP quota — expect 429 on "
            "essentially every call. Either set the key or flip "
            "cfg.ss_enabled to false to skip SS entirely.",
            _ss_key_env,
        )

    gen_dir = ROOT / "results" / "generations" / f"{args.dataset}_{args.model}"
    if not gen_dir.exists():
        print(f"[ERROR] Generation dir not found: {gen_dir}", file=sys.stderr)
        logger.error("Generation dir not found: %s", gen_dir)
        sys.exit(1)

    # Sources + checkpoint directories follow the same <dataset>_<model>
    # convention as gen_dir. Computed from CLI args
    # (not gen["dataset_id"]/["model_id"] fields), because some
    # generators — e.g. the SurGE_reference converter — write a combined
    # "SurGE_reference" as model_id, which would double-prefix otherwise.
    sources_dir = gen_dir / "sources"
    classification_checkpoint_dir = _classification_checkpoint_dir(
        cfg, args.dataset, args.model,
    )
    align_checkpoint_dir = _align_checkpoint_dir(cfg, args.dataset, args.model)

    # Output dir encodes the factcheck variant (scope × source × aggregation)
    # so comparing runs side-by-side doesn't require overwriting.
    run_id  = _run_id(cfg)
    variant = _variant_id(cfg)
    out_dir = (
        ROOT / "results" / "scores"
        / f"{args.dataset}_{args.model}_factuality_{run_id}_{variant}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Numeric-by-survey_id sort + id-based filter — same semantics as veriscore.
    # ``load_generation_files`` returns a lexical sort ("10.json" < "2.json");
    # re-sort so ``--limit N`` means "survey_id <= N" and navigation through
    # output files is natural on sparse sets (e.g. SurGE_reference).
    gen_files = load_generation_files(gen_dir)
    gen_files.sort(key=lambda p: int(p.stem) if p.stem.isdigit() else 10**9)

    if args.id is not None:
        gen_files = [f for f in gen_files if f.stem == str(args.id)]
    elif args.limit is not None:
        gen_files = [
            f for f in gen_files
            if f.stem.isdigit() and int(f.stem) <= args.limit
        ]

    banner = (
        f"[factuality] {args.dataset} / {args.model}\n"
        f"             {len(gen_files)} surveys → {out_dir}\n"
        f"             judge: {cfg['judge_model']}  "
        f"(alignscore: {'on' if alignscore_enabled else 'off'})"
    )
    print("\n" + banner + "\n")
    for line in banner.splitlines():
        logger.info(line)

    # Run-wide token counter — per-survey amounts are added to it after each
    # classification pass, so the outer bar can show cumulative cost.
    global_tokens = TokenCounter()

    all_results: list[dict] = []
    outer_bar = tqdm(
        total=len(gen_files),
        desc="surveys", unit="survey",
        leave=True, dynamic_ncols=True,
    )
    # Fail-fast policy — any exception on any survey stops the whole run
    # immediately. We still close the outer bar, flush the abstract cache,
    # and write a partial summary.csv for surveys completed so far, then
    # re-raise so the exit code / logs surface the failure clearly.
    fatal_exc: BaseException | None = None
    try:
        for gf in gen_files:
            outer_bar.set_postfix_str(f"{gf.stem}  {global_tokens.fmt()}")
            with open(gf) as f:
                gen = json.load(f)

            try:
                res = process_survey(
                    gen, claims_dir, out_dir, cfg,
                    client, alignscore_model, corpus_index,
                    sources_dir, classification_checkpoint_dir, align_checkpoint_dir,
                    chunk_embedder=chunk_embedder,
                    global_tokens=global_tokens,
                    abstract_cache=abstract_cache,
                )
            except Exception as e:
                # Log with full traceback and save the exception to re-raise
                # after the finally-block has run cleanup.
                logger.exception("[%s] failed: %s", gen.get("id", gf.stem), e)
                fatal_exc = e
                break

            if res:
                all_results.append(res)
                outer_bar.set_postfix_str(
                    f"{gf.stem} → n={res.get('n_claims','?')}  {global_tokens.fmt()}"
                )
            outer_bar.update(1)
    finally:
        outer_bar.close()

    # Persist shared abstract-cache once at the end (fetch_evidence also
    # writes it periodically in its save_every batches, but a final flush
    # covers the tail that didn't hit a save boundary).
    save_abstract_cache(abstract_cache)

    if all_results:
        write_summary(all_results, out_dir)

    if fatal_exc is not None:
        # Surface the original traceback by re-raising. The summary banner
        # below is skipped; the log already has the full context.
        logger.error(
            "[factuality] stopping run due to fatal error — "
            "%d surveys completed, %d remaining not processed",
            len(all_results), len(gen_files) - len(all_results),
        )
        raise fatal_exc

    n_err = len(gen_files) - len(all_results)

    summary_lines = [
        f"[factuality] done — ok={len(all_results)}  err={n_err}",
        (
            f"             usage: in={global_tokens.in_tokens}  "
            f"out={global_tokens.out_tokens}  "
            f"cost=${global_tokens.cost_usd:.6f}  ({global_tokens.fmt()})"
        ),
    ]
    print("\n" + "\n".join(summary_lines))
    for line in summary_lines:
        logger.info(line)


if __name__ == "__main__":
    main()
