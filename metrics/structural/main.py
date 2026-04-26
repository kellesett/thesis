#!/usr/bin/env python3
"""
metrics/structural/main.py
Structural quality metrics for generated surveys (group A).

Computes two metrics per survey:

  M_contr  — cross-section contradiction rate (A.1)
             SPECTER similarity candidates → LLM topic filter → LLM contradiction check.
             Lower is better (0 = no contradictions).

  M_rep    — cross-section repetition rate (A.3)
             SPECTER pre-filter → bi-directional NLI entailment.
             Lower is better.

Output per survey: results/scores/<dataset_id>_<model_id>_structural/<survey_id>.json
Summary:           results/scores/<dataset_id>_<model_id>_structural/summary.csv

Usage (inside Docker):
    python metrics/structural/main.py --dataset SurGE --model perplexity_dr
"""
import argparse
import csv
import itertools
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

ROOT   = Path(__file__).parent.parent.parent
CONFIG = Path(__file__).parent / "config.yaml"
sys.path.insert(0, str(ROOT))

import diskcache
from tqdm import tqdm

from src.log_setup import setup_logging
from metrics.utils import (
    make_client, load_config, check_and_load_cache,
    load_generation_files, filter_by_limit,
    write_summary_csv,
)
from metrics.structural.contradiction.aggregate import compute_m_contr


# ── Hyperparams ID ────────────────────────────────────────────────────────────

def build_hyperparams_id(cfg: dict) -> str:
    """Short human-readable key that captures cache-relevant hyperparameters.

    Used as part of the intermediate stage path so that runs with different
    thresholds or judge models don't share cached intermediate results.
    """
    sim = cfg.get("similarity_threshold", 0.6)
    rp  = cfg.get("rep_embedding_prefilter", 0.7)
    jid = cfg.get("judge_id", "judge")
    return f"sim{sim}_rp{rp}_{jid}"


# ── Model loading ─────────────────────────────────────────────────────────────

def load_nli_model(path: str):
    """Load DeBERTa-v2-xlarge-mnli as a text-classification pipeline."""
    from transformers import pipeline
    print(f"  Loading NLI: {path}")
    return pipeline(
        "text-classification",
        model=path,
        device=-1,
        top_k=None,
    )


def load_specter(path: str):
    """Load SPECTER sentence-transformers model for embeddings."""
    from sentence_transformers import SentenceTransformer
    print(f"  Loading SPECTER: {path}")
    return SentenceTransformer(path, device="cpu")


# ── Text utilities ────────────────────────────────────────────────────────────

_HEADING_RE = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)
_SENT_RE    = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_to_sentences(text: str) -> list[str]:
    """Naive sentence splitter — good enough for scientific English."""
    # First strip citation markers
    text = re.sub(r"\[[^\]]{1,40}\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in _SENT_RE.split(text) if len(s.strip()) > 30]


def split_sections(text: str) -> list[dict]:
    """Return list of {"title": str, "sentences": list[str]}."""
    headings = list(_HEADING_RE.finditer(text))
    sections = []
    for i, m in enumerate(headings):
        title = m.group(1).strip()
        start = m.end()
        end   = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body  = text[start:end].strip()
        sents = split_to_sentences(body)
        if sents:
            sections.append({"title": title, "sentences": sents})
    if not sections:
        sents = split_to_sentences(text)
        sections = [{"title": "Full text", "sentences": sents}]
    return sections


# ── NLI helpers ───────────────────────────────────────────────────────────────

def nli_scores(nli_pipe, premise: str, hypothesis: str) -> dict:
    """
    Run NLI and return dict of label→score.
    DeBERTa-xlarge-mnli labels: ENTAILMENT, NEUTRAL, CONTRADICTION.
    """
    result = nli_pipe(f"{premise} [SEP] {hypothesis}", truncation=True, max_length=512)
    # result is a list of [{"label": ..., "score": ...}]
    if isinstance(result[0], list):
        result = result[0]
    return {r["label"].upper(): r["score"] for r in result}



# ── A.3 — Repetition detection ────────────────────────────────────────────────

def compute_m_rep(
    sections: list[dict],
    specter_model,
    nli_pipe,
    cfg: dict,
    stage_dir: Path,
    rep_bar=None,
    nli_bar=None,
) -> dict:
    """Compute cross-section repetition rate (A.3).

    Stage 4: SPECTER pre-filter → candidate pairs  → candidates.json
    Stage 5: Bi-directional NLI entailment check   → duplicates.json

    Each stage loads from disk if its output file already exists (resume support).

    Args:
        sections:      List of sections with sentences.
        specter_model: SPECTER sentence encoder for pre-filtering.
        nli_pipe:      NLI pipeline for entailment verification.
        cfg:           Config with similarity and entailment thresholds.
        stage_dir:     Directory for intermediate JSON files.
        rep_bar:       Optional tqdm bar for stage 4 (SPECTER pair iteration).
        nli_bar:       Optional tqdm bar for stage 5 (NLI check).

    Returns:
        Dict with m_rep (rate), n_total_sentences, n_candidates, n_duplicates.
    """
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    stage_dir.mkdir(parents=True, exist_ok=True)

    emb_threshold = cfg.get("rep_embedding_prefilter", 0.7)
    nli_threshold = cfg.get("rep_nli_threshold", 0.5)

    # Flatten sentences — always needed for final index resolution
    all_sents: list[tuple[str, str]] = [
        (sent, sec["title"])
        for sec in sections
        for sent in sec["sentences"]
    ]
    n_total = len(all_sents)
    if n_total < 2:
        for bar in [rep_bar, nli_bar]:
            if bar is not None:
                bar.reset(total=1)
                bar.update(1)
                bar.set_postfix_str("n/a (< 2 sentences)")
        return {"m_rep": 0.0, "n_total_sentences": n_total, "n_candidates": 0, "n_duplicates": 0}

    texts  = [s for s, _ in all_sents]
    titles = [t for _, t in all_sents]
    n_pairs = n_total * (n_total - 1) // 2

    # ── Stage 4 — SPECTER rep candidates ─────────────────────────────────────
    cands_file = stage_dir / "candidates.json"
    if cands_file.exists():
        rep_candidates = json.loads(cands_file.read_text())
        if rep_bar is not None:
            rep_bar.reset(total=n_pairs)
            rep_bar.update(n_pairs)
            rep_bar.set_postfix_str(f"→ {len(rep_candidates)} sel (cached)")
    else:
        embs = specter_model.encode(texts, batch_size=64, show_progress_bar=False)
        sims = cos_sim(embs, embs)

        if rep_bar is not None:
            rep_bar.reset(total=n_pairs)

        rep_candidates = []
        for i in range(n_total):
            for j in range(i + 1, n_total):
                if rep_bar is not None:
                    rep_bar.update(1)
                if titles[i] == titles[j]:
                    continue
                if sims[i, j] >= emb_threshold:
                    rep_candidates.append({
                        "i": i, "j": j,
                        "s1": texts[i], "s2": texts[j],
                        "section_i": titles[i], "section_j": titles[j],
                        "similarity": round(float(sims[i, j]), 4),
                    })

        if rep_bar is not None:
            rep_bar.set_postfix_str(f"→ {len(rep_candidates)} sel")

        cands_file.write_text(json.dumps(rep_candidates, ensure_ascii=False))

    # ── Stage 5 — Bi-directional NLI ─────────────────────────────────────────
    dup_file = stage_dir / "duplicates.json"
    if dup_file.exists():
        dup_results = json.loads(dup_file.read_text())
        n_dup = sum(1 for r in dup_results if r.get("is_duplicate"))
        if nli_bar is not None:
            n_cands = max(len(rep_candidates), 1)
            nli_bar.reset(total=n_cands)
            nli_bar.update(n_cands)
            nli_bar.set_postfix_str(f"{n_dup}/{len(rep_candidates)} dup (cached)")
    else:
        if nli_bar is not None:
            nli_bar.reset(total=max(len(rep_candidates), 1))

        dup_results = []
        n_dup = 0
        for cand in rep_candidates:
            try:
                fwd = nli_scores(nli_pipe, cand["s1"], cand["s2"])
                bwd = nli_scores(nli_pipe, cand["s2"], cand["s1"])
                is_dup = (
                    fwd.get("ENTAILMENT", 0) >= nli_threshold and
                    bwd.get("ENTAILMENT", 0) >= nli_threshold
                )
                dup_results.append({**cand, "is_duplicate": is_dup})
                if is_dup:
                    n_dup += 1
            except Exception:
                raise
            finally:
                if nli_bar is not None:
                    nli_bar.update(1)
                    nli_bar.set_postfix_str(f"{n_dup}/{nli_bar.n} dup")

        dup_file.write_text(json.dumps(dup_results, ensure_ascii=False))

    duplicate_indices: set[int] = set()
    for r in dup_results:
        if r.get("is_duplicate"):
            duplicate_indices.add(r["i"])
            duplicate_indices.add(r["j"])

    m_rep = len(duplicate_indices) / n_total if n_total else 0.0
    return {
        "m_rep":             round(m_rep, 4),
        "n_total_sentences": n_total,
        "n_candidates":      len(rep_candidates),
        "n_duplicates":      len(duplicate_indices),
    }


# ── Per-survey processing ─────────────────────────────────────────────────────

def process_survey(
    gen: dict,
    out_path: Path,
    cfg: dict,
    nli_pipe,
    specter_model,
    client: OpenAI,
    cache,
    stage_base: Path,
    hparams_id: str,
    specter_bar=None,
    topic_bar=None,
    contr_bar=None,
    rep_bar=None,
    nli_bar=None,
) -> dict | None:
    survey_id  = gen["id"]
    dataset_id = gen["dataset_id"]
    model_id   = gen["model_id"]
    out_file   = out_path / f"{survey_id}.json"

    cached = check_and_load_cache(out_file, cfg, survey_id)
    if cached is not None:
        return cached

    if not gen.get("success", False):
        tqdm.write(f"  [SKIP] {survey_id} — generation not successful")
        return None

    text = gen.get("text", "").strip()
    if not text:
        tqdm.write(f"  [SKIP] {survey_id} — empty text")
        return None

    t0 = time.time()

    sections = split_sections(text)
    n_sents  = sum(len(s["sentences"]) for s in sections)
    tqdm.write(f"  [PROC] {survey_id} | {len(sections)} sec / {n_sents} sent")

    contr_stage_dir = stage_base / "contradiction" / dataset_id / model_id / hparams_id / survey_id
    rep_stage_dir   = stage_base / "repetition"    / dataset_id / model_id / hparams_id / survey_id

    try:
        contr = compute_m_contr(
            survey_id, sections, specter_model, client, cfg, cache, contr_stage_dir,
            specter_bar=specter_bar, topic_bar=topic_bar, contr_bar=contr_bar,
        )
    except Exception:
        logger.exception(f"Error computing m_contr for {survey_id}")
        raise

    try:
        rep = compute_m_rep(
            sections, specter_model, nli_pipe, cfg, rep_stage_dir,
            rep_bar=rep_bar, nli_bar=nli_bar,
        )
    except Exception:
        logger.exception(f"Error computing m_rep for {survey_id}")
        raise

    result = {
        "survey_id":   survey_id,
        "dataset_id":  gen["dataset_id"],
        "model_id":    gen["model_id"],
        "query":       gen.get("query", ""),
        "n_sections":  len(sections),
        "n_sentences": n_sents,
        **{f"contr_{k}": v for k, v in contr.items()},
        **{f"rep_{k}":   v for k, v in rep.items()},
        "latency_sec": round(time.time() - t0, 1),
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }

    with open(out_file, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    tqdm.write(
        f"         ✓ m_contr={contr.get('m_contr')}  "
        f"(cands={contr.get('n_candidates_stage1')} "
        f"topic={contr.get('n_after_topic_filter')} "
        f"confirmed={contr.get('n_contradictions')})  "
        f"m_rep={rep.get('m_rep')}  "
        f"({result['latency_sec']}s)"
    )
    return result


# ── Summary CSV ───────────────────────────────────────────────────────────────

def write_summary(results: list[dict], out_path: Path) -> None:
    fields = [
        "survey_id", "query",
        "contr_m_contr", "contr_n_candidates_stage1",
        "contr_n_after_topic_filter", "contr_n_contradictions", "contr_n_failed",
        "rep_m_rep", "rep_n_total_sentences", "rep_n_duplicates",
        "latency_sec",
    ]
    write_summary_csv(results, out_path, fields, "structural")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    setup_logging("structural")
    parser = argparse.ArgumentParser(description="Structural quality metrics (A.1, A.2, A.3)")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model",   required=True)
    parser.add_argument("--limit",   type=int, default=None,
                        help="Process only surveys with survey_id <= LIMIT "
                             "(inclusive, id-based — not positional).")
    args = parser.parse_args()

    cfg    = load_config(CONFIG)
    client = make_client(cfg)

    print("\n[structural] Loading models...")
    nli_pipe      = load_nli_model(cfg["nli_model_path"])
    specter_model = load_specter(cfg["specter_model_path"])

    cache_dir = Path(cfg.get("cache_dir", "/tmp/structural/cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = diskcache.Cache(str(cache_dir))

    # Intermediate stage files live one level up from the diskcache dir
    stage_base = cache_dir.parent
    hparams_id = build_hyperparams_id(cfg)
    print(f"             hparams_id={hparams_id}")

    gen_dir = ROOT / "results" / "generations" / f"{args.dataset}_{args.model}"
    if not gen_dir.exists():
        print(f"[ERROR] Generation dir not found: {gen_dir}", file=sys.stderr)
        sys.exit(1)

    run_id  = f"{cfg['judge_id']}_{cfg['judge_comment']}"
    out_dir = ROOT / "results" / "scores" / f"{args.dataset}_{args.model}_structural_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_files = filter_by_limit(load_generation_files(gen_dir), args.limit)

    tqdm.write(f"\n[structural] {args.dataset} / {args.model}")
    tqdm.write(f"             {len(gen_files)} surveys → {out_dir}\n")

    # ── Progress bars (all leave=True, reset between surveys) ─────────────────
    # position=5 = top line, position=0 = bottom line
    survey_bar  = tqdm(total=len(gen_files), desc="Surveys         ", position=0, leave=True, unit="survey")
    specter_bar = tqdm(total=1, desc="  1/5 SPECTER   ", position=1, leave=True, unit="pair")
    topic_bar   = tqdm(total=1, desc="  2/5 topic flt ", position=2, leave=True, unit="pair")
    contr_bar   = tqdm(total=1, desc="  3/5 contr chk ", position=3, leave=True, unit="pair")
    rep_bar     = tqdm(total=1, desc="  4/5 rep SPECTER", position=4, leave=True, unit="pair")
    nli_bar     = tqdm(total=1, desc="  5/5 NLI       ", position=5, leave=True, unit="pair")

    all_results = []
    try:
        for gf in gen_files:
            with open(gf) as f:
                gen = json.load(f)

            survey_bar.set_postfix_str(gen.get("query", "")[:55])

            res = process_survey(
                gen, out_dir, cfg, nli_pipe, specter_model, client, cache,
                stage_base, hparams_id,
                specter_bar=specter_bar, topic_bar=topic_bar, contr_bar=contr_bar,
                rep_bar=rep_bar, nli_bar=nli_bar,
            )
            survey_bar.update(1)
            if res:
                all_results.append(res)

            # Reset inner bars — clears postfix so next survey starts fresh
            for bar in [specter_bar, topic_bar, contr_bar, rep_bar, nli_bar]:
                bar.reset(total=1)
                bar.set_postfix_str("")
    finally:
        for bar in [nli_bar, rep_bar, contr_bar, topic_bar, specter_bar, survey_bar]:
            bar.close()

    if all_results:
        write_summary(all_results, out_dir)

    ok = len(all_results)
    n_err = len(gen_files) - ok
    print(f"\n[structural] done — ok={ok}  err={n_err}")


if __name__ == "__main__":
    main()
