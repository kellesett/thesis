#!/usr/bin/env python3
"""
metrics/claimify/main.py
Atomic claim extraction for generated surveys using the Claimify pipeline.

Implements Metropolitansky & Larson, ACL 2025 (arXiv:2502.10855):
  Sentence Splitting → Selection → Disambiguation → Decomposition

Results are saved to:
  results/scores/<dataset_id>_<model_id>_claims/<survey_id>.json

Format:
  {
    "survey_id": 0,
    "dataset_id": "SurGE",
    "model_id":   "perplexity_dr",
    "query":      "...",
    "n_sentences": 142,
    "n_claims":    312,
    "claims": [
      {"claim_id": 0, "claim": "...", "source_sentence": "..."},
      ...
    ],
    "judge_model": "openai/gpt-4o-mini",
    "timestamp":   "2026-04-11T..."
  }

factuality/ and expert/ metrics read this cache and fail if it's absent.

Usage (inside Docker):
    python metrics/claimify/main.py --dataset SurGE --model perplexity_dr
"""
import argparse
import asyncio
import json
import logging
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

load_dotenv()

logger = logging.getLogger(__name__)

ROOT   = Path(__file__).parent.parent.parent
CONFIG = Path(__file__).parent / "config.yaml"
sys.path.insert(0, str(ROOT))

from src.log_setup import setup_logging

from metrics.claimify.claim_extractor import (
    ClaimExtractor,
    _SEL_COMPLETIONS,
    _DIS_COMPLETIONS,
)
from metrics.utils import load_config, check_and_load_cache, load_generation_files, filter_by_limit


# ── Client creation ───────────────────────────────────────────────────────────

def make_client(cfg: dict) -> AsyncOpenAI:
    """Create AsyncOpenAI client for Claimify pipeline.

    Args:
        cfg: Configuration dict with judge_api_key_env and judge_base_url.

    Returns:
        Initialized AsyncOpenAI client.

    Raises:
        RuntimeError: If API key environment variable not set.
    """
    api_key_env = cfg.get("judge_api_key_env")
    if not api_key_env:
        raise RuntimeError("Config missing 'judge_api_key_env'")
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise RuntimeError(
            f"API key not set: env var '{api_key_env}'"
        )
    return AsyncOpenAI(api_key=api_key, base_url=cfg["judge_base_url"])


# ── Per-survey processing ─────────────────────────────────────────────────────

def _make_bars(desc: str, total: int | None, unit: str) -> "tqdm":
    return tqdm(total=total, desc=desc, unit=unit, leave=True, dynamic_ncols=True)


async def process_survey(
    gen: dict,
    out_path: Path,
    tmp_path: Path,
    cfg: dict,
    extractor: ClaimExtractor,
    surveys_bar,
) -> dict | None:
    """Run Claimify pipeline on one survey, stage by stage.

    Intermediate results are cached in tmp_path/<survey_id>/.
    Stages: Selection → Disambiguation → Decomposition.

    Args:
        gen: Generation dict with "id", "text", "query", "success", etc.
        out_path: Output directory for final claims JSON.
        tmp_path: Temporary directory for intermediate stage caches.
        cfg: Configuration dict with model, max_concurrent, etc.
        extractor: Initialized ClaimExtractor instance.
        surveys_bar: tqdm progress bar for survey tracking.

    Returns:
        Summary dict with claims and metadata, or None on skip/error.
    """
    survey_id = str(gen["id"])
    out_file  = out_path / f"{survey_id}.json"

    cached = check_and_load_cache(out_file, cfg, survey_id, required_keys=("survey_id", "claims"))
    if cached is not None:
        tqdm.write(f"  [SKIP] {survey_id} — {cached['n_claims']} claims already saved", file=sys.stderr)
        return cached

    if not gen.get("success", False):
        logger.info(f"{survey_id} — generation not successful, skipping")
        tqdm.write(f"  [SKIP] {survey_id} — generation not successful", file=sys.stderr)
        return None

    text = gen.get("text", "").strip()
    if not text:
        logger.info(f"{survey_id} — empty text, skipping")
        tqdm.write(f"  [SKIP] {survey_id} — empty text", file=sys.stderr)
        return None

    question       = gen.get("query", "")
    max_concurrent = cfg.get("max_concurrent", 5)
    survey_tmp     = tmp_path / survey_id
    survey_tmp.mkdir(parents=True, exist_ok=True)

    sel_cache = survey_tmp / "selection.json"
    dis_cache = survey_tmp / "disambiguation.json"

    tqdm.write(f"  [PROC] {survey_id} | {question[:70]}", file=sys.stderr)

    sentences = extractor.split_sentences(text)
    n         = len(sentences)

    # ── Stage 2: Selection ─────────────────────────────────────────────────────
    if sel_cache.exists():
        tqdm.write(f"         [cache] selection ({sel_cache})", file=sys.stderr)
        selected = json.loads(sel_cache.read_text())
    else:
        sel_sent_bar = _make_bars("  sel sents", total=n,                   unit="sent")
        sel_llm_bar  = _make_bars("  sel LLM  ", total=n * _SEL_COMPLETIONS, unit="call")
        selected = await extractor.run_selection(
            question, sentences, max_concurrent,
            bars={"sel_sent": sel_sent_bar, "sel_llm": sel_llm_bar},
        )
        sel_sent_bar.close()
        sel_llm_bar.close()
        sel_cache.write_text(json.dumps(selected, ensure_ascii=False))

    n_selected = sum(1 for s in selected if s is not None)
    tqdm.write(f"         sel: {n_selected}/{n} sentences passed", file=sys.stderr)

    # ── Stage 3: Disambiguation ────────────────────────────────────────────────
    if dis_cache.exists():
        tqdm.write(f"         [cache] disambiguation ({dis_cache})", file=sys.stderr)
        disambiguated = json.loads(dis_cache.read_text())
    else:
        dis_sent_bar = _make_bars("  dis sents", total=n_selected,                   unit="sent")
        dis_llm_bar  = _make_bars("  dis LLM  ", total=n_selected * _DIS_COMPLETIONS, unit="call")
        disambiguated = await extractor.run_disambiguation(
            question, sentences, selected, max_concurrent,
            bars={"dis_sent": dis_sent_bar, "dis_llm": dis_llm_bar},
        )
        dis_sent_bar.close()
        dis_llm_bar.close()
        dis_cache.write_text(json.dumps(disambiguated, ensure_ascii=False))

    n_disambiguated = sum(1 for s in disambiguated if s is not None)
    tqdm.write(f"         dis: {n_disambiguated}/{n_selected} sentences passed", file=sys.stderr)

    # ── Stage 4: Decomposition ─────────────────────────────────────────────────
    dec_sent_bar = _make_bars("  dec sents", total=n_disambiguated, unit="sent")
    dec_llm_bar  = _make_bars("  dec LLM  ", total=n_disambiguated, unit="call")
    claims_nested = await extractor.run_decomposition(
        question, sentences, disambiguated, max_concurrent,
        bars={"dec_sent": dec_sent_bar, "dec_llm": dec_llm_bar},
    )
    dec_sent_bar.close()
    dec_llm_bar.close()

    claims_raw = [c for claims in claims_nested for c in claims]
    claims = [
        {"claim_id": i, "claim": c, "source_sentence": ""}
        for i, c in enumerate(claims_raw)
    ]

    result = {
        "survey_id":   survey_id,
        "dataset_id":  gen["dataset_id"],
        "model_id":    gen["model_id"],
        "query":       question,
        "n_sentences": n,
        "n_selected":  n_selected,
        "n_disambiguated": n_disambiguated,
        "n_claims":    len(claims),
        "claims":      claims,
        "judge_model": cfg["judge_model"],
        "pipeline":    "claimify",
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }

    with open(out_file, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    tqdm.write(f"         → {len(claims)} claims", file=sys.stderr)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

async def main_async() -> None:
    setup_logging("claimify")
    parser = argparse.ArgumentParser(description="Claimify — atomic claim extraction")
    parser.add_argument("--dataset", required=True, help="Dataset id (e.g. SurGE)")
    parser.add_argument("--model",   required=True, help="Model id (e.g. perplexity_dr)")
    parser.add_argument("--limit",   type=int, default=None,
                        help="Process only surveys with survey_id <= LIMIT "
                             "(inclusive, id-based — not positional).")
    args = parser.parse_args()

    cfg       = load_config(CONFIG)
    client    = make_client(cfg)
    extractor = ClaimExtractor(
        client,
        model_name=cfg["judge_model"],
        provider=cfg.get("judge_provider") or None,
        reasoning_effort=cfg.get("judge_reasoning_effort") or None,
        max_tokens=cfg.get("judge_max_tokens") or None,
    )

    gen_dir = ROOT / "results" / "generations" / f"{args.dataset}_{args.model}"
    if not gen_dir.exists():
        logger.error(f"Generation dir not found: {gen_dir}")
        sys.exit(1)

    out_dir = ROOT / "results" / "scores" / f"{args.dataset}_{args.model}_claims"
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_files = filter_by_limit(load_generation_files(gen_dir), args.limit)

    print(f"\n[claimify] {args.dataset} / {args.model}")
    print(f"           {len(gen_files)} surveys → {out_dir}")
    print(f"           model: {cfg['judge_model']}")
    print(f"           max_concurrent: {cfg.get('max_concurrent', 5)}\n")

    run_id   = f"{args.dataset}_{args.model}"
    tmp_path = Path("/tmp") / "claimify" / run_id
    tmp_path.mkdir(parents=True, exist_ok=True)
    tqdm.write(f"           tmp:  {tmp_path}\n", file=sys.stderr)

    n_ok, n_skip = 0, 0
    total_claims = 0

    surveys_bar = tqdm(
        total=len(gen_files), desc="surveys", unit="survey",
        leave=True, dynamic_ncols=True,
    )

    for gf in gen_files:
        surveys_bar.set_postfix_str(gf.stem)
        with open(gf) as f:
            gen = json.load(f)

        result = await process_survey(gen, out_dir, tmp_path, cfg, extractor, surveys_bar)
        if result is None:
            n_skip += 1
        else:
            n_ok += 1
            total_claims += result["n_claims"]
            surveys_bar.set_postfix_str(f"{gf.stem} → {result['n_claims']} claims")
        surveys_bar.update(1)

    surveys_bar.close()

    print(f"\n[claimify] done — ok={n_ok} skip={n_skip}")
    print(f"           total claims: {total_claims}")
    if n_ok > 0:
        print(f"           avg per survey: {total_claims // n_ok}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
