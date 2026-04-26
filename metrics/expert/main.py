#!/usr/bin/env python3
"""
metrics/expert/main.py
Expert writing property metrics for generated surveys (group C).

Requires pre-computed Claimify output:
  results/scores/<dataset_id>_<model_id>_claims/

Fails with a clear error if the claims cache is absent.

For each survey, runs a single pass over the Claimify claim list and
computes four metrics via LLM-as-judge (C.1–C.4):

  M_crit       — fraction of claims that are critical (C.1)
  M_comp_total — fraction of claims that are comparative (C.2)
  M_comp_valid — fraction of comparative claims that are valid (C.2)
  M_open       — fraction of claims that formulate open questions (C.3)
  M_mod        — Shannon entropy of epistemic modality distribution (C.4)

Judges for C.1, C.2 (is-comparative), C.3, C.4 are parallelised per claim.
IsValidComparison (C.2) runs sequentially only for claims flagged as comparative.

Output per survey: results/scores/<dataset_id>_<model_id>_expert_<run_id>/<survey_id>.json
Summary:           .../summary.csv

Usage (inside Docker):
    python metrics/expert/main.py --dataset SurGE --model perplexity_dr
"""
import argparse
import json
import logging
import math
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

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
    resolve_claims_dir, load_generation_files, filter_by_limit,
    check_and_load_cache, write_summary_csv, llm_json_call,
)



# ── C.1–C.4 combined judge ────────────────────────────────────────────────────

_ALL_SYS = (
    "You are evaluating an atomic claim from a scientific survey along four independent dimensions. "
    "Analyse the claim carefully and answer all four parts in a single JSON object."
)

_ALL_USER = """\
The claim:
"{claim}"

Evaluate the claim on the four dimensions below and return a single JSON object with all keys.

## C.1 — Criticality
A claim is critical if it:
- Points out a limitation, weakness, or failure mode of a cited work
- Mentions a negative result or conditions under which a method fails
- Identifies a contradiction or tension between different works
- Questions the validity, generalizability, or scope of a claim
- Discusses trade-offs where one approach sacrifices something for another
A claim is NOT critical if it merely describes what a paper does, presents results neutrally, or states general facts.
Keys: "is_critical" (bool), "critical_type" ("limitation"|"negative_result"|"contradiction"|"trade_off"|"none"), "c1_reasoning" (brief)

## C.2 — Comparativeness
A claim is comparative if it explicitly compares two or more methods, models, approaches, or results using constructions like "outperforms", "in contrast to", "unlike", "better than", "whereas", or ranks/orders approaches.
A claim is NOT comparative if it only describes a single item or mentions multiple works without drawing contrasts.
Keys: "is_comparative" (bool), "compared_entities" (list of strings | null), "c2_reasoning" (brief)

## C.3 — Open question
A claim is an open question if it explicitly states something remains unknown/unresolved/disputed, points to gaps in understanding, identifies future research directions, or uses markers like "remains unclear", "open question", "further research is needed".
A claim is NOT an open question if it describes established results or uses hedging only for style.
Keys: "is_open_question" (bool), "question_type" ("generalization"|"scalability"|"mechanism"|"theoretical"|"empirical"|"none"), "c3_reasoning" (brief)

## C.4 — Epistemic modality
Classify the confidence level expressed by the claim's linguistic markers (not whether the claim is true):
1. Categorical — no hedging, uses "is", "does", "demonstrates"
2. Strong — mild qualifiers, uses "generally", "typically", "has been shown to"
3. Moderate — clear hedging, uses "often", "tends to", "can"
4. Weak — strong hedging, uses "may", "might", "could", "suggests", "appears to"
5. Explicit uncertainty — "remains unclear", "is debated", "open question"
Keys: "modality_level" (1–5), "hedging_markers" (list of strings), "c4_reasoning" (brief)

Respond with a single JSON object containing all keys listed above."""


def judge_all(
    claim: str,
    client: OpenAI,
    model: str,
    max_retries: int,
    provider: str | None = None,
    disable_reasoning: bool = False,
    token_counter: TokenCounter | None = None,
    reasoning_effort: str | None = None,
    max_tokens: int | None = None,
) -> dict:
    """Single LLM call evaluating C.1–C.4 for one claim.

    Returns a flat dict with all keys for criticality, comparativeness,
    open question, and modality. Falls back to safe defaults on parse error.
    """
    return llm_json_call(
        client, model,
        messages=[
            {"role": "system", "content": _ALL_SYS},
            {"role": "user",   "content": _ALL_USER.format(claim=claim[:600])},
        ],
        max_retries=max_retries,
        provider=provider,
        disable_reasoning=disable_reasoning,
        token_counter=token_counter,
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
        on_failure=lambda e: {},  # expert code expects {} on failure
    )


# ── C.2 validity (kept for future use, not included in metrics yet) ───────────

_VALID_SYS = "You are evaluating whether a comparative claim in a scientific survey makes a VALID comparison."

_VALID_USER = """\
The comparative claim:
"{claim}"

Source context (if available):
"{context}"

A comparison is valid if:
- The compared entities are actually comparable along the dimension being compared
- They were evaluated under comparable conditions (same dataset, same metric, similar experimental setup)
- The compared quantities are reported in the sources
- The comparison does not conflate different experimental setups

A comparison is INVALID if:
- The entities were evaluated on different datasets or benchmarks
- Different metrics are being compared as if they were the same
- Experimental conditions differ in ways that make direct comparison misleading
- The numbers or claims being compared are not supported by the sources

Respond with a JSON object:
{{"is_valid": true | false, "reasoning": "brief explanation", "invalidity_type": "incomparable_setup" | "different_metrics" | "unsupported_numbers" | "missing_context" | "none"}}"""


def judge_valid_comparison(claim: str, context: str, client: OpenAI, model: str, max_retries: int, provider: str | None = None, disable_reasoning: bool = False) -> dict:
    """Judge validity of a comparative claim given source context.

    Determines if a comparison is fair and properly supported (C.2 validity).
    Not yet wired into metric computation — kept for future use.
    """
    return llm_json_call(
        client, model,
        messages=[
            {"role": "system", "content": _VALID_SYS},
            {"role": "user", "content": _VALID_USER.format(claim=claim[:600], context=context[:800] if context else "Not available")},
        ],
        max_retries=max_retries,
        provider=provider,
        disable_reasoning=disable_reasoning,
        on_failure=lambda e: {},  # expert code expects {} on failure
    )


# ── Metric aggregation ────────────────────────────────────────────────────────

def compute_modality_entropy(modality_levels: list[int]) -> float:
    """Shannon entropy over modality distribution (levels 1–5)."""
    if not modality_levels:
        return 0.0
    counts = [modality_levels.count(k) for k in range(1, 6)]
    total  = sum(counts)
    probs  = [c / total for c in counts if c > 0]
    return round(-sum(p * math.log(p) for p in probs), 4)


# ── Per-claim parallel judging ────────────────────────────────────────────────

def judge_claim(
    claim_record: dict,
    client: OpenAI,
    model: str,
    max_retries: int,
    source_context: str,
    provider: str | None = None,
    disable_reasoning: bool = False,
    token_counter: TokenCounter | None = None,
    reasoning_effort: str | None = None,
    max_tokens: int | None = None,
) -> dict:
    """Run combined C.1–C.4 judge for a single claim. Returns enriched claim dict.

    Issues a single LLM call via judge_all() instead of four parallel ones.
    Validity check (C.2b) is not yet wired into metric computation.
    """
    claim = claim_record["claim"]

    res = judge_all(claim, client, model, max_retries, provider, disable_reasoning,
                    token_counter=token_counter,
                    reasoning_effort=reasoning_effort,
                    max_tokens=max_tokens)

    result = {
        **claim_record,
        # C.1
        "is_critical":      res.get("is_critical", False),
        "critical_type":    res.get("critical_type", "none"),
        # C.2
        "is_comparative":   res.get("is_comparative", False),
        "compared_entities": res.get("compared_entities"),
        # C.3
        "is_open_question": res.get("is_open_question", False),
        "question_type":    res.get("question_type", "none"),
        # C.4
        "modality_level":   res.get("modality_level", 1),
        "hedging_markers":  res.get("hedging_markers", []),
        # validity not yet in metrics — field absent intentionally
    }

    return result


# ── Per-survey processing ─────────────────────────────────────────────────────

def process_survey(
    gen: dict,
    claims_dir: Path,
    out_path: Path,
    cfg: dict,
    client: OpenAI,
) -> dict | None:
    survey_id = gen["id"]
    out_file  = out_path / f"{survey_id}.json"

    cached = check_and_load_cache(out_file, cfg, survey_id)
    if cached is not None:
        return cached

    if not gen.get("success", False):
        print(f"  [SKIP] {survey_id} — generation not successful")
        return None

    claims_file = claims_dir / f"{survey_id}.json"
    if not claims_file.exists():
        print(f"  [SKIP] {survey_id} — no claims file (run claimify first)")
        return None

    with open(claims_file) as f:
        claims_data = json.load(f)

    claims = claims_data.get("claims", [])
    if not claims:
        print(f"  [SKIP] {survey_id} — empty claims")
        return None

    print(f"  [PROC] {survey_id} | {gen.get('query', '')[:60]} | {len(claims)} claims")
    t0 = time.time()

    model             = cfg["judge_model"]
    max_retries       = cfg.get("max_retries", 3)
    provider          = cfg.get("judge_provider")
    disable_reasoning = cfg.get("judge_disable_reasoning", False)
    reasoning_effort  = cfg.get("judge_reasoning_effort") or None   # "low"|"medium"|"high", takes priority
    max_tokens        = cfg.get("judge_max_tokens") or None
    workers           = cfg.get("judge_workers", 4)
    # Shared source context (first reference abstract) for validity checks
    source_ctx = ""  # enriched per model if corpus available

    token_counter = TokenCounter()
    judged_claims: list[dict | None] = [None] * len(claims)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                judge_claim, cr, client, model, max_retries,
                source_ctx, provider, disable_reasoning, token_counter,
                reasoning_effort, max_tokens,
            ): i
            for i, cr in enumerate(claims)
        }
        with tqdm(total=len(claims), desc="  judging", unit="claim", leave=True) as bar:
            for future in as_completed(futures):
                i = futures[future]
                try:
                    judged_claims[i] = future.result()
                except Exception as e:
                    judged_claims[i] = {**claims[i], "_judge_error": str(e)}
                bar.update(1)
                bar.set_postfix_str(token_counter.fmt())

    # ── Compute metrics ────────────────────────────────────────────────────
    n = len(judged_claims)

    m_crit       = round(sum(1 for c in judged_claims if c.get("is_critical")) / n, 4) if n else 0.0
    m_comp_total = round(sum(1 for c in judged_claims if c.get("is_comparative")) / n, 4) if n else 0.0
    comparative  = [c for c in judged_claims if c.get("is_comparative")]

    # m_comp_valid not yet computed — judge_valid_comparison not wired in
    m_comp_valid = None

    m_open = round(sum(1 for c in judged_claims if c.get("is_open_question")) / n, 4) if n else 0.0

    modality_levels = [c.get("modality_level", 1) for c in judged_claims]
    m_mod = compute_modality_entropy(modality_levels)

    # Modality distribution
    modality_dist = {str(k): modality_levels.count(k) for k in range(1, 6)}

    result = {
        "survey_id":    survey_id,
        "dataset_id":   gen["dataset_id"],
        "model_id":     gen["model_id"],
        "query":        gen.get("query", ""),
        "n_claims":     n,
        # C.1
        "m_crit":       m_crit,
        "n_critical":   sum(1 for c in judged_claims if c.get("is_critical")),
        # C.2 (validity not yet included)
        "m_comp_total": m_comp_total,
        "n_comparative": len(comparative),
        # C.3
        "m_open":       m_open,
        "n_open":       sum(1 for c in judged_claims if c.get("is_open_question")),
        # C.4
        "m_mod":        m_mod,
        "modality_dist": modality_dist,
        # Raw claims
        "claims":        judged_claims,
        "judge_model":   model,
        "latency_sec":   round(time.time() - t0, 1),
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    }

    with open(out_file, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(
        f"         m_crit={m_crit}  m_comp={m_comp_total}  "
        f"m_open={m_open}  m_mod={m_mod}  "
        f"({result['latency_sec']}s)"
    )
    return result


# ── Summary CSV ───────────────────────────────────────────────────────────────

def write_summary(results: list[dict], out_path: Path) -> None:
    fields = [
        "survey_id", "query", "n_claims",
        "m_crit", "n_critical",
        "m_comp_total", "n_comparative",
        "m_open", "n_open",
        "m_mod",
        "latency_sec",
    ]
    write_summary_csv(results, out_path, fields, "expert")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    setup_logging("expert")
    parser = argparse.ArgumentParser(description="Expert writing property metrics (C.1–C.4)")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model",   required=True)
    parser.add_argument("--limit",   type=int, default=None,
                        help="Process only surveys with survey_id <= LIMIT "
                             "(inclusive, id-based — not positional).")
    args = parser.parse_args()

    cfg    = load_config(CONFIG)
    client = make_client(cfg)

    claims_dir = resolve_claims_dir(args.dataset, args.model)

    gen_dir = ROOT / "results" / "generations" / f"{args.dataset}_{args.model}"
    if not gen_dir.exists():
        print(f"[ERROR] Generation dir not found: {gen_dir}", file=sys.stderr)
        sys.exit(1)

    run_id  = f"{cfg['judge_id']}_{cfg['judge_comment']}"
    out_dir = ROOT / "results" / "scores" / f"{args.dataset}_{args.model}_expert_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_files = filter_by_limit(load_generation_files(gen_dir), args.limit)

    print(f"\n[expert] {args.dataset} / {args.model}")
    print(f"         {len(gen_files)} surveys → {out_dir}\n")

    all_results = []
    for gf in gen_files:
        with open(gf) as f:
            gen = json.load(f)

        res = process_survey(gen, claims_dir, out_dir, cfg, client)
        if res:
            all_results.append(res)

    if all_results:
        write_summary(all_results, out_dir)

    print(f"\n[expert] done — ok={len(all_results)}")


if __name__ == "__main__":
    main()
