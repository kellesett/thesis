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
import csv
import json
import logging
import math
import os
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

from metrics.utils import make_client, load_config


# ── Config & client ───────────────────────────────────────────────────────────


# ── Claimify cache guard ──────────────────────────────────────────────────────

def resolve_claims_dir(dataset: str, model: str) -> Path:
    claims_dir = ROOT / "results" / "scores" / f"{dataset}_{model}_claims"
    if not claims_dir.exists() or not any(claims_dir.glob("*.json")):
        print(
            f"\n[ERROR] Claimify cache not found at:\n  {claims_dir}\n\n"
            f"Run first:\n"
            f"  make evaluate DATASET={dataset} MODEL={model} METRIC=claimify\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return claims_dir


# ── LLM call helper ───────────────────────────────────────────────────────────

def llm_json(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    max_retries: int,
    provider: str | None = None,
    disable_reasoning: bool = False,
) -> dict:
    """Call LLM, parse JSON response. Returns {} on persistent failure."""
    extra_body: dict = {}
    if provider:
        extra_body["provider"] = {"order": [provider], "allow_fallbacks": False}
    if disable_reasoning:
        extra_body["reasoning"] = {"enabled": False}
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0,
                extra_body=extra_body or None,
            )
            content = resp.choices[0].message.content
            if content is None:
                print(f"[ERROR] OpenRouter None content (finish_reason={resp.choices[0].finish_reason}).\n"
                      f"Full response: {resp}", file=sys.stderr)
                raise RuntimeError(
                    f"OpenRouter returned None content "
                    f"(finish_reason={resp.choices[0].finish_reason})."
                )
            raw = content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                return {"_error": str(e)}
    return {}


# ── C.1 — Criticality ────────────────────────────────────────────────────────

_CRIT_SYS = "You are evaluating whether an atomic claim from a scientific survey is critical in nature."

_CRIT_USER = """\
The claim:
"{claim}"

A claim is considered critical if it does one of the following:
- Points out a limitation, weakness, or failure mode of a cited work
- Mentions a negative result or conditions under which a method fails
- Identifies a contradiction or tension between different works
- Questions the validity, generalizability, or scope of a claim
- Discusses trade-offs where one approach sacrifices something for another

A claim is NOT critical if it:
- Merely describes what a paper does
- Presents results neutrally without evaluation
- Compares approaches without judgment about their relative merits
- States general facts about a research area

Respond with a JSON object:
{{"is_critical": true | false, "reasoning": "brief explanation", "critical_type": "limitation" | "negative_result" | "contradiction" | "trade_off" | "none"}}"""


def judge_criticality(claim: str, client: OpenAI, model: str, max_retries: int, provider: str | None = None, disable_reasoning: bool = False) -> dict:
    """Judge whether a claim makes a critical evaluation or judgment.

    Identifies claims that point out limitations, negative results, contradictions,
    or trade-offs (C.1).
    """
    return llm_json(client, model, _CRIT_SYS, _CRIT_USER.format(claim=claim[:600]), max_retries, provider, disable_reasoning)


# ── C.2 — Comparativeness ────────────────────────────────────────────────────

_COMP_SYS = "You are evaluating whether an atomic claim makes an explicit comparison between multiple works or approaches."

_COMP_USER = """\
The claim:
"{claim}"

A claim is comparative if it:
- Explicitly compares two or more methods, models, approaches, or results
- Uses comparative constructions like "outperforms", "in contrast to", "unlike", "better than", "more than", "whereas"
- Contrasts characteristics of different works
- Ranks or orders approaches

A claim is NOT comparative if it:
- Only describes a single method or result
- Mentions multiple works without comparing them
- Lists related approaches without drawing contrasts

Respond with a JSON object:
{{"is_comparative": true | false, "reasoning": "brief explanation", "compared_entities": ["entity_1", "entity_2"] | null}}"""

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


def judge_comparative(claim: str, client: OpenAI, model: str, max_retries: int, provider: str | None = None, disable_reasoning: bool = False) -> dict:
    """Judge whether a claim makes an explicit comparison between approaches.

    Identifies comparative claims that contrast methods, models, or results (C.2).
    """
    return llm_json(client, model, _COMP_SYS, _COMP_USER.format(claim=claim[:600]), max_retries, provider, disable_reasoning)


def judge_valid_comparison(claim: str, context: str, client: OpenAI, model: str, max_retries: int, provider: str | None = None, disable_reasoning: bool = False) -> dict:
    """Judge validity of a comparative claim given source context.

    Determines if a comparison is fair and properly supported, handling only
    claims flagged as comparative (C.2).
    """
    return llm_json(
        client, model, _VALID_SYS,
        _VALID_USER.format(claim=claim[:600], context=context[:800] if context else "Not available"),
        max_retries, provider, disable_reasoning,
    )


# ── C.3 — Open questions ──────────────────────────────────────────────────────

_OPEN_SYS = "You are evaluating whether an atomic claim from a scientific survey explicitly formulates an open question, unresolved problem, or direction for future research."

_OPEN_USER = """\
The claim:
"{claim}"

A claim is considered an open question formulation if it:
- Explicitly states that something remains unknown, unresolved, or disputed
- Points to gaps in current understanding
- Identifies directions for future research
- Uses hedging markers like "remains unclear", "open question", "further research is needed", "not yet established"
- Discusses problems that have not been solved by the cited works

A claim is NOT an open question if it:
- Describes established results
- Makes claims about what is known
- Uses hedging for stylistic reasons without genuinely pointing to unresolved issues

Respond with a JSON object:
{{"is_open_question": true | false, "reasoning": "brief explanation", "question_type": "generalization" | "scalability" | "mechanism" | "theoretical" | "empirical" | "none"}}"""


def judge_open_question(claim: str, client: OpenAI, model: str, max_retries: int, provider: str | None = None, disable_reasoning: bool = False) -> dict:
    """Judge whether a claim formulates an open question or unresolved problem.

    Identifies claims that acknowledge gaps, unknowns, or future directions (C.3).
    """
    return llm_json(client, model, _OPEN_SYS, _OPEN_USER.format(claim=claim[:600]), max_retries, provider, disable_reasoning)


# ── C.4 — Modality diversity ──────────────────────────────────────────────────

_MOD_SYS = "You are classifying an atomic claim from a scientific survey by its epistemic modality — the level of confidence the claim expresses."

_MOD_USER = """\
The claim:
"{claim}"

Classify the claim into one of five levels:
1. Categorical assertion — stated as an established fact, no hedging. Uses "is", "does", "demonstrates".
2. Strong assertion — high confidence with mild qualifiers. Uses "generally", "typically", "has been shown to".
3. Moderate assertion — clear hedging indicating general but not universal applicability. Uses "often", "tends to", "can".
4. Weak assertion — strong hedging indicating tentative nature. Uses "may", "might", "could", "suggests", "appears to".
5. Explicit uncertainty — explicitly acknowledges the matter is unresolved. Uses "remains unclear", "is debated", "open question".

Focus on the linguistic markers of confidence, not on whether the claim is actually true.

Respond with a JSON object:
{{"modality_level": 1 | 2 | 3 | 4 | 5, "reasoning": "brief explanation citing specific hedging markers", "hedging_markers": ["marker_1"] | []}}"""


def judge_modality(claim: str, client: OpenAI, model: str, max_retries: int, provider: str | None = None, disable_reasoning: bool = False) -> dict:
    """Judge epistemic modality level (confidence) of a claim.

    Classifies claims on a 5-level scale from categorical to explicitly uncertain (C.4).
    """
    return llm_json(client, model, _MOD_SYS, _MOD_USER.format(claim=claim[:600]), max_retries, provider, disable_reasoning)


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
) -> dict:
    """Run all 4 LLM judges for a single claim. Returns enriched claim dict.

    Parallelizes criticality, comparativeness, open question, and modality judges
    via ThreadPoolExecutor (4 workers, one per judge). Validity check runs
    sequentially for claims flagged as comparative.
    """
    claim = claim_record["claim"]

    # ThreadPoolExecutor with max_workers=4 parallelizes the four judges.
    # Rate limiting is handled by the OpenAI client's internal queue.
    with ThreadPoolExecutor(max_workers=4) as ex:
        f_crit  = ex.submit(judge_criticality,  claim, client, model, max_retries, provider, disable_reasoning)
        f_comp  = ex.submit(judge_comparative,   claim, client, model, max_retries, provider, disable_reasoning)
        f_open  = ex.submit(judge_open_question, claim, client, model, max_retries, provider, disable_reasoning)
        f_mod   = ex.submit(judge_modality,      claim, client, model, max_retries, provider, disable_reasoning)

        crit_res = f_crit.result()
        comp_res = f_comp.result()
        open_res = f_open.result()
        mod_res  = f_mod.result()

    result = {
        **claim_record,
        # C.1
        "is_critical":    crit_res.get("is_critical", False),
        "critical_type":  crit_res.get("critical_type", "none"),
        # C.2
        "is_comparative": comp_res.get("is_comparative", False),
        "compared_entities": comp_res.get("compared_entities"),
        # C.3
        "is_open_question": open_res.get("is_open_question", False),
        "question_type":    open_res.get("question_type", "none"),
        # C.4
        "modality_level": mod_res.get("modality_level", 1),
        "hedging_markers": mod_res.get("hedging_markers", []),
    }

    # C.2 validity check — only for comparative claims
    if result["is_comparative"]:
        valid_res = judge_valid_comparison(claim, source_context, client, model, max_retries, provider, disable_reasoning)
        result["is_valid_comparison"] = valid_res.get("is_valid", False)
        result["invalidity_type"]     = valid_res.get("invalidity_type", "none")
    else:
        result["is_valid_comparison"] = None
        result["invalidity_type"]     = None

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

    if cfg.get("resume") and out_file.exists():
        try:
            with open(out_file) as f:
                existing = json.load(f)
            print(f"  [SKIP] {survey_id} — already scored")
            return existing
        except Exception:
            print(f"  [WARN] {survey_id} — corrupt cache, re-processing")

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

    model       = cfg["judge_model"]
    max_retries = cfg.get("max_retries", 3)
    provider           = cfg.get("judge_provider")           # optional — None means let OpenRouter choose
    disable_reasoning  = cfg.get("judge_disable_reasoning", False)
    # Shared source context (first reference abstract) for validity checks
    source_ctx  = ""  # enriched per model if corpus available

    judged_claims: list[dict] = []
    for claim_record in tqdm(claims, desc="  judging", unit="claim", leave=True):
        try:
            judged = judge_claim(claim_record, client, model, max_retries, source_ctx, provider, disable_reasoning)
            judged_claims.append(judged)
        except Exception as e:
            judged_claims.append({**claim_record, "_judge_error": str(e)})

    # ── Compute metrics ────────────────────────────────────────────────────
    n = len(judged_claims)

    m_crit       = round(sum(1 for c in judged_claims if c.get("is_critical")) / n, 4) if n else 0.0
    m_comp_total = round(sum(1 for c in judged_claims if c.get("is_comparative")) / n, 4) if n else 0.0

    comparative  = [c for c in judged_claims if c.get("is_comparative")]
    m_comp_valid = (
        round(sum(1 for c in comparative if c.get("is_valid_comparison")) / len(comparative), 4)
        if comparative else None
    )

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
        # C.2
        "m_comp_total": m_comp_total,
        "m_comp_valid": m_comp_valid,
        "n_comparative": len(comparative),
        "n_valid_comp":  sum(1 for c in comparative if c.get("is_valid_comparison")),
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
        f"m_comp_valid={m_comp_valid}  m_open={m_open}  m_mod={m_mod}  "
        f"({result['latency_sec']}s)"
    )
    return result


# ── Summary CSV ───────────────────────────────────────────────────────────────

def write_summary(results: list[dict], out_path: Path) -> None:
    csv_path = out_path / "summary.csv"
    fields = [
        "survey_id", "query", "n_claims",
        "m_crit", "n_critical",
        "m_comp_total", "m_comp_valid", "n_comparative", "n_valid_comp",
        "m_open", "n_open",
        "m_mod",
        "latency_sec",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\n[expert] summary → {csv_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Expert writing property metrics (C.1–C.4)")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model",   required=True)
    args = parser.parse_args()

    cfg    = load_config()
    client = make_client(cfg)

    claims_dir = resolve_claims_dir(args.dataset, args.model)

    gen_dir = ROOT / "results" / "generations" / f"{args.dataset}_{args.model}"
    if not gen_dir.exists():
        print(f"[ERROR] Generation dir not found: {gen_dir}", file=sys.stderr)
        sys.exit(1)

    run_id  = f"{cfg['judge_id']}_{cfg['judge_comment']}"
    out_dir = ROOT / "results" / "scores" / f"{args.dataset}_{args.model}_expert_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_files = sorted(gen_dir.glob("*.json"))
    gen_files = [f for f in gen_files if not re.search(r"_(raw|old)\.json$", f.name)]

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
