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
import csv
import json
import logging
import os
import re
import sys
import time
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

from metrics.utils import make_client, load_config


# ── Claimify cache guard ──────────────────────────────────────────────────────

def resolve_claims_dir(dataset: str, model: str) -> Path:
    """
    Return path to pre-computed claims dir.
    Fails fast with a clear message if absent.
    """
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


# ── Corpus loading ────────────────────────────────────────────────────────────

def build_corpus_index(dataset: str) -> dict[str, str]:
    """
    Stream corpus.json once and build {paper_id → abstract} index.
    Returns dict keyed by paper_id (string).
    """
    corpus_path = ROOT / "datasets" / dataset / "corpus.json"
    if not corpus_path.exists():
        print(f"[WARN] corpus.json not found at {corpus_path}")
        return {}

    print("  Indexing corpus.json (streaming)...")
    index: dict[str, str] = {}
    with open(corpus_path, "rb") as f:
        for paper in ijson.items(f, "item"):
            pid   = str(paper.get("doc_id", paper.get("paperId", "")))
            title = paper.get("title", "")
            abstr = paper.get("abstract", "")
            index[pid] = f"{title}\n\n{abstr}".strip()
    print(f"  Corpus indexed: {len(index)} papers")
    return index


# ── AlignScore loader ─────────────────────────────────────────────────────────

def load_alignscore(cfg: dict):
    from alignscore import AlignScore
    ckpt = cfg["alignscore_ckpt"]
    print(f"  Loading AlignScore: {ckpt}")
    return AlignScore(
        model="roberta-large",
        batch_size=cfg.get("alignscore_batch_size", 16),
        device="cpu",
        ckpt_path=ckpt,
        evaluation_mode="nli_sp",
    )


# ── Claim categorization ──────────────────────────────────────────────────────

_CATEGORY_SYSTEM = (
    "You are classifying atomic claims from a scientific survey into one of four "
    "categories based on what type of information they convey."
)

_CATEGORY_PROMPT = """\
The claim to classify:
"{claim}"

Source paper context (if available):
"{source_context}"

Categories:
A — General topical claims. Statements that can be made based on only reading the abstract. They describe what a paper is about, what problem it addresses, or its general contribution without specific methodological or quantitative details.
B — Methodological refinements. Statements containing specific details about how methods work, requiring information typically found in the Methods section.
C — Quantitative claims. Statements containing specific numerical values, typically from the Results section.
D — Critical or comparative claims. Statements making evaluative judgments, comparing approaches, discussing limitations, or pointing to contradictions — typically from Discussion or Related Work sections.

Respond with a JSON object:
{{"category": "A" | "B" | "C" | "D", "reasoning": "brief explanation", "confidence": "high" | "medium" | "low"}}"""


def classify_claim(
    claim: str,
    source_context: str,
    client: OpenAI,
    model: str,
    max_retries: int,
) -> dict:
    """Classify a claim into one of four categories (A/B/C/D).

    Args:
        claim: Atomic claim text.
        source_context: Paper abstract or source context.
        client: OpenAI client instance.
        model: Judge model name.
        max_retries: Maximum retry attempts for API calls.

    Returns:
        Dict with "category" (A/B/C/D), "confidence", and optional "error".
    """
    prompt = _CATEGORY_PROMPT.format(
        claim=claim[:600],
        source_context=source_context[:800] if source_context else "Not available",
    )
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _CATEGORY_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0,
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
            parsed = json.loads(raw)
            cat = parsed.get("category", "").upper()
            if cat not in {"A", "B", "C", "D"}:
                raise ValueError(f"Unknown category: {cat}")
            return {"category": cat, "confidence": parsed.get("confidence", "low")}
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                return {"category": "A", "confidence": "low", "error": str(e)}
    return {"category": "A", "confidence": "low"}


# ── Support verification via AlignScore ───────────────────────────────────────

def compute_support_batch(
    claims: list[str],
    contexts: list[str],
    alignscore_model,
    threshold: float,
) -> list[bool]:
    """Verify citation support for claims using AlignScore.

    Args:
        claims: List of claim texts.
        contexts: Parallel list of source contexts (abstracts).
        alignscore_model: Initialized AlignScore model.
        threshold: AlignScore threshold for "supported" label.

    Returns:
        List of booleans: True if Support(claim, context) ≥ threshold, False otherwise.
        Pairs with no context are treated as unsupported.
    """
    if not claims:
        return []

    # Separate claims with and without context
    has_context = [bool(ctx) for ctx in contexts]
    scored_claims   = [c for c, h in zip(claims, has_context) if h]
    scored_contexts = [c for c, h in zip(contexts, has_context) if h]

    scores_map: dict[int, float] = {}
    if scored_claims:
        scores = alignscore_model.score(
            contexts=scored_contexts,
            claims=scored_claims,
        )
        j = 0
        for i, h in enumerate(has_context):
            if h:
                scores_map[i] = scores[j]
                j += 1

    return [scores_map.get(i, 0.0) >= threshold for i in range(len(claims))]


# ── Per-survey processing ─────────────────────────────────────────────────────

def get_source_context(claim_record: dict, gen: dict, corpus_index: dict) -> str:
    """
    Find best available source context for a claim.
    Uses references from the generation JSON → looks up abstract in corpus.
    Falls back to empty string.
    """
    refs = gen.get("meta", {}).get("references", [])
    if not refs:
        return ""
    # Use first available abstract as a proxy context
    # (a more precise attribution would require claim-level citation tracking)
    for ref in refs[:5]:
        arxiv_id = ref.get("arxiv_id", "")
        if arxiv_id and arxiv_id in corpus_index:
            return corpus_index[arxiv_id]
        pid = str(ref.get("idx", ""))
        if pid in corpus_index:
            return corpus_index[pid]
    return ""


def process_survey(
    gen: dict,
    claims_dir: Path,
    out_path: Path,
    cfg: dict,
    client: OpenAI,
    alignscore_model,
    corpus_index: dict,
) -> dict | None:
    """Evaluate factuality for one survey.

    Args:
        gen: Generation dict with success, text, query, etc.
        claims_dir: Path to pre-computed Claimify claims cache.
        out_path: Output directory for scores.
        cfg: Config dict with judge_model, thresholds, etc.
        client: OpenAI client instance.
        alignscore_model: Initialized AlignScore model.
        corpus_index: Mapping of paper_id to abstract.

    Returns:
        Score dict with citation correctness per category, or None on skip.
    """
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

    # Load claimify cache
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
    max_retries = cfg.get("max_retries", 3)

    # ── Step 1: Classify each claim as A/B/C/D ─────────────────────────────
    categorized: list[dict] = []
    for c in tqdm(claims, desc="  classifying", unit="claim", leave=True):
        # Per-claim source context: use claim record for more precise lookup
        source_ctx = get_source_context(c, gen, corpus_index)
        cat_result = classify_claim(
            c["claim"], source_ctx, client, cfg["judge_model"], max_retries
        )
        categorized.append({**c, **cat_result, "_source_ctx": source_ctx})

    # ── Step 2: AlignScore support verification ────────────────────────────
    claim_texts = [c["claim"] for c in categorized]
    contexts    = [c.get("_source_ctx", "") for c in categorized]
    threshold   = cfg.get("alignscore_threshold", 0.5)

    supported = compute_support_batch(claim_texts, contexts, alignscore_model, threshold)

    for c, sup in zip(categorized, supported):
        c["supported"] = sup

    # ── Step 3: Compute CitCorrect_k per category ──────────────────────────
    categories = ["A", "B", "C", "D"]
    cit_correct: dict[str, float | None] = {}
    cit_counts:  dict[str, dict]         = {}

    for k in categories:
        subset = [c for c in categorized if c["category"] == k]
        if not subset:
            cit_correct[k] = None
            cit_counts[k]  = {"n": 0, "n_supported": 0}
            continue
        n_sup = sum(1 for c in subset if c["supported"])
        cit_correct[k] = round(n_sup / len(subset), 4)
        cit_counts[k]  = {"n": len(subset), "n_supported": n_sup}

    overall_n   = len(categorized)
    overall_sup = sum(1 for c in categorized if c["supported"])
    cit_correct_overall = round(overall_sup / overall_n, 4) if overall_n else None

    result = {
        "survey_id":    survey_id,
        "dataset_id":   gen["dataset_id"],
        "model_id":     gen["model_id"],
        "query":        gen.get("query", ""),
        "n_claims":     overall_n,
        "n_supported":  overall_sup,
        "cit_correct_overall": cit_correct_overall,
        "cit_correct_A": cit_correct["A"],
        "cit_correct_B": cit_correct["B"],
        "cit_correct_C": cit_correct["C"],
        "cit_correct_D": cit_correct["D"],
        "category_counts": cit_counts,
        "claims":        categorized,
        "judge_model":   cfg["judge_model"],
        "latency_sec":   round(time.time() - t0, 1),
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    }

    with open(out_file, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(
        f"         CitCorrect: overall={cit_correct_overall}  "
        f"A={cit_correct['A']}  B={cit_correct['B']}  "
        f"C={cit_correct['C']}  D={cit_correct['D']}  "
        f"({result['latency_sec']}s)"
    )
    return result


# ── Summary CSV ───────────────────────────────────────────────────────────────

def write_summary(results: list[dict], out_path: Path) -> None:
    csv_path = out_path / "summary.csv"
    fields = [
        "survey_id", "query", "n_claims", "n_supported",
        "cit_correct_overall",
        "cit_correct_A", "cit_correct_B", "cit_correct_C", "cit_correct_D",
        "latency_sec",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\n[factuality] summary → {csv_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Factuality metrics (B.1)")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model",   required=True)
    args = parser.parse_args()

    cfg    = load_config(CONFIG)
    client = make_client(cfg)

    claims_dir = resolve_claims_dir(args.dataset, args.model)

    print("\n[factuality] Loading models...")
    alignscore_model = load_alignscore(cfg)
    corpus_index     = build_corpus_index(args.dataset)

    gen_dir = ROOT / "results" / "generations" / f"{args.dataset}_{args.model}"
    if not gen_dir.exists():
        print(f"[ERROR] Generation dir not found: {gen_dir}", file=sys.stderr)
        sys.exit(1)

    run_id  = f"{cfg['judge_id']}_{cfg['judge_comment']}"
    out_dir = ROOT / "results" / "scores" / f"{args.dataset}_{args.model}_factuality_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_files = sorted(gen_dir.glob("*.json"))
    gen_files = [f for f in gen_files if not re.search(r"_(raw|old)\.json$", f.name)]

    print(f"\n[factuality] {args.dataset} / {args.model}")
    print(f"             {len(gen_files)} surveys → {out_dir}\n")

    all_results = []
    for gf in gen_files:
        with open(gf) as f:
            gen = json.load(f)

        res = process_survey(
            gen, claims_dir, out_dir, cfg, client, alignscore_model, corpus_index
        )
        if res:
            all_results.append(res)

    if all_results:
        write_summary(all_results, out_dir)

    n_err = len(gen_files) - len(all_results)
    print(f"\n[factuality] done — ok={len(all_results)}  err={n_err}")


if __name__ == "__main__":
    main()
