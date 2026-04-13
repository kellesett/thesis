#!/usr/bin/env python3
"""
scripts/structural_specter_thr.py

Helper for calibrating the SPECTER similarity threshold in structural/config.yaml.

For each of up to N random surveys from a generation folder:
  1. Parse sections → flatten to (sentence, section) pairs
  2. SPECTER-encode all sentences (normalize_embeddings=True)
  3. Build cosine similarity matrix (dot product of unit vectors)
  4. For each similarity bucket in np.arange(0.5, 0.9, 0.04), sample 1 random
     cross-section pair whose similarity falls in [bucket, bucket+0.04)

All sampled pairs are merged across surveys, sorted by similarity ascending, and
saved to results/hyperopt/specter_thr.json for manual inspection.

Usage:
    python scripts/structural_specter_thr.py \\
        --gen-dir results/generations/SurGE_perplexity_dr \\
        --specter  models_cache/sentence-transformers--allenai-specter \\
        [--surveys 5]          # max surveys to sample (default: 5)
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from metrics.structural.main import split_sections


# ── Bucketing ─────────────────────────────────────────────────────────────────

BUCKET_START = 0.5
BUCKET_END   = 0.9
BUCKET_STEP  = 0.04
BUCKETS      = np.arange(BUCKET_START, BUCKET_END, BUCKET_STEP)  # 10 values


def _bucket_for(sim: float) -> float | None:
    """Return the left edge of the bucket this similarity falls into, or None."""
    for b in BUCKETS:
        if b <= sim < b + BUCKET_STEP:
            return round(float(b), 4)
    return None


# ── Per-survey sampling ───────────────────────────────────────────────────────

def sample_survey(survey: dict, specter_model) -> list[dict]:
    """
    For one survey: encode sentences, build similarity matrix, sample 1 pair per bucket.

    Returns list of dicts:
        {survey_id, similarity, bucket, section_i, s1, section_j, s2}
    """
    survey_id = survey.get("id", "unknown")
    text = survey.get("text", "").strip()
    if not text:
        return []

    sections = split_sections(text)
    flat: list[dict] = [
        {"text": sent, "section": sec["title"]}
        for sec in sections
        for sent in sec["sentences"]
    ]

    if len(flat) < 2:
        return []

    texts = [s["text"] for s in flat]
    embeddings = specter_model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    sims = np.dot(embeddings, embeddings.T)

    # Collect all cross-section pairs, grouped by bucket
    buckets: dict[float, list[dict]] = {round(float(b), 4): [] for b in BUCKETS}
    n = len(flat)
    for i in range(n):
        for j in range(i + 1, n):
            if flat[i]["section"] == flat[j]["section"]:
                continue
            sim = float(sims[i, j])
            bucket = _bucket_for(sim)
            if bucket is None:
                continue
            buckets[bucket].append({
                "survey_id":  survey_id,
                "similarity": round(sim, 4),
                "bucket":     bucket,
                "section_i":  flat[i]["section"],
                "s1":         flat[i]["text"],
                "section_j":  flat[j]["section"],
                "s2":         flat[j]["text"],
            })

    # Sample 1 pair per non-empty bucket
    sampled = []
    for bucket_pairs in buckets.values():
        if bucket_pairs:
            sampled.append(random.choice(bucket_pairs))

    return sampled


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Sample SPECTER pairs per similarity bucket")
    parser.add_argument("--gen-dir",  required=True,  help="Path to generations folder")
    parser.add_argument("--specter",  required=True,  help="Path to SPECTER model")
    parser.add_argument("--surveys",  type=int, default=5, help="Max surveys to sample (default: 5)")
    parser.add_argument("--seed",     type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    gen_dir = Path(args.gen_dir)
    if not gen_dir.exists():
        print(f"[ERROR] Generation dir not found: {gen_dir}", file=sys.stderr)
        sys.exit(1)

    gen_files = sorted(gen_dir.glob("*.json"))
    gen_files = [f for f in gen_files if not f.name.endswith(("_raw.json", "_old.json"))]
    if not gen_files:
        print(f"[ERROR] No generation files in {gen_dir}", file=sys.stderr)
        sys.exit(1)

    sampled_files = random.sample(gen_files, min(args.surveys, len(gen_files)))
    print(f"[specter_thr] Loading SPECTER: {args.specter}")
    from sentence_transformers import SentenceTransformer
    specter_model = SentenceTransformer(args.specter, device="cpu")

    print(f"[specter_thr] Sampling {len(sampled_files)} surveys "
          f"× up to {len(BUCKETS)} buckets = up to {len(sampled_files) * len(BUCKETS)} pairs\n")

    all_pairs: list[dict] = []
    for f in sampled_files:
        survey = json.loads(f.read_text())
        if not survey.get("success", False):
            print(f"  [SKIP] {f.stem} — not successful")
            continue
        pairs = sample_survey(survey, specter_model)
        print(f"  {f.stem}: {len(pairs)} pairs sampled")
        all_pairs.extend(pairs)

    if not all_pairs:
        print("[WARN] No pairs sampled — check generation files and text content")
        sys.exit(0)

    all_pairs.sort(key=lambda p: p["similarity"])

    out_dir = ROOT / "results" / "hyperopt"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "specter_thr.json"
    out_file.write_text(json.dumps(all_pairs, ensure_ascii=False, indent=2))

    print(f"\n[specter_thr] {len(all_pairs)} pairs → {out_file}")
    print(f"              similarity range: "
          f"{all_pairs[0]['similarity']:.3f} … {all_pairs[-1]['similarity']:.3f}")

    # Quick bucket summary
    from collections import Counter
    bucket_counts = Counter(p["bucket"] for p in all_pairs)
    print("\nBucket coverage:")
    for b in sorted(bucket_counts):
        bar = "█" * bucket_counts[b]
        print(f"  [{b:.2f}, {b + BUCKET_STEP:.2f})  {bar}  ({bucket_counts[b]})")


if __name__ == "__main__":
    main()
