#!/usr/bin/env python3
"""
Migrate old generation files from results/surge_perplexity_surge/generations/
to the new unified format at results/generations/SurGE_perplexity_dr/.

Old format:
  {system_id, model, category, topic_id, topic, generated_text,
   references, arxiv_ids, input_tokens, output_tokens, cost_usd, latency_sec, error}

New format:
  {id, dataset_id, model_id, query, text, success, meta:{...}}
"""

import json
import pathlib

SRC = pathlib.Path("results/surge_perplexity_surge/generations")
DST = pathlib.Path("results/generations/SurGE_perplexity_dr")

DATASET_ID = "SurGE"
MODEL_ID   = "perplexity_dr"

def migrate(old: dict) -> dict:
    error = old.get("error")
    return {
        "id":         old["topic_id"],
        "dataset_id": DATASET_ID,
        "model_id":   MODEL_ID,
        "query":      old["topic"],
        "text":       old.get("generated_text") or "",
        "success":    error is None,
        "meta": {
            "model":        old.get("model"),
            "category":     old.get("category"),
            "generated_at": None,          # not stored in old format
            "latency_sec":  old.get("latency_sec"),
            "cost_usd":     old.get("cost_usd"),
            "input_tokens": old.get("input_tokens"),
            "output_tokens":old.get("output_tokens"),
            "references":   old.get("references"),
            "arxiv_ids":    old.get("arxiv_ids"),
            "error":        error,
        },
    }

def main():
    files = sorted(SRC.glob("*.json"))
    if not files:
        print(f"No files found in {SRC}")
        return

    DST.mkdir(parents=True, exist_ok=True)

    for src_path in files:
        old = json.loads(src_path.read_text())
        new = migrate(old)
        dst_path = DST / f"{new['id']}.json"
        dst_path.write_text(json.dumps(new, ensure_ascii=False, indent=2))
        print(f"  {src_path.name}  →  {dst_path}")

    print(f"\nDone: {len(files)} file(s) migrated to {DST}")

if __name__ == "__main__":
    main()