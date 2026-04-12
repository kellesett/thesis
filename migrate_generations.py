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
import logging
import pathlib

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SRC = pathlib.Path("results/surge_perplexity_surge/generations")
DST = pathlib.Path("results/generations/SurGE_perplexity_dr")

DATASET_ID = "SurGE"
MODEL_ID   = "perplexity_dr"

def migrate(old: dict) -> dict | None:
    """Migrate old generation format to new unified format.

    Args:
        old: Generation in old format.

    Returns:
        Generation in new format, or None if required fields are missing.

    Raises:
        KeyError: If required fields (topic_id, topic) are missing.
    """
    # Validate required fields
    if "topic_id" not in old:
        logger.warning(f"Skipping record with missing topic_id: {old.get('system_id', 'unknown')}")
        return None
    if "topic" not in old:
        logger.warning(f"Skipping record {old.get('topic_id', 'unknown')} with missing topic")
        return None

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

def main() -> None:
    """Migrate old generation files to new unified format with validation and idempotency."""
    files = sorted(SRC.glob("*.json"))
    if not files:
        logger.info(f"No files found in {SRC}")
        return

    DST.mkdir(parents=True, exist_ok=True)

    n_migrated = 0
    n_skipped = 0

    for src_path in files:
        try:
            old = json.loads(src_path.read_text())
            new = migrate(old)

            if new is None:
                n_skipped += 1
                continue

            dst_path = DST / f"{new['id']}.json"

            # Idempotency check: warn if destination already exists
            if dst_path.exists():
                logger.warning(f"  {src_path.name}  →  {dst_path} (already exists, skipping)")
                n_skipped += 1
                continue

            dst_path.write_text(json.dumps(new, ensure_ascii=False, indent=2))
            logger.info(f"  {src_path.name}  →  {dst_path}")
            n_migrated += 1
        except Exception as e:
            logger.error(f"Error migrating {src_path.name}: {e}", exc_info=True)
            n_skipped += 1

    logger.info(f"\nDone: {n_migrated} file(s) migrated, {n_skipped} skipped, output to {DST}")

if __name__ == "__main__":
    main()