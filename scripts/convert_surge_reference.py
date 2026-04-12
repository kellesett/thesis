#!/usr/bin/env python
"""
scripts/convert_surge_reference.py
Converts SurGE reference surveys into the unified generation format
(same schema as results/generations/<dataset>_<model>/<id>.json).

Output: results/generations/SurGE_reference/<id>.json

Usage:
    python scripts/convert_surge_reference.py
"""
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.log_setup import setup_logging
from src.datasets import load_dataset as load_dataset_cls


def load_registry(path: Path) -> dict[str, str]:
    """Load dataset registry from YAML file and return {dataset_id: path} mapping."""
    with open(path, encoding="utf-8") as f:
        reg = yaml.safe_load(f)
    return {e["id"]: e["path"] for e in reg["datasets"]}


def main() -> None:
    """Convert SurGE reference surveys to unified generation format."""
    setup_logging("convert_surge_ref")
    registry = load_registry(ROOT / "datasets" / "registry.yaml")
    dataset = load_dataset_cls("SurGE", registry["SurGE"])
    instances = list(dataset)

    out_dir = ROOT / "results" / "generations" / "SurGE_reference"
    out_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).isoformat()
    n_written = 0
    n_skipped = 0

    for instance in instances:
        out_file = out_dir / f"{instance.id}.json"

        if out_file.exists():
            n_skipped += 1
            continue

        try:
            text = instance.reference or ""
            meta = instance.meta or {}

            record = {
                "id":         instance.id,
                "dataset_id": "SurGE",
                "model_id":   "reference",
                "query":      instance.query,
                "text":       text,
                "success":    bool(text),
                "meta": {
                    "model":        "human",
                    "generated_at": generated_at,
                    "latency_sec":  None,
                    "cost_usd":     None,
                    "error":        None if text else "No reference text in dataset",
                    "references":   [],
                    "authors":      meta.get("authors", []),
                    "year":         meta.get("year"),
                    "abstract":     meta.get("abstract"),
                },
            }

            out_file.write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            n_written += 1
        except Exception as e:
            logger.error(f"Failed to convert instance {instance.id}: {e}", exc_info=True)

    logger.info(f"Written : {n_written}")
    logger.info(f"Skipped : {n_skipped} (already exist)")
    logger.info(f"Output  : {out_dir}")


if __name__ == "__main__":
    main()
