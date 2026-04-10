#!/usr/bin/env python
"""
models/autosurvey/main.py
Converter for AutoSurvey baseline generations from SurGE.

Reads pre-generated outputs from repos/SurGE/baselines/Autosurvey/output/
and converts them to the unified generation format, saving results to:
  results/generations/<dataset_id>_<model_id>/

AutoSurvey output format (per file):
  {"survey": str, "reference": {"1": "arxiv_id_v1", "2": "arxiv_id_v2", ...}}

Usage:
    python models/autosurvey/main.py --dataset SurGE
"""
import sys
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.models.base import BaseModel


class AutoSurvey(BaseModel):
    """
    Converter for pre-generated AutoSurvey outputs from the SurGE repository.

    Does not call any model API — only reads existing files and converts them
    to the unified generation format used by all other models.
    """

    def __init__(self) -> None:
        super().__init__(Path(__file__).parent)
        self.baselines_dir: Path = ROOT / self.cfg["baselines_dir"]
        if not self.baselines_dir.exists():
            raise SystemExit(
                f"AutoSurvey baselines not found: {self.baselines_dir}\n"
                f"Make sure the SurGE repository is cloned under repos/SurGE/"
            )

    def generate(self, instance) -> dict:
        """Not used — AutoSurvey converts existing files, it does not generate."""
        raise NotImplementedError("AutoSurvey is a converter; use convert() instead of run()")

    def convert(self, dataset_id: str) -> None:
        """
        Convert AutoSurvey baseline JSONs for dataset_id to unified format.

        File mapping:
          baselines_dir/<query_id>/<any>.json  →  output/<query_id>.json

        Each baseline folder contains exactly one JSON file.
        The folder name is the instance ID (= query_id in SurGE).
        """
        registry = self.load_registry(ROOT / "datasets" / "registry.yaml")
        if dataset_id not in registry:
            raise SystemExit(
                f"Dataset '{dataset_id}' not in registry. "
                f"Available: {list(registry)}"
            )

        # Build id → query mapping from the dataset
        from src.datasets import load_dataset as load_dataset_cls
        dataset = load_dataset_cls(dataset_id, registry[dataset_id])
        id_to_query: dict[str, str] = {inst.id: inst.query for inst in dataset}

        out_dir = ROOT / "results" / "generations" / f"{dataset_id}_{self.model_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Dataset  : {dataset_id}")
        print(f"Model    : {self.model_id}")
        print(f"Source   : {self.baselines_dir}")
        print(f"Output   : {out_dir}\n")

        n_ok = n_skip = n_miss = 0

        for folder in sorted(self.baselines_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1):
            if not folder.is_dir():
                continue

            instance_id = folder.name  # e.g. "0", "1", ..., "40"

            # Find the single JSON in this folder
            json_files = list(folder.glob("*.json"))
            if not json_files:
                print(f"  [WARN]  {instance_id}: no JSON found, skipping")
                n_miss += 1
                continue

            src_file = json_files[0]  # exactly one expected

            out_file = out_dir / f"{instance_id}.json"
            if out_file.exists():
                try:
                    existing = json.loads(out_file.read_text(encoding="utf-8"))
                    if existing.get("success") and existing.get("text"):
                        print(f"  [SKIP]  {instance_id}")
                        n_skip += 1
                        continue
                except Exception:
                    pass

            with open(src_file, encoding="utf-8") as f:
                raw = json.load(f)

            survey_text = raw.get("survey", "")
            ref_dict    = raw.get("reference", {})  # {"1": "2301.12345v1", ...}

            references = self.normalize_arxiv_references(ref_dict)

            query = id_to_query.get(instance_id, "")
            if not query:
                print(f"  [WARN]  {instance_id}: not found in dataset, query will be empty")

            record = {
                "id":         instance_id,
                "dataset_id": dataset_id,
                "model_id":   self.model_id,
                "query":      query,
                "text":       survey_text,
                "success":    bool(survey_text.strip()),
                "meta": {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "latency_sec":  None,
                    "error":        None,
                    "references":   references,
                    "source_file":  str(src_file.relative_to(ROOT)),
                },
            }

            out_file.write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            print(f"  [OK]    {instance_id} | refs={len(references)}")
            n_ok += 1

        print(f"\nDone: {n_ok} converted, {n_skip} skipped, {n_miss} missing")
        print(f"Generations saved to: {out_dir}")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert AutoSurvey SurGE baselines to unified generation format"
    )
    parser.add_argument("--dataset", required=True, help="Dataset id (e.g. SurGE)")
    args = parser.parse_args()

    AutoSurvey().convert(args.dataset)
