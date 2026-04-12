"""
src/models/base.py
Abstract base class for all survey generation models.

Each model subclass implements generate(instance) -> dict and is responsible
only for its own generation logic.  The common loop (registry, dataset, resume,
JSON serialisation, progress output) lives here in run().

Unified output contract for generate():
    {
        "text":    str,
        "success": bool,
        "meta": {
            "references":  list[dict],   # [{idx, title, url, arxiv_id, ...}]
            "latency_sec": float | None,
            "error":       str | None,
            # + any model-specific keys
        }
    }

run() wraps that into the canonical generation record:
    {
        "id":         str,
        "dataset_id": str,
        "model_id":   str,
        "query":      str,
        "text":       str,
        "success":    bool,
        "meta": {
            "generated_at": str (ISO-8601 UTC),
            ...                          # everything from generate()["meta"]
        }
    }
"""
from __future__ import annotations

import argparse
import json
import sys
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets import load_dataset as load_dataset_cls


class BaseModel(ABC):

    def __init__(self, model_dir: Path, fail_fast: bool = True) -> None:
        """
        Initialize base model.

        Args:
            model_dir: Path to model directory containing config.yaml
            fail_fast: If True, exit on first generation failure; if False, log and continue
        """
        self.model_dir = model_dir
        self.fail_fast = fail_fast
        config_path = model_dir / "config.yaml"
        with open(config_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.model_id: str = self.cfg["model_id"]
        # Set by run() before the generation loop so generate() can access it
        self.out_dir: Path | None = None

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def generate(self, instance) -> dict:
        """
        Generate a survey for one dataset instance.

        Args:
            instance: Dataset instance with id, query, and optionally reference fields

        Returns:
            Dict with keys:
            - text: Generated survey text (str)
            - success: Whether generation succeeded (bool)
            - meta: Dict with references, latency_sec, error, and model-specific keys
        """

    # ── Shared helpers ────────────────────────────────────────────────────────

    @staticmethod
    def load_registry(registry_path: Path) -> dict[str, str]:
        """Return {dataset_id: path} from datasets/registry.yaml."""
        with open(registry_path, encoding="utf-8") as f:
            reg = yaml.safe_load(f)
        return {entry["id"]: entry["path"] for entry in reg["datasets"]}

    @staticmethod
    def normalize_arxiv_references(ref_dict: dict) -> list[dict]:
        """
        Convert {idx: arxiv_id} dict (SurveyForge / AutoSurvey format) to
        the unified reference list used across all models.

            {"1": "2301.12345v1", ...}
            → [{"idx": 1, "title": None, "url": "...", "arxiv_id": "2301.12345v1",
                "canonical_title": None}, ...]
        """
        import re
        refs = []
        for idx, arxiv_id in sorted(ref_dict.items(), key=lambda x: int(x[0])):
            bare_id = re.sub(r"v\d+$", "", str(arxiv_id))
            refs.append({
                "idx":             int(idx),
                "title":           None,
                "url":             f"https://arxiv.org/abs/{arxiv_id}",
                "arxiv_id":        bare_id,
                "canonical_title": None,
            })
        return refs

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, dataset_id: str) -> None:
        """
        Load the dataset and iterate over instances, calling self.generate()
        for each one, then serialise the result to JSON.
        """
        n_surveys = self.cfg.get("n_surveys", None)
        resume    = self.cfg.get("resume", True)

        registry = self.load_registry(ROOT / "datasets" / "registry.yaml")
        if dataset_id not in registry:
            raise SystemExit(
                f"Dataset '{dataset_id}' not in registry. "
                f"Available: {list(registry)}"
            )

        dataset = load_dataset_cls(dataset_id, registry[dataset_id])
        instances = list(dataset)
        if n_surveys is not None:
            instances = instances[:n_surveys]

        self.out_dir = ROOT / "results" / "generations" / f"{dataset_id}_{self.model_id}"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        n_label = f"using first {n_surveys}" if n_surveys else "all"
        print(f"Dataset  : {dataset_id} ({len(dataset)} surveys, {n_label})")
        print(f"Model    : {self.model_id}")
        print(f"Output   : {self.out_dir}\n")

        for instance in instances:
            out_file = self.out_dir / f"{instance.id}.json"

            if resume and out_file.exists():
                try:
                    existing = json.loads(out_file.read_text(encoding="utf-8"))
                    if existing.get("success") and existing.get("text"):
                        logger.info(f"[SKIP] {instance.id}")
                        continue
                except Exception:
                    pass

            logger.info(f"[GEN]  {instance.id} | {instance.query[:70]}")

            result = self.generate(instance)

            record = {
                "id":         instance.id,
                "dataset_id": dataset_id,
                "model_id":   self.model_id,
                "query":      instance.query,
                "text":       result["text"],
                "success":    result["success"],
                "meta": {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    **result["meta"],
                },
            }

            out_file.write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            status = "OK" if result["success"] else "FAIL"
            meta   = result["meta"]
            parts  = [f"    [{status}]"]
            if meta.get("latency_sec") is not None:
                parts.append(f"latency={meta['latency_sec']}s")
            if result["success"] and meta.get("references"):
                parts.append(f"refs={len(meta['references'])}")
            if meta.get("error"):
                parts.append(f"error={str(meta['error'])[:80]}")
            logger.info(" ".join(parts))

            if not result["success"]:
                if self.fail_fast:
                    sys.exit(1)
                else:
                    logger.warning(f"Generation failed for {instance.id}, continuing")

        logger.info(f"Generations saved to: {self.out_dir}")
