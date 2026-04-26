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
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets import load_dataset as load_dataset_cls
from src.log_setup import setup_logging


def _fmt_k(n: int) -> str:
    """Compact integer for tqdm postfix: `12345` → `12k`, `123` → `123`."""
    return f"{n // 1000}k" if n >= 1000 else str(n)


def _fmt_postfix(
    sid: str, status: str,
    cost: float, in_tok: int, out_tok: int, rsn_tok: int,
    n_ok: int, n_skip: int, n_fail: int,
) -> str:
    """Build the per-iteration tqdm postfix string.

    Shows `$cost in:Xk out:Yk rsn:Zk` only when the running totals are
    non-zero (so for generators that don't surface usage we don't pollute
    the bar with zeros). Always shows last instance id + outcome counters.
    """
    parts = [f"id={sid} [{status}]"]
    if cost > 0 or in_tok > 0 or out_tok > 0:
        parts.append(f"${cost:.4f}")
        parts.append(f"in:{_fmt_k(in_tok)}")
        parts.append(f"out:{_fmt_k(out_tok)}")
        if rsn_tok > 0:
            parts.append(f"rsn:{_fmt_k(rsn_tok)}")
    parts.append(f"ok={n_ok}/skip={n_skip}/fail={n_fail}")
    return "  ".join(parts)


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

    def run(self, dataset_id: str, limit: int | None = None) -> None:
        """Load the dataset and iterate instances, calling :meth:`generate`.

        Args:
            dataset_id: Registry key (e.g. ``"SurGE"``).
            limit: If set, process only instances with numeric ``id <= limit``
                — id-based, NOT positional. Same semantics as ``--limit`` in
                metrics. Non-numeric ids are dropped from the filtered set.
                ``None`` → process all instances.
        """
        # Centralised log file at results/logs/generate.log — append-mode,
        # one banner per run. Without this the `logger.info("[SKIP] …")`
        # / `[GEN] …` messages below are silently dropped (root logger
        # has no handlers by default), and tqdm-on-stderr doesn't help.
        setup_logging("generate")

        resume = self.cfg.get("resume", True)

        registry = self.load_registry(ROOT / "datasets" / "registry.yaml")
        if dataset_id not in registry:
            raise SystemExit(
                f"Dataset '{dataset_id}' not in registry. "
                f"Available: {list(registry)}"
            )

        dataset = load_dataset_cls(dataset_id, registry[dataset_id])
        # SurGE's surveys.json is NOT in numeric survey_id order (first 15 ids
        # are [0, 1, 41, 42, …, 53]). Sort by numeric id so iteration matches
        # the ordering expected by downstream consumers (factuality, viewer).
        # Non-numeric ids sort lexicographically after the numeric bucket.
        instances = sorted(
            dataset,
            key=lambda inst: (0, int(inst.id)) if str(inst.id).isdigit()
                             else (1, str(inst.id)),
        )
        if limit is not None:
            instances = [
                inst for inst in instances
                if str(inst.id).isdigit() and int(inst.id) <= limit
            ]

        self.out_dir = ROOT / "results" / "generations" / f"{dataset_id}_{self.model_id}"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        n_label = f"limit={limit} → {len(instances)}" if limit is not None else f"all ({len(instances)})"
        print(f"Dataset  : {dataset_id} ({len(dataset)} surveys, {n_label})")
        print(f"Model    : {self.model_id}")
        print(f"Output   : {self.out_dir}\n")

        # Cumulative usage stats across all instances. We display whatever
        # `generate()` surfaces in `meta`: if a generator writes
        # `meta.cost_usd` and/or `meta.tokens.{prompt,completion,reasoning}`
        # — we accumulate and show in the tqdm postfix. Generators that
        # don't expose this info contribute zero (the postfix shows $0.00,
        # which honestly reflects what we know).
        total_cost  = 0.0
        total_in    = 0
        total_out   = 0
        total_rsn   = 0
        n_ok = n_skip = n_fail = 0

        bar = tqdm(
            instances,
            desc=f"{self.model_id}", unit="survey",
            dynamic_ncols=True, leave=True,
        )
        # `logging_redirect_tqdm` маршрутизирует всё что попадает в root
        # logger (наши `[SKIP]`/`[GEN]`/`[OK]` строки) через `tqdm.write` —
        # бар приостанавливается, строка печатается выше, бар
        # перерисовывается ниже. Без этого `\n` от logger ломает `\r`-ные
        # redraw'ы tqdm и они визуально перемешиваются.
        with logging_redirect_tqdm():
            for instance in bar:
                out_file = self.out_dir / f"{instance.id}.json"

                if resume and out_file.exists():
                    try:
                        existing = json.loads(out_file.read_text(encoding="utf-8"))
                        if existing.get("success") and existing.get("text"):
                            logger.info(f"[SKIP] {instance.id}")
                            n_skip += 1
                            bar.set_postfix_str(_fmt_postfix(
                                instance.id, "skip",
                                total_cost, total_in, total_out, total_rsn,
                                n_ok, n_skip, n_fail,
                            ), refresh=True)
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

                # Aggregate usage. Defensive on shapes — surveygen_i and
                # surveyforge currently write `cost_usd: None` and no `tokens`
                # field, so we coerce both to 0.
                cost_usd = meta.get("cost_usd") or 0
                total_cost += float(cost_usd)
                tokens = meta.get("tokens") or {}
                total_in  += int(tokens.get("prompt")     or 0)
                total_out += int(tokens.get("completion") or 0)
                total_rsn += int(tokens.get("reasoning")  or 0)

                if result["success"]:
                    n_ok += 1
                else:
                    n_fail += 1

                bar.set_postfix_str(_fmt_postfix(
                    instance.id, status.lower(),
                    total_cost, total_in, total_out, total_rsn,
                    n_ok, n_skip, n_fail,
                ), refresh=True)

                if not result["success"]:
                    if self.fail_fast:
                        bar.close()
                        sys.exit(1)
                    else:
                        logger.warning(f"Generation failed for {instance.id}, continuing")

        bar.close()
        logger.info(f"Generations saved to: {self.out_dir}")
        if total_cost > 0 or total_in > 0:
            logger.info(
                "Run usage: ok=%d skip=%d fail=%d  cost=$%.4f  in=%s  out=%s  rsn=%s",
                n_ok, n_skip, n_fail, total_cost,
                _fmt_k(total_in), _fmt_k(total_out), _fmt_k(total_rsn),
            )
