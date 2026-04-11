#!/usr/bin/env python
"""
metrics/surge/main.py
Evaluation runner for the SurGE metric suite.

Loads unified generation JSONs from:
  results/generations/<dataset_id>_<model_id>/

Evaluates each with SurGEEvaluator and saves per-survey scores + judge_log to:
  results/scores/<dataset_id>_<model_id>_surge_<judge_slug>_<judge_comment>/

where judge_slug is the judge_model name sanitized for use in a path.

Usage (inside Docker):
    python metrics/surge/main.py --dataset SurGE --model perplexity_dr
"""
import csv
import json
import re
import sys
import time
import traceback
import argparse
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT   = Path(__file__).parent.parent.parent
CONFIG = Path(__file__).parent / "config.yaml"
sys.path.insert(0, str(ROOT))

from src.datasets import load_dataset as load_dataset_cls
from src.evaluators import load_evaluator
from src.evaluators.surge import JudgeFailedError


def _slug(text: str) -> str:
    """Sanitize a string for use in a directory name."""
    return re.sub(r"[^a-zA-Z0-9_.\-]", "_", text)


def load_registry(registry_path: Path) -> dict[str, str]:
    with open(registry_path, encoding="utf-8") as f:
        reg = yaml.safe_load(f)
    return {entry["id"]: entry["path"] for entry in reg["datasets"]}


class RunLogger:
    """Simple append-only text logger with timestamps.

    Writes to a .log file and optionally mirrors to stdout.
    Each line: [YYYY-MM-DD HH:MM:SS] message
    """

    def __init__(self, path: Path, echo: bool = False) -> None:
        self._path = path
        self._echo = echo
        self._f    = open(path, "a", encoding="utf-8", buffering=1)  # line-buffered

    def __call__(self, message: str) -> None:
        ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}\n"
        self._f.write(line)
        if self._echo:
            print(line, end="")

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *_) -> None:
        self.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generations with SurGE metrics")
    parser.add_argument("--dataset", required=True, help="Dataset id (e.g. SurGE)")
    parser.add_argument("--model",   required=True, help="Model id (e.g. perplexity_dr)")
    args = parser.parse_args()

    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    eval_list     = cfg["eval_list"]
    judge_comment = cfg.get("judge_comment", "")
    resume        = cfg.get("resume", True)

    # ── Build output path ─────────────────────────────────────────────────────
    judge_id    = _slug(cfg["judge_id"])
    scores_name = f"{args.dataset}_{args.model}_surge_{judge_id}"
    if judge_comment:
        scores_name += f"_{_slug(judge_comment)}"
    scores_dir = ROOT / "results" / "scores" / scores_name
    scores_dir.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────────
    registry = load_registry(ROOT / "datasets" / "registry.yaml")
    if args.dataset not in registry:
        raise SystemExit(f"Dataset '{args.dataset}' not in registry.")

    dataset = load_dataset_cls(args.dataset, registry[args.dataset])
    print(f"Dataset  : {args.dataset} ({len(dataset)} surveys)")

    # ── Load evaluator (initialised once for the whole run) ───────────────────
    evaluator = load_evaluator(cfg["evaluator_id"], cfg, dataset)
    print(f"Evaluator: surge | judge={cfg['judge_model']} ({cfg['judge_id']})")
    print(f"Metrics  : {eval_list}")
    print(f"Scores → {scores_dir}\n")

    # ── Load generations ──────────────────────────────────────────────────────
    gen_dir = ROOT / "results" / "generations" / f"{args.dataset}_{args.model}"
    if not gen_dir.exists():
        raise SystemExit(f"Generations not found at {gen_dir}")

    gen_files = sorted(
        f for f in gen_dir.glob("*.json")
        if re.fullmatch(r'\d+\.json', f.name)
    )
    if not gen_files:
        raise SystemExit(f"No generation files (N.json) found in {gen_dir}")

    print(f"Evaluating {len(gen_files)} file(s), metrics: {eval_list}\n")

    # ── Run logger ────────────────────────────────────────────────────────────
    log = RunLogger(scores_dir / "run.log")
    log(f"=== run started ===")
    log(f"dataset={args.dataset}  model={args.model}")
    log(f"judge={cfg['judge_model']}  eval_list={eval_list}")
    log(f"files to evaluate: {len(gen_files)}")

    all_results = []

    for gen_file in gen_files:
        generation = json.loads(gen_file.read_text(encoding="utf-8"))
        gid        = generation["id"]
        score_file = scores_dir / f"{gid}.json"
        log_file   = scores_dir / f"{gid}.log.json"

        if resume and score_file.exists():
            try:
                all_results.append(json.loads(score_file.read_text(encoding="utf-8")))
                print(f"  [SKIP] {gid}")
                log(f"skip survey {gid}: already scored (resume=true)")
                continue
            except Exception as e:
                print(f"  [WARN] {gid}: corrupt score file, re-evaluating — {e}")
                log(f"warn survey {gid}: corrupt score file, re-evaluating — {e}")

        if not generation.get("success"):
            print(f"  [SKIP] {gid} — generation failed, no text to evaluate")
            log(f"skip survey {gid}: generation failed")
            continue

        try:
            instance = dataset.get_by_id(gid)
        except KeyError as e:
            print(f"  [WARN] {e}")
            log(f"skip survey {gid}: not in dataset — {e}")
            continue

        print(f"  [EVAL] {gid} | {instance.query[:60]}")
        log(f"--- survey {gid}: {instance.query[:80]} ---")

        # flush_fn writes judge_log incrementally so we don't lose it on crash
        judge_log_ref = []

        def flush_fn(path=log_file, log_ref=judge_log_ref):
            if log_ref:
                path.write_text(
                    json.dumps(log_ref, ensure_ascii=False, indent=2), encoding="utf-8"
                )

        scores = {}
        t0 = time.time()
        try:
            out = evaluator.evaluate(generation, flush_fn=flush_fn, log_fn=log)
            scores           = out["scores"]
            judge_log_ref[:] = out["judge_log"]
        except JudgeFailedError as e_:
            print(f"  [FAIL] Evaluation aborted: {e_}")
            log(f"FAIL survey {gid}: judge exhausted retries — {e_}")
        except Exception as e_:
            print(f"  [ERR]  {e_}")
            log(f"ERROR survey {gid}:\n{traceback.format_exc()}")

        duration = round(time.time() - t0, 1)
        for k, v in scores.items():
            log(f"score {k} = {v}")
        log(f"survey {gid} done in {duration}s  judge_calls={len(judge_log_ref)}")

        result = {
            "id":           gid,
            "dataset_id":   args.dataset,
            "model_id":     args.model,
            "survey_title": instance.query,
            "scores":       scores,
            "judge_log":    judge_log_ref,
        }
        score_file.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        all_results.append(result)

        score_str = "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in scores.items()
        )
        print(f"         {score_str or '(no scores)'}\n")

    if not all_results:
        print("No results to summarize.")
        return

    # ── Summary CSV ───────────────────────────────────────────────────────────
    csv_path = scores_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "dataset_id", "model_id", "survey_title"] + eval_list
        )
        writer.writeheader()
        for r in all_results:
            row = {
                "id":           r["id"],
                "dataset_id":   r["dataset_id"],
                "model_id":     r["model_id"],
                "survey_title": r["survey_title"],
            }
            for key in eval_list:
                row[key] = r["scores"].get(key)
            writer.writerow(row)

    # ── Aggregate stats ───────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  Results: {len(all_results)} survey(s)")
    for key in eval_list:
        vals = [r["scores"][key] for r in all_results if r["scores"].get(key) is not None]
        if vals:
            avg = sum(vals) / len(vals)
            print(f"  {key:<20} avg={avg:.4f}  min={min(vals):.4f}  max={max(vals):.4f}")
    print(f"{'─' * 60}")
    print(f"  Scores → {scores_dir}")
    print(f"  CSV    → {csv_path}")

    log(f"=== run finished: {len(all_results)} evaluated ===")
    log.close()   # also reachable via context manager: `with RunLogger(...) as log:`


if __name__ == "__main__":
    main()
