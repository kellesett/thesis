#!/usr/bin/env python
"""
Experiment: exp02_perplexity_dr
System:     Perplexity Sonar Deep Research

Группы метрик (флаги):
  --bench-metrics   генерация по SurveyBench + оценка SWR/CWR
  --surge-metrics   генерация по SurGE      + оценка SH-Recall/Structure_Quality/Logic

Без флагов запускаются обе группы.
"""
import sys
import argparse
import subprocess
import yaml
from pathlib import Path

EXP_DIR = Path(__file__).parent
ROOT    = EXP_DIR.parent.parent   # thesis/
CFG     = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
PYTHON  = sys.executable
SRC     = ROOT / "src"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bench-metrics", action="store_true",
                   help="Запустить SurveyBench: генерация + SWR/CWR оценка")
    p.add_argument("--surge-metrics", action="store_true",
                   help="Запустить SurGE: генерация + SH-Recall/Structure_Quality/Logic")
    args = p.parse_args()
    # Без флагов — запускаем всё
    if not args.bench_metrics and not args.surge_metrics:
        args.bench_metrics = True
        args.surge_metrics = True
    return args


def step(label: str, cmd: list[str]) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {label}")
    print(f"{'-' * 60}")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    system    = CFG["system"]
    topics    = CFG.get("topics", 5)
    eval_cfg  = CFG.get("evaluation", {})
    surge_cfg = CFG.get("surge", {})

    print(f"\n{'=' * 60}")
    print(f"  Experiment    : {CFG['experiment']}")
    print(f"  System        : {system}")
    print(f"  bench-metrics : {args.bench_metrics}")
    print(f"  surge-metrics : {args.surge_metrics}")
    print(f"{'=' * 60}")

    # ── SurveyBench (bench-metrics) ───────────────────────────────────────────
    if args.bench_metrics:
        sb_out      = ROOT / "results" / CFG["experiment"]
        sb_gen_out  = sb_out / "generated"
        sb_eval_out = sb_out / "eval"

        methods = eval_cfg.get("methods",    ["swr", "cwr"])
        etypes  = eval_cfg.get("eval_types", ["outline", "content"])
        judges  = ROOT / eval_cfg.get("judges", "configs/judges.json")
        k_refs  = eval_cfg.get("k_refs", 1)

        step("Step 1 / SurveyBench Generation", [
            PYTHON, str(SRC / "generate.py"),
            "--systems", system,
            "--topics",  str(topics),
            "--out",     str(sb_gen_out),
            "--resume",
        ])

        eval_cmd = [
            PYTHON, str(SRC / "evaluate.py"),
            "--eval",        ",".join(etypes),
            "--judges",      str(judges),
            "--systems",     system,
            "--results-dir", str(sb_gen_out),
            "--out",         str(sb_eval_out),
            "--resume",
            "--k-refs",      str(k_refs),
        ]
        if "swr" in methods: eval_cmd.append("--swr")
        if "cwr" in methods: eval_cmd.append("--cwr")
        step("Step 2 / SurveyBench Evaluation (SWR/CWR)", eval_cmd)

    # ── SurGE (surge-metrics) ─────────────────────────────────────────────────
    if args.surge_metrics:
        dataset     = surge_cfg.get("dataset",     "surge")
        model_name  = surge_cfg.get("model_name",  system)
        eval_method = surge_cfg.get("eval_method", "surge")
        run_dir     = ROOT / "results" / f"{dataset}_{model_name}_{eval_method}"
        surge_gen    = run_dir / "generations"
        surge_scores = run_dir / "scores"

        surge_n      = surge_cfg.get("n_surveys", 5)
        surge_judges = ROOT / surge_cfg.get("judges",       "configs/judges.json")
        surveys_path = ROOT / surge_cfg.get("surveys_path", "datasets/SurGE/surveys.json")
        embed_model  = surge_cfg.get("embedding_model",     "BAAI/bge-large-en-v1.5")
        surge_eval   = ",".join(surge_cfg.get("eval_list",  ["sh_recall", "structure_quality", "logic"]))

        print(f"  SurGE run     : {run_dir.name}")

        step("Step 3 / SurGE Generation", [
            PYTHON, str(SRC / "surge_generate.py"),
            "--system",       system,
            "--surveys-path", str(surveys_path),
            "--n-surveys",    str(surge_n),
            "--out",          str(surge_gen),
            "--resume",
        ])

        step("Step 4 / SurGE Evaluation", [
            PYTHON, str(SRC / "surge_evaluate.py"),
            "--surveys-path",    str(surveys_path),
            "--gen-dir",         str(surge_gen),
            "--scores-dir",      str(surge_scores),
            "--judges",          str(surge_judges),
            "--embedding-model", embed_model,
            "--eval",            surge_eval,
            "--resume",
        ])

    # ── Done ──────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  [OK]  {CFG['experiment']} complete")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
