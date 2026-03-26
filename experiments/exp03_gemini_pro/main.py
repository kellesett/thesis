#!/usr/bin/env python
"""
Experiment: exp01_openai_dr
System:     OpenAI Deep Research (o4-mini)
Pipeline:   generation → SWR/CWR evaluation → summary
"""
import sys
import subprocess
import yaml
from pathlib import Path

EXP_DIR = Path(__file__).parent
ROOT    = EXP_DIR.parent.parent   # thesis/
CFG     = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
PYTHON  = sys.executable
SRC     = ROOT / "src"
OUT     = ROOT / "results" / CFG["experiment"]


def step(label: str, cmd: list[str]) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    subprocess.run(cmd, check=True)


def main() -> None:
    system    = CFG["system"]
    topics    = CFG.get("topics", 5)
    eval_cfg  = CFG.get("evaluation", {})
    methods   = eval_cfg.get("methods",    ["swr", "cwr"])
    etypes    = eval_cfg.get("eval_types", ["outline", "content"])
    judges    = eval_cfg.get("judges",     str(ROOT / "configs" / "judges.json"))

    gen_out  = OUT / "generated"
    eval_out = OUT / "eval"

    print(f"\n{'═' * 60}")
    print(f"  Experiment : {CFG['experiment']}")
    print(f"  System     : {system}")
    print(f"  Topics     : {topics}")
    print(f"  Methods    : {methods}")
    print(f"{'═' * 60}")

    # ── Step 1: Generation ────────────────────────────────────
    step("Step 1 / Generation", [
        PYTHON, str(SRC / "generate.py"),
        "--systems", system,
        "--topics",  str(topics),
        "--out",     str(gen_out),
        "--resume",
    ])

    # ── Step 2: Evaluation ────────────────────────────────────
    eval_cmd = [
        PYTHON, str(SRC / "evaluate.py"),
        "--eval",    ",".join(etypes),
        "--judges",  str(judges),
        "--systems", system,
        "--out",     str(eval_out),
        "--resume",
    ]
    if "swr" in methods: eval_cmd.append("--swr")
    if "cwr" in methods: eval_cmd.append("--cwr")

    step("Step 2 / Evaluation", eval_cmd)

    # ── Done ─────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  ✅  {CFG['experiment']} complete")
    print(f"  generated → {gen_out}")
    print(f"  eval      → {eval_out}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
