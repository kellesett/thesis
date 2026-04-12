#!/usr/bin/env python
"""
Evaluate generated surveys using SurGE metrics:
  - SH-Recall         (Soft Heading Recall via BAAI/bge-large-en-v1.5 embeddings)
  - Structure_Quality (LLM-judge 0-5: outline hierarchy vs. reference)
  - Logic             (LLM-judge 0-5: coherence of sampled paragraphs)

Citation-based metrics (Coverage, Relevance-*) are intentionally excluded.

Monkey-patches SurGE's hardcoded gpt-4o calls to use the configured judge model.

Per-survey results are saved as {stem}.json with two top-level keys:
  "scores"    — final numeric scores
  "judge_log" — raw judge responses for debugging (metric, call index, raw text, tries)
A summary.csv aggregates scores across all surveys.

Usage:
    python src/surge_evaluate.py \\
        --surveys-path datasets/SurGE/surveys.json \\
        --gen-dir  results/surge_perplexity_surge/generations \\
        --scores-dir results/surge_perplexity_surge/scores \\
        --judges   experiments/exp02_perplexity_dr/judges.json \\
        --embedding-model BAAI/bge-large-en-v1.5 \\
        --eval sh_recall,structure_quality,logic \\
        --resume
"""
import os
import sys
import re
import json
import csv
import time
import argparse
import logging
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv
from src.log_setup import setup_logging

load_dotenv()

logger = logging.getLogger(__name__)

ROOT      = Path(__file__).parent.parent
SURGE_SRC = ROOT / "repos" / "SurGE" / "src"

# Import JudgeFailedError from exceptions module
sys.path.insert(0, str(ROOT / "src"))
from exceptions import JudgeFailedError


# ── Monkey-patch factory (moved from surge.py to avoid duplication) ──────────────

MAX_JUDGE_TRIES = 5
MAX_RETRY_DELAY = 32  # seconds, cap for exponential backoff


def _make_chat_openai_patched(judge_model: str, log: list, ctx: dict, flush_fn=None):
    """
    Return a drop-in replacement for SurGE's chat_openai functions.
    Uses judge_model instead of hardcoded gpt-4o.
    Appends one entry to log per attempt.
    """
    def chat_openai(prompt: str, client: OpenAI, try_number: int):
        if try_number >= MAX_JUDGE_TRIES:
            raise JudgeFailedError(
                f"metric={ctx.get('metric')} paragraph={ctx.get('paragraph_idx', 0)}: "
                f"failed to get valid score after {MAX_JUDGE_TRIES} tries"
            )

        entry = {
            "metric":         ctx.get("metric"),
            "paragraph_idx":  ctx.get("paragraph_idx", 0),
            "attempt_idx":    try_number,
            "prompt_preview": prompt[:100],
            "reasoning":      None,
            "raw":            None,
            "score":          None,
            "error":          None,
        }

        try:
            response = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user",   "content": prompt},
                ],
                stream=False,
                max_tokens=1024,
            )
            entry["reasoning"] = getattr(response.choices[0].message, "reasoning", None)
            raw = response.choices[0].message.content
            if raw is None:
                choice = response.choices[0]
                finish  = choice.finish_reason
                refusal = getattr(choice.message, "refusal", None)
                entry["error"] = (
                    f"content is None"
                    f", finish_reason={finish!r}"
                    + (f", refusal={refusal!r}" if refusal else "")
                )
                log.append(entry)
                if flush_fn:
                    flush_fn()
                return chat_openai(prompt, client, try_number + 1)

            content = raw.strip()
            entry["raw"] = content

            if not re.match(r"^[0-5]$", content):
                entry["error"] = f"invalid response: '{content}'"
                log.append(entry)
                if flush_fn:
                    flush_fn()
                return chat_openai(prompt, client, try_number + 1)

            entry["score"] = int(content)
            log.append(entry)
            if flush_fn:
                flush_fn()
            ctx["paragraph_idx"] = ctx.get("paragraph_idx", 0) + 1
            return entry["score"]

        except JudgeFailedError:
            raise
        except Exception as e:
            entry["error"] = str(e)
            log.append(entry)
            if flush_fn:
                flush_fn()
            delay = min(2 ** try_number, MAX_RETRY_DELAY)
            logger.warning(f"LLM judge error: {e}")
            logger.info(f"Retry in {delay}s...")
            time.sleep(delay)
            return chat_openai(prompt, client, try_number + 1)

    return chat_openai


# ── SurGE path setup ──────────────────────────────────────────────────────────

def _add_surge_to_path() -> None:
    """Insert SurGE's src/ into sys.path so its modules can be imported."""
    path_str = str(SURGE_SRC)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


# ── Judge client ──────────────────────────────────────────────────────────────

def load_judge_client(judges_path: Path) -> tuple[OpenAI, str]:
    """
    Parse judges.json and create an OpenAI-compatible client for the first entry.
    Supports both 'url' (literal base URL) and 'url_env' (env var name) fields.
    Returns (client, model_name).
    """
    judges = json.loads(judges_path.read_text(encoding="utf-8"))
    j = judges[0]

    if "url" in j:
        base_url = j["url"]
    elif "url_env" in j:
        base_url = os.getenv(j["url_env"])
        if not base_url:
            raise SystemExit(f"Env var '{j['url_env']}' is not set")
    else:
        base_url = None

    api_key = os.getenv(j["api_key_env"])
    if not api_key:
        raise SystemExit(f"Env var '{j['api_key_env']}' is not set")

    return OpenAI(base_url=base_url, api_key=api_key), j["model"]




# ── Data loading ──────────────────────────────────────────────────────────────

def load_surveys_map(surveys_path: Path) -> dict[str, dict]:
    """Load surveys.json and index entries by survey_id (cast to str)."""
    with open(surveys_path, encoding="utf-8") as f:
        surveys = json.load(f)
    surveys_map = {}
    for s in surveys:
        if not s.get("survey_id"):
            logger.warning("Survey missing survey_id, skipping")
            continue
        surveys_map[str(s["survey_id"])] = s
    return surveys_map


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_survey(
    md_path: Path,
    target_survey: dict,
    eval_list: list[str],
    flag_model,
    judge_client: OpenAI,
    judge_model: str,
    judge_log: list,
    flush_fn=None,
) -> dict:
    """
    Run the requested SurGE metrics on one generated .md file.

    Args:
        md_path: Path to generated markdown file
        target_survey: Reference survey dict
        eval_list: List of metrics to compute
        flag_model: Embedding model for SH-Recall
        judge_client: OpenAI client for judge
        judge_model: Model name for judge
        judge_log: List to append judge log entries to
        flush_fn: Optional callback to persist judge_log after each entry

    Returns:
        Dict with metric_name: score pairs

    Note:
        Monkey-patches SurGE modules with logging-aware chat_openai before evaluation.
        judge_log is passed in from outside so it remains accessible even if an
        exception occurs mid-evaluation.
    """
    import structureFuncs
    import informationFuncs
    from markdownParser import parse_markdown

    ctx = {"metric": None, "paragraph_idx": 0}

    patched = _make_chat_openai_patched(judge_model, judge_log, ctx, flush_fn=flush_fn)
    structureFuncs.chat_openai   = patched
    informationFuncs.chat_openai = patched

    psg_node = parse_markdown(str(md_path))
    scores   = {}

    if "sh_recall" in eval_list:
        scores["sh_recall"] = structureFuncs.eval_SHRecall(
            target_survey, psg_node, flag_model
        )

    if "structure_quality" in eval_list:
        ctx["metric"]         = "structure_quality"
        ctx["paragraph_idx"]  = 0
        scores["structure_quality"] = structureFuncs.eval_structure_quality_client(
            target_survey, psg_node, judge_client
        )

    if "logic" in eval_list:
        ctx["metric"]        = "logic"
        ctx["paragraph_idx"] = 0
        # Guard against empty surveys — original code raises ZeroDivisionError
        content_blocks = informationFuncs.get_content_list(psg_node)
        if content_blocks:
            scores["logic"] = informationFuncs.eval_logic_client(psg_node, judge_client)
        else:
            scores["logic"] = None

    return scores


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    setup_logging("surge_evaluate")
    parser = argparse.ArgumentParser(description="Evaluate generated surveys with SurGE metrics")
    parser.add_argument("--surveys-path",    required=True,
                        help="Path to SurGE's surveys.json")
    parser.add_argument("--gen-dir",         required=True,
                        help="Directory containing generated .md files")
    parser.add_argument("--scores-dir",      required=True,
                        help="Output directory for per-survey JSON and summary.csv")
    parser.add_argument("--judges",          required=True,
                        help="Path to judges.json")
    parser.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5",
                        help="FlagEmbedding model name or local path for SH-Recall")
    parser.add_argument("--eval",            default="sh_recall,structure_quality,logic",
                        help="Comma-separated metrics to compute")
    parser.add_argument("--resume",          action="store_true",
                        help="Skip files that already have a scores JSON")
    args = parser.parse_args()

    eval_list  = [e.strip() for e in args.eval.split(",")]
    gen_dir    = Path(args.gen_dir)
    scores_dir = Path(args.scores_dir)
    scores_dir.mkdir(parents=True, exist_ok=True)

    # ── Bootstrap SurGE imports ───────────────────────────────────────────────
    if not SURGE_SRC.exists():
        raise SystemExit(
            f"SurGE source not found at {SURGE_SRC}\n"
            "Clone the repo: git clone https://github.com/oneal2000/SurGE repos/SurGE"
        )
    _add_surge_to_path()

    # ── Judge client ──────────────────────────────────────────────────────────
    judge_client, judge_model = load_judge_client(Path(args.judges))
    print(f"Judge model: {judge_model}")

    # ── Embedding model for SH-Recall ─────────────────────────────────────────
    flag_model = None
    if "sh_recall" in eval_list:
        try:
            from FlagEmbedding import FlagModel
        except ImportError:
            raise SystemExit(
                "FlagEmbedding is required for SH-Recall.\n"
                "Install: pip install FlagEmbedding"
            )
        print(f"Loading embedding model: {args.embedding_model}")
        import torch
        flag_model = FlagModel(args.embedding_model, use_fp16=torch.cuda.is_available())

    # ── Reference surveys ─────────────────────────────────────────────────────
    surveys_map = load_surveys_map(Path(args.surveys_path))
    print(f"Loaded {len(surveys_map)} reference surveys")

    # ── Evaluate each .md file ────────────────────────────────────────────────
    md_files = sorted(gen_dir.glob("*.md"))
    if not md_files:
        raise SystemExit(f"No .md files found in {gen_dir}")

    print(f"\nEvaluating {len(md_files)} file(s), metrics: {eval_list}\n")
    all_results = []

    for md_path in md_files:
        # Expected filename: {topic_id}__{system_id}.md
        stem  = md_path.stem
        parts = stem.split("__", 1)
        if len(parts) != 2:
            print(f"  [SKIP] Unrecognized filename pattern: {md_path.name}")
            continue
        topic_id, system_id = parts

        score_file = scores_dir / f"{stem}.json"
        if args.resume and score_file.exists():
            print(f"  [SKIP] {stem}")
            try:
                all_results.append(json.loads(score_file.read_text(encoding="utf-8")))
            except Exception:
                pass
            continue

        if topic_id not in surveys_map:
            print(f"  [WARN] survey_id={topic_id} not in surveys.json — skipping")
            continue

        target_survey = surveys_map[topic_id]
        print(f"  [EVAL] {stem}")
        print(f"         {target_survey['survey_title'][:70]}")

        judge_log = []
        log_file  = scores_dir / f"{stem}.log.json"

        def flush_fn(path=log_file, log=judge_log):
            path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

        try:
            scores = evaluate_survey(
                md_path, target_survey, eval_list, flag_model, judge_client, judge_model,
                judge_log, flush_fn=flush_fn,
            )
        except JudgeFailedError as e:
            # Judge failed to evaluate paragraph — stop evaluation of this survey
            logger.error(f"Evaluation aborted: {e}")
            scores = {}
        except Exception as e:
            logger.exception(f"Error during evaluation: {e}")
            scores = {}

        result = {
            "topic_id":     topic_id,
            "system_id":    system_id,
            "survey_title": target_survey["survey_title"],
            "scores":       scores,
            "judge_log":    judge_log,
        }

        score_file.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        all_results.append(result)

        score_str = "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in scores.items()
        )
        logger.info(f"Scores: {score_str}")

    if not all_results:
        print("No results to summarize.")
        return

    # ── Summary CSV ───────────────────────────────────────────────────────────
    # Use eval_list as canonical columns — all_results[0]["scores"] may be empty
    # if the first survey's evaluation failed entirely
    csv_path = scores_dir / "summary.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["topic_id", "system_id", "survey_title"] + eval_list
        )
        writer.writeheader()
        for r in all_results:
            row = {
                "topic_id":     r["topic_id"],
                "system_id":    r["system_id"],
                "survey_title": r["survey_title"],
            }
            for key in eval_list:
                row[key] = r["scores"].get(key)
            writer.writerow(row)

    # ── Print aggregate statistics ────────────────────────────────────────────
    logger.info(f"\nResults: {len(all_results)} survey(s)")
    for key in eval_list:
        vals = [
            r["scores"][key] for r in all_results
            if r["scores"].get(key) is not None
        ]
        if vals:
            avg = sum(vals) / len(vals)
            logger.info(f"  {key:<20} avg={avg:.4f}  min={min(vals):.4f}  max={max(vals):.4f}")
    logger.info(f"Scores → {scores_dir}")
    logger.info(f"CSV    → {csv_path}")


if __name__ == "__main__":
    main()
