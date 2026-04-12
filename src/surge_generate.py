#!/usr/bin/env python
"""
Generate surveys on SurGE topics using an existing generation system.

Loads topics from SurGE's surveys.json, runs the specified system from
generate.py, and saves each result as both a .json file (full SurveyResult)
and a .md file (generated_text only, for SurGE's markdownParser).
The .md file is only written when generation succeeds.

Output filename pattern:  {survey_id}__{system_id}.json / .md

Usage:
    python src/surge_generate.py \\
        --system perplexity_dr \\
        --surveys-path datasets/SurGE/surveys.json \\
        --n-surveys 5 \\
        --out results/surge_perplexity_surge_20250330/generations \\
        --resume
"""
import os
import json
import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from generate import SYSTEMS, make_client, run_system

load_dotenv()


def load_surge_topics(surveys_path: Path, n: int) -> list[dict]:
    """
    Load the first n surveys from SurGE's surveys.json as generation topics.
    Each entry contains survey_id, topic_id (str), and topic (survey_title).

    Args:
        surveys_path: Path to surveys.json file
        n: Number of surveys to load

    Returns:
        List of dicts with topic_id, topic fields
    """
    with open(surveys_path, encoding="utf-8") as f:
        surveys = json.load(f)

    topics = []
    for s in surveys[:n]:
        # Validate survey data
        if not s.get("survey_id") or not s.get("survey_title"):
            logger.warning("Survey missing required fields, skipping")
            continue
        topics.append({
            "topic_id": str(s["survey_id"]),
            "topic":    s["survey_title"],
        })

    logger.info(f"Loaded {len(topics)} topics from {surveys_path}")
    return topics


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate surveys on SurGE topics")
    parser.add_argument("--system",        required=True,
                        help="System ID (key in SYSTEMS dict, e.g. perplexity_dr)")
    parser.add_argument("--surveys-path",  required=True,
                        help="Path to SurGE's surveys.json")
    parser.add_argument("--n-surveys",     type=int, default=5,
                        help="Number of surveys to generate")
    parser.add_argument("--out",           required=True,
                        help="Output directory for .json and .md files")
    parser.add_argument("--resume",        action="store_true",
                        help="Skip already-generated files")
    args = parser.parse_args()

    system_id = args.system
    if system_id not in SYSTEMS:
        raise SystemExit(f"Unknown system: '{system_id}'. Available: {list(SYSTEMS)}")
    config = SYSTEMS[system_id]

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set in .env")
    client = make_client(api_key)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    topics = load_surge_topics(Path(args.surveys_path), args.n_surveys)

    print(f"\nSystem   : {system_id}")
    print(f"Surveys  : {len(topics)}")
    print(f"Output   : {out_dir}\n")

    for t in topics:
        stem      = f"{t['topic_id']}__{system_id}"
        json_file = out_dir / f"{stem}.json"
        md_file   = out_dir / f"{stem}.md"

        if args.resume and json_file.exists():
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                # Check that generated_text is not just present but also non-empty
                if data.get("generated_text") and len(data.get("generated_text", "")) > 0:
                    logger.info(f"[SKIP] {stem}")
                    continue
            except Exception:
                pass

        logger.info(f"[GEN]  {stem} | {t['topic'][:60]}")
        result = run_system(client, system_id, config, t["topic_id"], t["topic"])

        # Save full result as JSON regardless of success (preserves error info)
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        if result.success and result.generated_text:
            # .md is consumed by SurGE's markdownParser — skip on failed generation
            md_file.write_text(result.generated_text, encoding="utf-8")

        status = "OK" if result.success else "FAIL"
        logger.info(
            f"[{status}] {result.word_count} words | "
            f"${result.cost_usd:.3f} | {result.latency_sec:.1f}s"
        )
        if result.error:
            logger.error(f"Error: {result.error}")

    logger.info(f"Generations saved to: {out_dir}")


if __name__ == "__main__":
    main()
