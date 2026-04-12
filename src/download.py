"""
Download the U4R/SurveyBench dataset from HuggingFace and save to datasets/.

The dataset stores files (markdown/PDF/etc.) — not standard parquet format,
so we use snapshot_download to get the entire repository,
then parse files manually.

Installation:
    pip install huggingface_hub

Usage:
    python src/download.py               # download and parse
    python src/download.py --inspect     # show structure of downloaded repo
    python src/download.py --raw         # only download without parsing

Output files:
    datasets/
    ├── raw/                     # original files from HuggingFace repo
    └── human_surveys/
        ├── Graph_Neural_Networks.json
        └── ...
"""

import os
import re
import json
import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DATASET_ID = "U4R/SurveyBench"
ROOT_DIR   = Path(__file__).parent.parent
RAW_DIR    = ROOT_DIR / "datasets" / "raw"
OUT_DIR    = ROOT_DIR / "datasets" / "human_surveys"


# Mapping folders/tags in repository → our topic IDs (SurveyBench topics.txt)
# We'll populate after --inspect; for now use heuristic based on filename
TOPIC_ALIASES = {
    "3d_gaussian":       "3D Gaussian Splatting",
    "gaussian_splatting":"3D Gaussian Splatting",
    "3d_object":         "3D Object Detection in Autonomous Driving",
    "autonomous":        "3D Object Detection in Autonomous Driving",
    "llm_eval":          "Evaluation of Large Language Models",
    "evaluation":        "Evaluation of Large Language Models",
    "multi_agent":       "LLM-based Multi-Agent",
    "multiagent":        "LLM-based Multi-Agent",
    "diffusion":         "Generative Diffusion Models",
    "gnn":               "Graph Neural Networks",
    "graph_neural":      "Graph Neural Networks",
    "hallucination":     "Hallucination in Large Language Models",
    "multimodal":        "Multimodal Large Language Models",
    "mllm":              "Multimodal Large Language Models",
    "rag":               "Retrieval-Augmented Generation for Large Language Models",
    "retrieval":         "Retrieval-Augmented Generation for Large Language Models",
    "vit":               "Vision Transformers",
    "vision_transformer":"Vision Transformers",
}


def get_token() -> str | None:
    return os.getenv("HF_TOKEN")


def snapshot(token: str | None = None) -> Path:
    """Download entire dataset repository to datasets/raw/."""
    from huggingface_hub import snapshot_download

    logger.info(f"Downloading {DATASET_ID} to {RAW_DIR}...")
    path = snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        local_dir=str(RAW_DIR),
        token=token,
        ignore_patterns=["*.git*", ".gitattributes"],
    )
    logger.info(f"Downloaded to {path}")
    return Path(path)


def inspect(raw_path: Path) -> None:
    """Print directory tree of downloaded repository."""
    logger.info(f"\nStructure of {raw_path}:")
    for p in sorted(raw_path.rglob("*")):
        if ".git" in p.parts:
            continue
        indent = "  " * (len(p.relative_to(raw_path).parts) - 1)
        size = f"  ({p.stat().st_size // 1024} KB)" if p.is_file() else ""
        logger.info(f"{indent}{'[+] ' if p.is_dir() else '[-] '}{p.name}{size}")


def guess_topic(path: Path) -> str | None:
    """Try to determine topic by path/filename."""
    key = path.stem.lower().replace("-", "_").replace(" ", "_")
    parent = path.parent.name.lower().replace("-", "_").replace(" ", "_")
    for token in [key, parent]:
        for alias, topic in TOPIC_ALIASES.items():
            if alias in token:
                return topic
    return None


def parse_and_save(raw_path: Path) -> None:
    """Parse downloaded files and save to datasets/human_surveys/."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Gather all text files
    text_extensions = {".md", ".txt", ".json"}
    files = [p for p in raw_path.rglob("*")
             if p.is_file()
             and p.suffix.lower() in text_extensions
             and ".git" not in p.parts
             and p.name not in ("README.md", "LICENSE", "NOTICE")]

    logger.info(f"\nFound {len(files)} text files")

    by_topic: dict[str, list[dict]] = {}
    unmatched: list[Path] = []

    for f in sorted(files):
        topic = guess_topic(f)
        text  = f.read_text(encoding="utf-8", errors="ignore")

        entry = {
            "title":    f.stem,
            "text":     text,
            "outline":  _extract_outline(text),
            "source":   str(f.relative_to(raw_path)),
        }

        if topic:
            by_topic.setdefault(topic, []).append(entry)
        else:
            unmatched.append(f)

    # Save by topic
    saved = []
    for topic, surveys in sorted(by_topic.items()):
        safe = re.sub(r"[^\w]", "_", topic).strip("_")
        out_file = OUT_DIR / f"{safe}.json"
        with open(out_file, "w", encoding="utf-8") as fp:
            json.dump({"topic": topic, "surveys": surveys}, fp,
                      ensure_ascii=False, indent=2)
        saved.append(safe)
        logger.info(f"  >> {out_file.name}  ({len(surveys)} survey(s))")

    if unmatched:
        logger.warning(f"\nCould not determine topic for {len(unmatched)} files:")
        for f in unmatched:
            logger.warning(f"     {f.relative_to(raw_path)}")
        logger.warning("  → Add mapping to TOPIC_ALIASES in src/download.py")

    meta = {
        "dataset_id": DATASET_ID,
        "topics": saved,
        "unmatched": [str(f.relative_to(raw_path)) for f in unmatched],
    }
    with open(ROOT_DIR / "datasets" / "metadata.json", "w") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)

    logger.info(f"\nDone. Topics saved: {len(saved)}")
    if unmatched:
        logger.info("   Run --inspect to see structure and refine TOPIC_ALIASES")


def _extract_outline(text: str) -> str:
    """Extract markdown headers as outline."""
    headers = [line.strip() for line in text.splitlines()
               if line.startswith("#")]
    return "\n".join(headers[:30])


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Download {DATASET_ID} from HuggingFace")
    parser.add_argument("--inspect", action="store_true",
                        help="Show structure of downloaded repo (without parsing)")
    parser.add_argument("--raw", action="store_true",
                        help="Only download without parsing by topics")
    parser.add_argument("--token", default=None,
                        help="HuggingFace token (overrides HF_TOKEN from .env)")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    token = args.token or get_token()

    raw_path = snapshot(token)

    if args.inspect or args.raw:
        inspect(raw_path)
        return

    parse_and_save(raw_path)


if __name__ == "__main__":
    main()
