"""
Download SurveyForge database (U4R/SurveyForge_database) from HuggingFace.

Actual structure after download:
  datasets/surveyforge_db/
  ├── database/                         ← FAISS indices and TinyDB
  │   ├── arxiv_paper_db_with_cc.json
  │   ├── arxivid_to_index_abs.json
  │   ├── faiss_paper_title_embeddings_FROM_2012_0101_TO_240926.bin
  │   ├── faiss_paper_title_abs_embeddings_FROM_2012_0101_TO_240926.bin
  │   ├── surveys_arxiv_paper_db.json
  │   ├── surveys_arxivid_to_index_abs.json
  │   ├── faiss_survey_title_embeddings_FROM_1501_TO_2409_gte.bin
  │   └── faiss_survey_title_abs_embeddings_FROM_1501_TO_2409_gte.bin
  ├── Final_outline.zip                 ← outline database for heuristics
  └── Final_outline_First.zip

Total ~6-10 GB. Need HF_TOKEN if dataset requires authentication.

Usage:
    python src/utils/download_surveyforge_db.py
    python src/utils/download_surveyforge_db.py --out datasets/surveyforge_db
    python src/utils/download_surveyforge_db.py --check   # only check if all files present
"""

import os
import sys
import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass

DATASET_ID  = "U4R/SurveyForge_database"
ROOT_DIR    = Path(__file__).parent.parent.parent
DEFAULT_OUT = ROOT_DIR / "datasets" / "surveyforge_db"

# Файлы в подпапке database/ — нужны для database.py / database_survey из SurveyForge
DB_SUBDIR = "database"
REQUIRED_DB_FILES = [
    # БД статей (arXiv CS, 2012–2024)
    "arxiv_paper_db_with_cc.json",
    "arxivid_to_index_abs.json",
    "faiss_paper_title_embeddings_FROM_2012_0101_TO_240926.bin",
    "faiss_paper_title_abs_embeddings_FROM_2012_0101_TO_240926.bin",
    # БД survey-статей (2015–2024)
    "surveys_arxiv_paper_db.json",
    "surveys_arxivid_to_index_abs.json",
    "faiss_survey_title_embeddings_FROM_1501_TO_2409_gte.bin",
    "faiss_survey_title_abs_embeddings_FROM_1501_TO_2409_gte.bin",
]

# Файлы в корне репо — outline-база для heuristics
REQUIRED_ROOT_FILES = [
    "Final_outline.zip",
    "Final_outline_First.zip",
]


def check(out_dir: Path) -> bool:
    """Check that all required files are present. Return True if ok."""
    db_dir = out_dir / DB_SUBDIR
    logger.info(f"\nChecking database in {out_dir}\n")

    all_ok = True
    total_size = 0

    logger.info(f"  [{DB_SUBDIR}/]")
    for fname in REQUIRED_DB_FILES:
        fpath = db_dir / fname
        if fpath.exists() and fpath.stat().st_size > 0:
            size_mb = fpath.stat().st_size / 1024 / 1024
            total_size += size_mb
            logger.info(f"    [OK] {fname}  ({size_mb:.0f} MB)")
        else:
            logger.error(f"    [!!] MISSING or EMPTY: {fname}")
            all_ok = False

    logger.info(f"\n  [root/]")
    for fname in REQUIRED_ROOT_FILES:
        fpath = out_dir / fname
        if fpath.exists() and fpath.stat().st_size > 0:
            size_mb = fpath.stat().st_size / 1024 / 1024
            total_size += size_mb
            logger.info(f"    [OK] {fname}  ({size_mb:.0f} MB)")
        else:
            logger.error(f"    [!!] MISSING or EMPTY: {fname}")
            all_ok = False

    logger.info(f"\n  Total: {total_size:.0f} MB")
    if all_ok:
        logger.info(f"All files present — database ready for use")
        logger.info(f"     db_path for SurveyForge: {db_dir}")
    else:
        logger.warning(f"Some files missing — run without --check to download")
    return all_ok


def download(out_dir: Path, token: str | None) -> None:
    """Download entire database via snapshot_download."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub not installed: pip install huggingface_hub")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {DATASET_ID}")
    logger.info(f"Destination: {out_dir}")
    logger.info(f"Expected size: ~6-10 GB. This may take several minutes.")

    try:
        snapshot_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            local_dir=str(out_dir),
            token=token,
            ignore_patterns=["*.git*", ".gitattributes", "README*"],
        )
    except Exception as e:
        logger.error(f"Download error: {e}")
        logger.error("If dataset is private, set HF_TOKEN in .env")
        sys.exit(1)

    check(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Download SurveyForge database ({DATASET_ID})"
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_OUT),
        help=f"Where to save (default: {DEFAULT_OUT})"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Only check if files are present, do not download"
    )
    parser.add_argument(
        "--token", default=None,
        help="HuggingFace token (overrides HF_TOKEN from .env)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    out_dir = Path(args.out)
    token   = args.token or os.getenv("HF_TOKEN")

    if args.check:
        ok = check(out_dir)
        sys.exit(0 if ok else 1)

    download(out_dir, token)
    logger.info(f"db_path to pass to SurveyForge: {out_dir / DB_SUBDIR}")


if __name__ == "__main__":
    main()
