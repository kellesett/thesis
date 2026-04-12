"""
Download embedding model Alibaba-NLP/gte-large-en-v1.5 (~1.7 GB)
to datasets/gte-large-en-v1.5/ for use with SurveyForge.

Usage:
    python src/utils/download_sf_model.py
    make sfmodel
"""
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
OUT  = ROOT / "datasets" / "gte-large-en-v1.5"
MODEL_ID = "Alibaba-NLP/gte-large-en-v1.5"

load_dotenv(ROOT / ".env")

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    if OUT.exists() and any(OUT.iterdir()):
        size_mb = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) // (1024 * 1024)
        logger.info(f"Model already exists: {OUT}  ({size_mb} MB)")
        return

    logger.info(f"Downloading {MODEL_ID} to {OUT}")
    logger.info("Size: ~1.7 GB")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("Install huggingface_hub: pip install huggingface_hub")
        sys.exit(1)

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(OUT),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.onnx", "onnx/*"],   # ONNX not needed for SentenceTransformer
    )

    # Check file integrity
    if not OUT.exists() or not any(OUT.iterdir()):
        logger.error(f"Download may have failed: {OUT} is empty")
        sys.exit(1)

    size_mb = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) // (1024 * 1024)
    logger.info(f"Downloaded: {OUT}  ({size_mb} MB)")
    logger.info("Path for config.yaml:")
    logger.info("     embedding_model: /app/datasets/gte-large-en-v1.5")

if __name__ == "__main__":
    main()
