#!/usr/bin/env python3
"""
scripts/download_metric_models.py
Downloads all models required for structural / factuality / expert metrics locally.
Run once from the repository root:

    python scripts/download_metric_models.py

Models are saved in models_cache/ and mounted in Docker via volume.
Total size: ~2.4 GB (see table below).

  fleonce--iter-scierc-scideberta-full    ~740 MB  (NER, Group A)
  microsoft--deberta-v2-xlarge-mnli       ~900 MB  (NLI, Groups A/B/C)
  yzha--AlignScore-large                  ~355 MB  (factuality support, Groups B/C)
  sentence-transformers--allenai-specter  ~400 MB  (embedding pre-filter, Group A)

Note: AlignScore expects a .ckpt checkpoint file, not HuggingFace safetensors.
After download, the file will be in:
  models_cache/yzha--AlignScore-large/AlignScore-large.ckpt
"""
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.log_setup import setup_logging

try:
    from huggingface_hub import snapshot_download, hf_hub_download
except ImportError:
    logger.error("Install dependencies: pip install huggingface_hub")
    sys.exit(1)

CACHE = ROOT / "models_cache"
CACHE.mkdir(exist_ok=True)

# (repo_id, local_dir_name, description, download_fn, extra_kwargs)
MODELS = [
    {
        "repo_id": "fleonce/iter-scierc-scideberta-full",
        "local":   "fleonce--iter-scierc-scideberta-full",
        "desc":    "NER (SciERC, EMNLP 2024) — ~740 MB",
        "fn":      "snapshot",
    },
    {
        "repo_id": "microsoft/deberta-v2-xlarge-mnli",
        "local":   "microsoft--deberta-v2-xlarge-mnli",
        "desc":    "NLI (DeBERTa-v2-xlarge) — ~900 MB",
        "fn":      "snapshot",
    },
    {
        "repo_id": "sentence-transformers/allenai-specter",
        "local":   "sentence-transformers--allenai-specter",
        "desc":    "SPECTER embeddings (embedding pre-filter) — ~400 MB",
        "fn":      "snapshot",
    },
    {
        "repo_id":  "yzha/AlignScore",
        "filename": "AlignScore-large.ckpt",
        "local":    "yzha--AlignScore-large",
        "desc":     "AlignScore-large checkpoint — ~355 MB",
        "fn":       "single_file",
    },
]


def download_model(m: dict) -> None:
    """Download a model from HuggingFace Hub to models_cache/.

    Args:
        m: Model configuration dict with repo_id, local, desc, fn (snapshot or single_file).
    """
    local_dir = CACHE / m["local"]
    local_dir.mkdir(exist_ok=True)

    if m["fn"] == "snapshot":
        # Check if model is already downloaded
        existing = list(local_dir.glob("*.safetensors")) + list(local_dir.glob("*.bin"))
        if existing:
            logger.info(f"  ✓ {m['local']} — already downloaded, skipping")
            return
        logger.info(f"  ↓ {m['local']} ({m['desc']}) ...")
        snapshot_download(
            repo_id=m["repo_id"],
            local_dir=str(local_dir),
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )

    elif m["fn"] == "single_file":
        target = local_dir / m["filename"]
        if target.exists():
            logger.info(f"  ✓ {m['local']}/{m['filename']} — already downloaded, skipping")
            return
        logger.info(f"  ↓ {m['local']}/{m['filename']} ({m['desc']}) ...")
        hf_hub_download(
            repo_id=m["repo_id"],
            filename=m["filename"],
            local_dir=str(local_dir),
        )

    logger.info(f"  ✓ {m['local']} — done")


def main() -> None:
    """Download all required models for metrics evaluation."""
    setup_logging("download_models")
    logger.info(f"\nDownloading models to {CACHE}/\n")
    for m in MODELS:
        try:
            download_model(m)
        except Exception as e:
            logger.error(f"  ✗ Error downloading {m['local']}: {e}")
            logger.error("    Try again or download manually from HuggingFace.")
    logger.info("\nDone. Make sure models_cache/ is mounted in Docker:")
    logger.info("  -v $(PWD)/models_cache:/app/models_cache\n")


if __name__ == "__main__":
    main()
