"""
Скачивает embedding-модель Alibaba-NLP/gte-large-en-v1.5 (~1.7 GB)
в datasets/gte-large-en-v1.5/ для использования в SurveyForge.

Запуск:
    python src/utils/download_sf_model.py
    make sfmodel
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent.parent
OUT  = ROOT / "datasets" / "gte-large-en-v1.5"
MODEL_ID = "Alibaba-NLP/gte-large-en-v1.5"

load_dotenv(ROOT / ".env")

def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        size_mb = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) // (1024 * 1024)
        print(f"[OK] Модель уже есть: {OUT}  ({size_mb} MB)")
        return

    print(f"[>>] Скачиваю {MODEL_ID} → {OUT}")
    print(f"     Размер: ~1.7 GB")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[!!] Установи huggingface_hub: pip install huggingface_hub")
        sys.exit(1)

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(OUT),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.onnx", "onnx/*"],   # ONNX не нужен для SentenceTransformer
    )

    size_mb = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) // (1024 * 1024)
    print(f"\n[OK] Скачано: {OUT}  ({size_mb} MB)")
    print(f"[::] Путь для config.yaml:")
    print(f"     embedding_model: /app/datasets/gte-large-en-v1.5")

if __name__ == "__main__":
    main()
