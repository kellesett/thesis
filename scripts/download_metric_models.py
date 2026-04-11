#!/usr/bin/env python3
"""
scripts/download_metric_models.py
Загружает локально все модели, необходимые для метрик structural / factuality / expert.
Запускай один раз из корня репозитория:

    python scripts/download_metric_models.py

Модели сохраняются в models_cache/ и монтируются в Docker через volume.
Итоговый размер: ~2.4 GB (см. таблицу ниже).

  fleonce--iter-scierc-scideberta-full    ~740 MB  (NER, группа A)
  microsoft--deberta-v2-xlarge-mnli       ~900 MB  (NLI, группы A/B/C)
  yzha--AlignScore-large                  ~355 MB  (factuality support, группы B/C)
  sentence-transformers--allenai-specter  ~400 MB  (embedding pre-filter, группа A)

Примечание: AlignScore ожидает checkpoint-файл .ckpt, а не HuggingFace safetensors.
После скачивания файл будет лежать в:
  models_cache/yzha--AlignScore-large/AlignScore-large.ckpt
"""
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, hf_hub_download
except ImportError:
    print("Установи зависимости: pip install huggingface_hub")
    sys.exit(1)

ROOT = Path(__file__).parent.parent
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
    local_dir = CACHE / m["local"]
    local_dir.mkdir(exist_ok=True)

    if m["fn"] == "snapshot":
        # Проверяем, не скачана ли уже модель
        existing = list(local_dir.glob("*.safetensors")) + list(local_dir.glob("*.bin"))
        if existing:
            print(f"  ✓ {m['local']} — уже скачана, пропускаем")
            return
        print(f"  ↓ {m['local']} ({m['desc']}) ...")
        snapshot_download(
            repo_id=m["repo_id"],
            local_dir=str(local_dir),
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )

    elif m["fn"] == "single_file":
        target = local_dir / m["filename"]
        if target.exists():
            print(f"  ✓ {m['local']}/{m['filename']} — уже скачан, пропускаем")
            return
        print(f"  ↓ {m['local']}/{m['filename']} ({m['desc']}) ...")
        hf_hub_download(
            repo_id=m["repo_id"],
            filename=m["filename"],
            local_dir=str(local_dir),
        )

    print(f"  ✓ {m['local']} — готово")


def main() -> None:
    print(f"\nЗагрузка моделей в {CACHE}/\n")
    for m in MODELS:
        try:
            download_model(m)
        except Exception as e:
            print(f"  ✗ Ошибка при загрузке {m['local']}: {e}")
            print("    Попробуй ещё раз или скачай вручную с HuggingFace.")
    print("\nГотово. Убедись что models_cache/ смонтирована в Docker:")
    print("  -v $(PWD)/models_cache:/app/models_cache\n")


if __name__ == "__main__":
    main()
