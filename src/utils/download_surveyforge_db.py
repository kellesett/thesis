"""
Скачивает базу данных SurveyForge (U4R/SurveyForge_database) с HuggingFace.

Реальная структура после скачивания:
  datasets/surveyforge_db/
  ├── database/                         ← FAISS индексы и TinyDB
  │   ├── arxiv_paper_db_with_cc.json
  │   ├── arxivid_to_index_abs.json
  │   ├── faiss_paper_title_embeddings_FROM_2012_0101_TO_240926.bin
  │   ├── faiss_paper_title_abs_embeddings_FROM_2012_0101_TO_240926.bin
  │   ├── surveys_arxiv_paper_db.json
  │   ├── surveys_arxivid_to_index_abs.json
  │   ├── faiss_survey_title_embeddings_FROM_1501_TO_2409_gte.bin
  │   └── faiss_survey_title_abs_embeddings_FROM_1501_TO_2409_gte.bin
  ├── Final_outline.zip                 ← outline-база для heuristics
  └── Final_outline_First.zip

Итого ~6-10 GB. Нужен HF_TOKEN если датасет требует авторизации.

Запуск:
    python src/utils/download_surveyforge_db.py
    python src/utils/download_surveyforge_db.py --out datasets/surveyforge_db
    python src/utils/download_surveyforge_db.py --check   # только проверить что всё на месте
"""

import os
import sys
import argparse
from pathlib import Path

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
    """Проверяет наличие всех нужных файлов. Возвращает True если всё на месте."""
    db_dir = out_dir / DB_SUBDIR
    print(f"\n[>>] Проверка БД в {out_dir}\n")

    all_ok = True
    total_size = 0

    print(f"  [{DB_SUBDIR}/]")
    for fname in REQUIRED_DB_FILES:
        fpath = db_dir / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / 1024 / 1024
            total_size += size_mb
            print(f"    [OK] {fname}  ({size_mb:.0f} MB)")
        else:
            print(f"    [!!] MISSING: {fname}")
            all_ok = False

    print(f"\n  [root/]")
    for fname in REQUIRED_ROOT_FILES:
        fpath = out_dir / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / 1024 / 1024
            total_size += size_mb
            print(f"    [OK] {fname}  ({size_mb:.0f} MB)")
        else:
            print(f"    [!!] MISSING: {fname}")
            all_ok = False

    print(f"\n  Total: {total_size:.0f} MB")
    if all_ok:
        print(f"\n[OK] Все файлы на месте — БД готова к использованию")
        print(f"     db_path для SurveyForge: {db_dir}\n")
    else:
        print(f"\n[??] Некоторые файлы отсутствуют — запусти без --check для скачивания\n")
    return all_ok


def download(out_dir: Path, token: str | None) -> None:
    """Скачивает всю базу через snapshot_download."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[!!] huggingface_hub не установлен: pip install huggingface_hub")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[>>] Скачиваем {DATASET_ID}")
    print(f"[>>] Назначение: {out_dir}")
    print(f"[??] Ожидаемый размер: ~6-10 GB. Это может занять несколько минут.\n")

    try:
        snapshot_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            local_dir=str(out_dir),
            token=token,
            ignore_patterns=["*.git*", ".gitattributes", "README*"],
        )
    except Exception as e:
        print(f"\n[!!] Ошибка при скачивании: {e}")
        print("     Если датасет приватный — задай HF_TOKEN в .env")
        sys.exit(1)

    print()
    check(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Скачать БД SurveyForge ({DATASET_ID})"
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_OUT),
        help=f"Куда сохранить (default: {DEFAULT_OUT})"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Только проверить наличие файлов, не скачивать"
    )
    parser.add_argument(
        "--token", default=None,
        help="HuggingFace токен (переопределяет HF_TOKEN из .env)"
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    token   = args.token or os.getenv("HF_TOKEN")

    if args.check:
        ok = check(out_dir)
        sys.exit(0 if ok else 1)

    download(out_dir, token)
    print(f"\n[>>] db_path для передачи в SurveyForge: {out_dir / DB_SUBDIR}")


if __name__ == "__main__":
    main()
