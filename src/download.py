"""
Скачивает датасет U4R/SurveyBench с HuggingFace и сохраняет в datasets/.

Установка:
    pip install datasets huggingface_hub

Запуск:
    python download_dataset.py               # скачать всё
    python download_dataset.py --inspect     # только посмотреть структуру датасета
"""

import json
import os
import argparse
from pathlib import Path

# Загружаем .env если есть
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


DATASET_ID = "U4R/SurveyBench"
ROOT_DIR   = Path(__file__).parent.parent
OUT_DIR    = ROOT_DIR / "datasets"


def inspect_dataset():
    """Печатает структуру датасета — поля, типы, пример строки."""
    from datasets import load_dataset

    print(f"Загружаем {DATASET_ID}...")
    ds = load_dataset(DATASET_ID)
    print(f"\nСплиты: {list(ds.keys())}")

    for split_name, split in ds.items():
        print(f"\n--- {split_name} ({len(split)} строк) ---")
        print(f"Колонки: {split.column_names}")
        print(f"Первая строка:")
        row = split[0]
        for k, v in row.items():
            val_repr = str(v)[:200] + "..." if len(str(v)) > 200 else str(v)
            print(f"  {k}: {val_repr}")


def download_dataset():
    """
    Скачивает датасет и сохраняет human surveys в datasets/.

    Структура выходных файлов:
        datasets/
        ├── human_surveys/
        │   ├── Graph Neural Networks.json
        │   ├── Multimodal Large Language Models.json
        │   └── ...
        └── metadata.json
    """
    from datasets import load_dataset

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    human_dir = OUT_DIR / "human_surveys"
    human_dir.mkdir(exist_ok=True)

    print(f"Загружаем {DATASET_ID}...")
    ds = load_dataset(DATASET_ID)

    # Берём первый доступный сплит
    split_name = list(ds.keys())[0]
    split = ds[split_name]

    print(f"Сплит: {split_name}, строк: {len(split)}")
    print(f"Колонки: {split.column_names}")

    # Пытаемся найти нужные поля — имена могут отличаться
    col = split.column_names

    def find_col(candidates):
        for c in candidates:
            if c in col:
                return c
        return None

    topic_col   = find_col(["topic", "domain", "category", "field"])
    title_col   = find_col(["title", "survey_title", "paper_title", "name"])
    text_col    = find_col(["text", "full_text", "content", "body", "survey_text", "paper_text"])
    outline_col = find_col(["outline", "structure", "toc", "sections"])

    print(f"\nОпределённые колонки:")
    print(f"  topic:   {topic_col}")
    print(f"  title:   {title_col}")
    print(f"  text:    {text_col}")
    print(f"  outline: {outline_col}")

    if not text_col and not outline_col:
        print("\n⚠️  Не найдена колонка с текстом. Запусти --inspect чтобы увидеть полную структуру.")
        print("   Доступные колонки:", col)
        # Всё равно сохраняем что есть
        text_col = col[0]

    # Группируем по теме
    by_topic: dict[str, list] = {}
    for row in split:
        topic = str(row.get(topic_col, "unknown")) if topic_col else "unknown"
        if topic not in by_topic:
            by_topic[topic] = []
        by_topic[topic].append(row)

    print(f"\nТем найдено: {len(by_topic)}")
    for t, rows in sorted(by_topic.items()):
        print(f"  {t}: {len(rows)} строк")

    # Сохраняем
    saved = []
    for topic, rows in by_topic.items():
        out = {
            "topic": topic,
            "surveys": []
        }
        for row in rows:
            entry = {
                "title":   str(row.get(title_col,   "")) if title_col   else "",
                "text":    str(row.get(text_col,    "")) if text_col    else "",
                "outline": str(row.get(outline_col, "")) if outline_col else "",
                "raw":     {k: str(v)[:500] for k, v in row.items()},  # копия для отладки
            }
            out["surveys"].append(entry)

        # Имя файла = название темы
        safe_name = topic.replace("/", "_").replace("\\", "_")
        out_file = human_dir / f"{safe_name}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        saved.append(safe_name)
        print(f"  ✓ {out_file.name} ({len(rows)} surveys)")

    # Метаданные
    meta = {
        "dataset_id": DATASET_ID,
        "split": split_name,
        "topics": saved,
        "columns": {
            "topic": topic_col,
            "title": title_col,
            "text": text_col,
            "outline": outline_col,
            "all": col,
        }
    }
    with open(OUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Готово. Сохранено в {OUT_DIR}/")
    print(f"   Тем: {len(saved)}")
    print(f"   Метаданные: {OUT_DIR}/metadata.json")
    print()
    print("Если поля text/outline пустые — запусти --inspect и уточни имена колонок в коде.")


def main():
    parser = argparse.ArgumentParser(description=f"Скачать {DATASET_ID} с HuggingFace")
    parser.add_argument("--inspect", action="store_true",
                        help="Только посмотреть структуру датасета, не сохранять")
    parser.add_argument("--token", default=None,
                        help="HuggingFace токен (если датасет приватный)")
    args = parser.parse_args()

    token = args.token or os.getenv("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token)

    if args.inspect:
        inspect_dataset()
    else:
        download_dataset()


if __name__ == "__main__":
    main()
