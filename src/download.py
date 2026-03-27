"""
Скачивает датасет U4R/SurveyBench с HuggingFace и сохраняет в datasets/.

Датасет хранит файлы (markdown/PDF/etc.) — не стандартный parquet-формат,
поэтому используем snapshot_download для получения всего репозитория,
затем разбираем файлы вручную.

Установка:
    pip install huggingface_hub

Запуск:
    python src/download.py               # скачать и разобрать
    python src/download.py --inspect     # показать структуру скачанного репо
    python src/download.py --raw         # только скачать без разбора

Выходные файлы:
    datasets/
    ├── raw/                     # оригинальные файлы из HuggingFace репо
    └── human_surveys/
        ├── Graph_Neural_Networks.json
        └── ...
"""

import os
import re
import json
import argparse
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DATASET_ID = "U4R/SurveyBench"
ROOT_DIR   = Path(__file__).parent.parent
RAW_DIR    = ROOT_DIR / "datasets" / "raw"
OUT_DIR    = ROOT_DIR / "datasets" / "human_surveys"


# Маппинг папок/тегов репозитория → наши ID тем (SurveyBench topics.txt)
# Заполним после --inspect, пока эвристика по имени файла
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
    """Скачивает весь репозиторий датасета в datasets/raw/."""
    from huggingface_hub import snapshot_download

    print(f"Скачиваем {DATASET_ID} → {RAW_DIR} ...")
    path = snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        local_dir=str(RAW_DIR),
        token=token,
        ignore_patterns=["*.git*", ".gitattributes"],
    )
    print(f"[OK] Скачано в {path}")
    return Path(path)


def inspect(raw_path: Path) -> None:
    """Печатает дерево скачанного репозитория."""
    print(f"\nСтруктура {raw_path}:")
    for p in sorted(raw_path.rglob("*")):
        if ".git" in p.parts:
            continue
        indent = "  " * (len(p.relative_to(raw_path).parts) - 1)
        size = f"  ({p.stat().st_size // 1024} KB)" if p.is_file() else ""
        print(f"{indent}{'[+] ' if p.is_dir() else '[-] '}{p.name}{size}")


def guess_topic(path: Path) -> str | None:
    """Пытается определить тему по пути/имени файла."""
    key = path.stem.lower().replace("-", "_").replace(" ", "_")
    parent = path.parent.name.lower().replace("-", "_").replace(" ", "_")
    for token in [key, parent]:
        for alias, topic in TOPIC_ALIASES.items():
            if alias in token:
                return topic
    return None


def parse_and_save(raw_path: Path) -> None:
    """Разбирает скачанные файлы и сохраняет в datasets/human_surveys/."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Собираем все текстовые файлы
    text_extensions = {".md", ".txt", ".json"}
    files = [p for p in raw_path.rglob("*")
             if p.is_file()
             and p.suffix.lower() in text_extensions
             and ".git" not in p.parts
             and p.name not in ("README.md", "LICENSE", "NOTICE")]

    print(f"\nНайдено {len(files)} текстовых файлов")

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

    # Сохраняем по темам
    saved = []
    for topic, surveys in sorted(by_topic.items()):
        safe = re.sub(r"[^\w]", "_", topic).strip("_")
        out_file = OUT_DIR / f"{safe}.json"
        with open(out_file, "w", encoding="utf-8") as fp:
            json.dump({"topic": topic, "surveys": surveys}, fp,
                      ensure_ascii=False, indent=2)
        saved.append(safe)
        print(f"  >> {out_file.name}  ({len(surveys)} survey(s))")

    if unmatched:
        print(f"\n[??] Не удалось определить тему для {len(unmatched)} файлов:")
        for f in unmatched:
            print(f"     {f.relative_to(raw_path)}")
        print("  → Добавь маппинг в TOPIC_ALIASES в src/download.py")

    meta = {
        "dataset_id": DATASET_ID,
        "topics": saved,
        "unmatched": [str(f.relative_to(raw_path)) for f in unmatched],
    }
    with open(ROOT_DIR / "datasets" / "metadata.json", "w") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)

    print(f"\n[OK] Готово. Тем сохранено: {len(saved)}")
    if unmatched:
        print("   Запусти --inspect чтобы посмотреть структуру и уточнить TOPIC_ALIASES")


def _extract_outline(text: str) -> str:
    """Вытаскивает markdown-заголовки как outline."""
    headers = [line.strip() for line in text.splitlines()
               if line.startswith("#")]
    return "\n".join(headers[:30])


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Скачать {DATASET_ID} с HuggingFace")
    parser.add_argument("--inspect", action="store_true",
                        help="Показать структуру скачанного репо (без разбора)")
    parser.add_argument("--raw", action="store_true",
                        help="Только скачать, не разбирать по темам")
    parser.add_argument("--token", default=None,
                        help="HuggingFace токен (переопределяет HF_TOKEN из .env)")
    args = parser.parse_args()

    token = args.token or get_token()

    raw_path = snapshot(token)

    if args.inspect or args.raw:
        inspect(raw_path)
        return

    parse_and_save(raw_path)


if __name__ == "__main__":
    main()
