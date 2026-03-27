#!/usr/bin/env python
"""
Experiment: exp04_surveyforge
System:     SurveyForge (ACL 2025) — RAG-based academic survey generation
Pipeline:   SurveyForge generation → SWR/CWR evaluation → summary

Backends:
  openrouter — облачные модели (gpt-4o-mini, claude, etc.) через OpenRouter
  local      — локальный OpenAI-compatible сервер (Qwen, DeepSeek, etc.)

.env:
  OPENROUTER_API_KEY  — для backend: openrouter
  LOCAL_API_BASE      — base URL локального сервера (напр. http://host:30000/v1)
  LOCAL_API_KEY       — ключ локального сервера (или "none")
  LOCAL_MODEL         — имя модели (напр. qwen2.5-72b)

Примечание о fairness:
  SurveyForge имеет структурное преимущество перед DR-системами:
  доступ к curated FAISS-индексу по 530K arXiv статей (2012–Sep 2024).
  Сравнивать следует с AutoSurvey/LiRA, а не с OpenAI/Perplexity DR.
"""
import os
import sys
import json
import time
import subprocess
import tempfile
import yaml
from pathlib import Path

EXP_DIR = Path(__file__).parent
ROOT    = EXP_DIR.parent.parent
CFG     = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
PYTHON  = sys.executable
SRC     = ROOT / "src"
SF_CODE = ROOT / "repos" / "SurveyForge" / "code"
OUT     = ROOT / "results" / CFG["experiment"]


def step(label: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {label}")
    print(f"{'-' * 60}")


def resolve_backend() -> tuple[str, str, str]:
    """
    Возвращает (api_url, api_key, model) в зависимости от backend в config.yaml.
    Имя модели всегда берётся из config.yaml — поле model.

    Для SurveyForge важно:
      - "deepseek" в имени → OpenAI SDK, api_url трактуется как base_url
      - иначе → raw requests.POST, api_url должен быть полным URL до /chat/completions
    """
    backend = CFG.get("backend", "openrouter")
    model   = CFG["model"]

    if backend == "local":
        base    = os.getenv("LOCAL_API_BASE", "http://localhost:8000/v1").rstrip("/")
        key     = os.getenv("LOCAL_API_KEY", "none")
        api_url = f"{base}/chat/completions"
    else:  # openrouter
        key = os.getenv("OPENROUTER_API_KEY", "")
        if not key:
            print("[!!] OPENROUTER_API_KEY не задан в .env")
            sys.exit(1)
        api_url = "https://openrouter.ai/api/v1/chat/completions"

    print(f"  [::] backend   : {backend}")
    print(f"  [::] api_url   : {api_url}")
    print(f"  [::] model     : {model}")
    return api_url, key, model


def load_topics(n: int) -> list[str]:
    """Загружает первые n тем из SurveyBench."""
    topics_file = ROOT / "repos" / "SurveyForge" / "SurveyBench" / "topics.txt"
    if not topics_file.exists():
        default = [
            "Graph Neural Networks",
            "Multimodal Large Language Models",
            "Retrieval-Augmented Generation for Large Language Models",
            "Hallucination in Large Language Models",
            "Generative Diffusion Models",
            "Vision Transformers",
            "LLM-based Multi-Agent",
            "Evaluation of Large Language Models",
            "3D Gaussian Splatting",
            "3D Object Detection in Autonomous Driving",
        ]
        return default[:n]
    with open(topics_file) as f:
        topics = [line.strip() for line in f if line.strip()]
    return topics[:n]


def run_surveyforge(topic: str, save_dir: Path, api_url: str, api_key: str, model: str) -> dict | None:
    """
    Запускает SurveyForge для одной темы через subprocess.
    Возвращает dict {'survey': str, 'reference': dict} или None при ошибке.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON,
        str(SF_CODE / "main.py"),
        "--topic",                 topic,
        "--saving_path",           str(save_dir),
        "--db_path",               CFG["db_path"],
        "--survey_outline_path",   CFG["survey_outline_path"],
        "--embedding_model",       CFG["embedding_model"],
        "--model",                 model,
        "--api_key",               api_key,
        "--api_url",               api_url,
        "--section_num",           str(CFG["section_num"]),
        "--subsection_len",        str(CFG["subsection_len"]),
        "--outline_reference_num", str(CFG["outline_reference_num"]),
        "--rag_num",               str(CFG["rag_num"]),
        "--rag_max_out",           str(CFG["rag_max_out"]),
    ]

    env = {**os.environ, "PYTHONPATH": str(SF_CODE)}

    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"    [!!] SurveyForge упал для темы: {topic!r}  (returncode={e.returncode})")
        return None

    out_json = save_dir / f"{topic}.json"
    if not out_json.exists():
        print(f"    [!!] Файл не найден: {out_json}")
        return None

    with open(out_json, encoding="utf-8") as f:
        return json.load(f)


def generate_all(topics: list[str], gen_out: Path, api_url: str, api_key: str, model: str) -> None:
    """Запускает SurveyForge для каждой темы, сохраняет в нашем формате."""
    gen_out.mkdir(parents=True, exist_ok=True)
    print(f"\n[>>] Генерация: {len(topics)} тем → {gen_out}")

    for i, topic in enumerate(topics, 1):
        safe_name = topic.replace(" ", "_").replace("/", "-")
        result_file = gen_out / f"{safe_name}.json"

        if result_file.exists():
            print(f"  [>>] [{i}/{len(topics)}] SKIP (уже есть): {topic}")
            continue

        print(f"\n  [>>] [{i}/{len(topics)}] {topic}")
        t0 = time.time()

        with tempfile.TemporaryDirectory() as tmp:
            sf_result = run_surveyforge(
                topic, Path(tmp) / safe_name,
                api_url, api_key, model
            )

        elapsed = time.time() - t0

        if sf_result is None:
            print(f"  [!!] Пропускаем: {topic}")
            continue

        record = {
            "topic":      topic,
            "system":     "surveyforge",
            "experiment": CFG["experiment"],
            "backend":    CFG.get("backend", "openrouter"),
            "model":      model,
            "text":       sf_result.get("survey", ""),
            "references": sf_result.get("reference", {}),
            "elapsed_s":  round(elapsed, 1),
        }

        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        wc = len(record["text"].split())
        print(f"  [OK] {topic}  ({wc} words, {elapsed:.0f}s)")

    generated = list(gen_out.glob("*.json"))
    print(f"\n  [::] Итого сгенерировано: {len(generated)}/{len(topics)} тем")


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    system   = CFG["system"]
    n_topics = CFG.get("topics", 5)
    eval_cfg = CFG.get("evaluation", {})
    methods  = eval_cfg.get("methods",    ["swr", "cwr"])
    etypes   = eval_cfg.get("eval_types", ["outline", "content"])
    judges   = eval_cfg.get("judges",     str(ROOT / "configs" / "judges.json"))
    k_refs   = eval_cfg.get("k_refs",     1)

    gen_out  = OUT / "generated"
    eval_out = OUT / "eval"

    print(f"\n{'=' * 60}")
    print(f"  Experiment : {CFG['experiment']}")
    print(f"  System     : {system}")
    print(f"  Topics     : {n_topics}")

    # Определяем backend
    api_url, api_key, model = resolve_backend()

    print(f"  Methods    : {methods}")
    print(f"{'=' * 60}")

    # Проверяем БД
    db_path = Path(CFG["db_path"])
    if not db_path.exists():
        print(f"\n[!!] БД SurveyForge не найдена: {db_path}")
        print("     Запусти: make sfdb")
        sys.exit(1)

    # Step 1: Topics
    step("Step 1 / Load Topics")
    topics = load_topics(n_topics)
    for t in topics:
        print(f"    * {t}")

    # Step 2: Generation
    step("Step 2 / Generation via SurveyForge")
    generate_all(topics, gen_out, api_url, api_key, model)

    # Step 3: Evaluation
    step("Step 3 / Evaluation")
    eval_cmd = [
        PYTHON, str(SRC / "evaluate.py"),
        "--eval",    ",".join(etypes),
        "--judges",  judges,
        "--systems", system,
        "--results-dir", str(gen_out),
        "--out",     str(eval_out),
        "--resume",
    ]
    eval_cmd += ["--k-refs", str(k_refs)]
    if "swr" in methods: eval_cmd.append("--swr")
    if "cwr" in methods: eval_cmd.append("--cwr")
    subprocess.run(eval_cmd, check=True)

    print(f"\n{'=' * 60}")
    print(f"  [OK] {CFG['experiment']} complete")
    print(f"  generated >> {gen_out}")
    print(f"  eval      >> {eval_out}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
