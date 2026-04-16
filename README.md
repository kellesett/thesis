# Survey Generation Benchmark

Исследовательская база магистерской диссертации:
**«Evaluating effectiveness of agentic systems for scientific survey generation»**

Цель — сравнить несколько систем автоматической генерации научных обзоров на едином бенчмарке и метриках.

---

## Архитектура

```
thesis/
├── models/                      # генерирующие системы
│   ├── perplexity_dr/           #   Perplexity Sonar Deep Research
│   ├── surveyforge/             #   SurveyForge (RAG + outline pipeline)
│   ├── surveygen_i/             #   SurveyGen-I (LangGraph async)
│   └── autosurvey/              #   AutoSurvey (конвертация baseline)
│
├── metrics/                     # метрики оценки
│   ├── surge/                   #   SurGE — LLM-judge (5 осей качества)
│   └── diversity/               #   Reference diversity (SPECTER2 + PCA)
│
├── src/                         # общий код
│   ├── models/base.py           #   BaseModel ABC
│   ├── datasets/                #   загрузка датасетов
│   └── utils/                   #   вспомогательные скрипты
│
├── datasets/                    # данные (не в git)
│   ├── registry.yaml            #   реестр датасетов
│   └── SurGE/                   #   бенчмарк SurGE
│
├── results/                     # результаты (не в git)
│   ├── generations/             #   <dataset>_<model>/<topic_id>.json
│   └── scores/                  #   <dataset>_<model>/<metric>/<topic_id>.json
│
├── repos/                       # клонированные репозитории (не в git)
│   ├── SurGE/
│   ├── SurveyForge/
│   └── SurveyGen-I/
│
├── app/                         # Streamlit viewer
├── scripts/                     # разовые скрипты
├── docker/                      # базовый Docker-образ
│   └── Dockerfile.base
└── experiments/                 # ⚠️ DEPRECATED — см. ниже
```

> **`experiments/`** — устаревший формат ранних экспериментов (exp01–exp04). Не используется в текущем пайплайне. Оставлен только для истории.

---

## Модели

Каждая модель в `models/<name>/` содержит:
- `config.yaml` — параметры (model_id, LLM, датасет)
- `main.py` — класс `XxxModel(BaseModel)` + точка входа
- `Dockerfile` — изолированный запуск
- `requirements.txt` — доп. зависимости поверх `thesis-base` (если нужны)

| Модель | Система | Подход | API |
|--------|---------|--------|-----|
| `perplexity_dr` | Perplexity Sonar Deep Research | Deep Research | OpenRouter |
| `surveyforge` | SurveyForge | RAG + outline pipeline | OpenRouter / local |
| `surveygen_i` | SurveyGen-I | LangGraph async | OpenAI / OpenRouter |
| `autosurvey` | AutoSurvey | конвертация baseline | — |

### Формат выходного файла

`results/generations/<dataset>_<model>/<topic_id>.json`

```json
{
  "topic":      "Graph Neural Networks",
  "text":       "## Introduction\n...",
  "success":    true,
  "meta": {
    "model":       "perplexity/sonar-deep-research",
    "latency_sec": 42.1,
    "cost_usd":    0.015,
    "references":  ["2301.00123", "2210.05234"]
  }
}
```

---

## Метрики

### SurGE (`metrics/surge/`)

LLM-judge оценка по 5 осям качества (Coverage, Relevance, Structure, Synthesis, Consistency).
Реализован по методологии бенчмарка SurGE.

### Diversity (`metrics/diversity/`)

Разнообразие источников на основе эмбеддингов SPECTER2:
- Новизна ссылок (NR), охват домена, PCA-визуализация
- PCA обучается на **эталонных** ссылках из датасета — все модели проецируются в одно 2D-пространство

---

## Быстрый старт

### 1. Клонировать репозиторий

```bash
git clone <this-repo> thesis && cd thesis

# Клонировать внешние репозитории
git clone https://github.com/weAIDB/SurveyForge   repos/SurveyForge
git clone https://github.com/weAIDB/SurGE          repos/SurGE
git clone https://github.com/xxx/SurveyGen-I       repos/SurveyGen-I

# Виртуальное окружение (для локальных утилит)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Переменные окружения

```bash
cp .env.example .env
```

| Переменная | Описание |
|------------|----------|
| `OPENROUTER_API_KEY` | OpenRouter — для perplexity_dr, surveyforge |
| `OPENAI_API_KEY` | OpenAI — для surveygen_i (прямой API) |
| `HF_TOKEN` | HuggingFace — для скачки датасетов |
| `LOCAL_API_BASE` | URL локального OpenAI-совместимого сервера |
| `LOCAL_API_KEY` | Ключ для локального сервера |

### 3. Загрузить данные

```bash
make download       # скачать датасет SurGE
make sfdb           # скачать БД SurveyForge (векторное хранилище)
make sfmodel        # скачать embedding-модель gte-large-en-v1.5
```

### 4. Собрать базовый Docker-образ

```bash
make base           # собирается один раз, кэшируется
make base-clean     # принудительная пересборка (после правок в docker/)
```

---

## Генерация

```bash
# Универсальный запуск
make generate MODEL=perplexity_dr DATASET=SurGE
make generate MODEL=surveygen_i   DATASET=SurGE

# Специализированные таргеты
make generate-sf        DATASET=SurGE   # SurveyForge (GPU)
make generate-sf-cpu    DATASET=SurGE   # SurveyForge (CPU, для отладки)
make generate-sgi       DATASET=SurGE   # SurveyGen-I (CPU)

# AutoSurvey — конвертация готовых baseline результатов (без Docker)
make convert-autosurvey DATASET=SurGE
```

Результаты сохраняются в `results/generations/<DATASET>_<MODEL>/`.

---

## SurGE_reference — human-written surveys as baseline

Pipeline, который приводит **человеческие** обзоры из SurGE к тому же формату,
что и LLM-выходы (`results/generations/<DATASET>_<MODEL>/<sid>.json`). Цель —
прогнать все метрики на «человеческом бейзлайне» через те же code paths, что и
на LLM-системах.

Состоит из четырёх шагов, все **resume-friendly** (повторный запуск пропускает
готовое, докидывает недостающее):

1. **`fetch_reference_latex.py`** — находит arxiv-id по названию обзора
   (multi-strategy title search + author verification) и качает source tarball
   в `datasets/surge/latex_src/<arxiv_id>/`.
2. **`merge_latex.sh`** — `latexpand` с инлайном `\input{}`/`\include{}` и
   вставкой `.bbl` → `merged.tex`.
3. **`match_ss_to_bibitems.py`** — сопоставляет `\bibitem`-ы из `merged.tex` с
   Semantic Scholar `/references`-ответом для arxiv-id статьи. Три режима:
   - `string` — детерминированный LCS, без сети, бесплатно;
   - `llm` — LLM-диспатчер (OpenRouter, полный bibitem-список в промпте);
   - `hybrid` — string отбирает top-K кандидатов, LLM re-ranks (рекомендуется:
     ~20× дешевле `llm`-режима при сравнимом качестве).
4. **`build_surge_reference.py`** — парсит `merged.tex` в body markdown с
   инлайн-цитатами `[N]`, обогащает ссылки данными из SS, пишет `merged.md`
   рядом с `.tex`, и собирает финальный
   `results/generations/SurGE_reference/<sid>.json`. Поля `arxiv_id` и
   `semantic_scholar_id` в каждой reference-записи — обязательные, `null` при
   отсутствии.

### End-to-end запуск через Make

```bash
make surge-reference                           # все обзоры, hybrid-режим
make surge-reference MODE=string LIMIT=40      # первые 40, без LLM (бесплатно)
make surge-reference MODE=hybrid LIMIT=40      # первые 40, hybrid (по умолчанию)
```

Для `MODE=hybrid` и `MODE=llm` нужен `OPENROUTER_API_KEY` в `.env`. Для стабильной
работы SS recommended — получить бесплатный Semantic Scholar API-ключ и
положить в `SEMANTIC_SCHOLAR_API_KEY` (без него лимит ~100 req / 5 min / IP).

### Шаги по отдельности

Иногда удобнее запускать стадии вручную (другой обзор, другой `--parallel`,
другая модель):

```bash
# 1. arxiv-поиск + tarball-скачка
python3 scripts/fetch_reference_latex.py --limit 40

# 2. LaTeX-merge (все папки в latex_src/)
bash scripts/merge_latex.sh

# 3. matching. Режим выбирается флагом --mode
python3 scripts/match_ss_to_bibitems.py \
    --mode hybrid --limit 40 --top-k 5 --parallel 50

# 4. финальная сборка с SS-enrichment
python3 scripts/build_surge_reference.py --mode hybrid --limit 40
```

Промежуточные артефакты остаются на диске для отладки:

```
datasets/surge/latex_src/<arxiv_id>/
    merged.tex                          # после шага 2
    ss_references.json                  # кэш SS-ответа (шаг 3)
    ss_matches_{string,llm,hybrid}.json # mapping bibitem ↔ SS (шаг 3)
    merged.md                           # clean GFM (шаг 4)
```

---

## Оценка

```bash
# SurGE метрика
make evaluate           DATASET=SurGE MODEL=perplexity_dr METRIC=surge

# Diversity метрика
make evaluate-diversity DATASET=SurGE MODEL=perplexity_dr
```

Результаты сохраняются в `results/scores/<DATASET>_<MODEL>/`.

---

## Viewer

Streamlit-приложение для просмотра и сравнения результатов:

```bash
make viewer
# → http://localhost:8501
```

Возможности:
- Просмотр сгенерированных обзоров по темам
- Сравнение моделей по SurGE-скорам
- PCA-визуализация diversity с наложением нескольких моделей

---

## BaseModel — добавить новую модель

Все модели наследуются от `src/models/base.py`:

```python
from src.models.base import BaseModel

class MyModel(BaseModel):
    def __init__(self):
        super().__init__(Path(__file__).parent)   # читает config.yaml

    def generate(self, instance: dict) -> dict:
        # instance: {"id": ..., "topic": ..., "references": [...]}
        ...
        return {
            "text":    survey_text,
            "success": True,
            "meta":    {"model": ..., "latency_sec": ..., "references": [...]},
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="SurGE")
    args = parser.parse_args()
    MyModel().run(args.dataset)
```

`BaseModel.run()` обрабатывает датасет, сохраняет результаты и поддерживает `resume`.

---

## Утилиты

```bash
make models         # проверить список доступных моделей
make models-ping    # пинг API-эндпоинтов
make enrich         # обогатить ссылки arxiv-метаданными
make clean          # удалить Docker-образы
```
