# Survey Generation Benchmark

Исследовательская база бакалаврской ВКР:
**«Evaluating effectiveness of agentic systems for scientific survey generation»**

Цель — сравнить несколько систем автоматической генерации научных обзоров на едином бенчмарке и метриках.

---

## Архитектура

```
thesis/
├── models/                      # генерирующие системы (см. §Модели)
│   ├── perplexity_dr/           #   Perplexity Sonar Deep Research
│   ├── surveyforge/             #   SurveyForge (RAG + outline pipeline)
│   ├── surveygen_i/             #   SurveyGen-I (LangGraph async)
│   └── autosurvey/              #   AutoSurvey (конвертация baseline)
│
├── metrics/                     # метрики оценки (см. §Метрики)
│   ├── claimify/                #   atomic-claim extraction (Claimify, 4-stage)
│   ├── veriscore/               #   atomic-claim extraction (VeriScore, alt)
│   ├── factuality/              #   B.1  CitCorrect_k (требует claimify/veriscore)
│   ├── structural/              #   A.1 contr / A.2 term / A.3 rep
│   ├── expert/                  #   C.1 crit / C.2 comp / C.3 open / C.4 modality
│   ├── diversity/               #   Reference diversity (SPECTER2 + PCA)
│   └── surge/                   #   SurGE — LLM-judge (структура + content)
│
├── src/                         # общий код
│   ├── models/base.py           #   BaseModel ABC
│   ├── evaluators/surge.py      #   обёртка над repos/SurGE
│   ├── datasets/                #   загрузка датасетов
│   ├── log_setup.py             #   единая конфигурация логов
│   ├── exceptions.py            #   ThesisError-иерархия
│   └── utils/                   #   вспомогательные скрипты
│
├── datasets/                    # данные (не в git)
│   ├── registry.yaml            #   реестр датасетов
│   ├── SurGE/                   #   бенчмарк SurGE
│   ├── factuality_cache/        #   кэш абстрактов (abstracts.json + full_texts/)
│   └── surge/latex_src/         #   arxiv-tarball'ы для SurGE_reference
│
├── results/                     # результаты (не в git)
│   ├── generations/             #   <dataset>_<model>/<sid>.json
│   │   └── <ds>_<mdl>/sources/  #   per-survey sources-файлы (factuality)
│   ├── scores/                  #   <dataset>_<model>_<metric>_<run_id>/<sid>.json
│   │                            #   + summary.csv
│   └── logs/                    #   <metric_name>.log (append-mode)
│
├── repos/                       # клонированные репозитории (не в git)
│   ├── SurGE/                   #   используется metrics/surge
│   ├── SurveyForge/             #   используется models/surveyforge
│   ├── SurveyGen-I/             #   используется models/surveygen_i
│   ├── AutoSurvey/              #   используется models/autosurvey
│   ├── AlignScore/              #   используется metrics/factuality
│   ├── VeriScore/               #   используется metrics/veriscore
│   └── grobid/                  #   PDF-парсинг (опционально)
│
├── app/main.py                  # Streamlit viewer
├── scripts/                     # инструменты
│   ├── colab_bulk_fetch.py      #   Colab: bulk-fetch evidence для factuality
│   ├── colab_fetch_stats.py     #   Colab: статистика покрытия sources-файлов
│   └── …                        #   (SurGE_reference pipeline + утилиты)
├── docker/                      # базовый Docker-образ
│   └── Dockerfile.base
├── tmp/                         # /tmp в контейнерах: stage-checkpoint'ы factuality, scratch
├── models_cache/                # локальный HF-style кэш моделей (не в git)
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

`results/generations/<dataset>_<model>/<sid>.json`

```json
{
  "id":         "0",
  "dataset_id": "SurGE",
  "model_id":   "perplexity_dr",
  "query":      "...",
  "text":       "## Introduction [1]\n...",
  "success":    true,
  "meta": {
    "model":       "perplexity/sonar-deep-research",
    "latency_sec": 42.1,
    "cost_usd":    0.015,
    "references": [
      {
        "idx":             1,
        "title":           "raw title string",
        "url":             "https://arxiv.org/abs/2301.00123",
        "arxiv_id":        "2301.00123",
        "canonical_title": "Graph Neural Networks…",
        "semantic_scholar_id": "abc123…",
        "doc_id":          42
      }
    ]
  }
}
```

Inline-цитаты в `text` — `[N]` (1-indexed, матчатся с `meta.references[k].idx`),
без пробела перед скобкой и без backslash. Несколько подряд: `[1][2]` или `[1, 2]`.

---

## Метрики

Каждая метрика — отдельная директория `metrics/<name>/` с `main.py`,
`config.yaml`, `Dockerfile`. Запуск через `make evaluate METRIC=<name>` (см.
§Оценка). Per-survey файлы пишутся в
`results/scores/<dataset>_<model>_<metric>_<run_id>/<sid>.json` плюс общий
`summary.csv` рядом.

### Preprocessing — извлечение atomic-claim'ов

`factuality` и `expert` работают на **per-claim** уровне, поэтому до них надо
прогнать один из двух экстракторов claim'ов. Оба пишут в общую директорию
`results/scores/<ds>_<mdl>_claims/` (так что не запускай оба последовательно
без переименования — перезатрут).

| Метрика | Подход | Скорость | Качество |
|---|---|---|---|
| `claimify` | 4-stage pipeline (split → select → disambiguate → decompose), Metropolitansky & Larson ACL 2025 | медленно | выше |
| `veriscore` | sentence-level LLM extraction | быстро | грубее |

### A. Структурное качество — `metrics/structural/`

Три под-метрики в одном run'е:

- **A.1 `M_contr`** — кросс-секционные противоречия (NER → DeBERTa-NLI → LLM-судья)
- **A.2 `M_term`** — терминологическая несогласованность (NER → SPECTER → HDBSCAN → LLM, *exploratory*)
- **A.3 `M_rep`** — кросс-секционная повторяемость (SPECTER cosine + bi-NLI entailment)

Все три — «чем меньше, тем лучше». Модели подгружаются из `models_cache/`
лениво (NER, NLI, SPECTER).

### B. Фактическая корректность — `metrics/factuality/`

Метрика **B.1 `CitCorrect_k`** — доля поддержанных claim'ов в каждой из четырёх
семантических категорий:

1. LLM-судья присваивает claim'у одну из категорий **A** (general/topical),
   **B** (methodology), **C** (quantitative), **D** (critical/comparative).
2. AlignScore (RoBERTa-large) проверяет, поддерживается ли claim текстом
   evidence: abstract, full text или их конкатенацией
   (`full_text_or_abstract`).
3. `CitCorrect_k = |{a : φ(a)=k ∧ Support(a)=1}| / |{a : φ(a)=k}|`.

**Зависимости:** требует `claimify` или `veriscore` — иначе падает с понятным
сообщением. Source-файлы (per-survey, со схемой
[`metrics/factuality/sources_io.py`](metrics/factuality/sources_io.py)) живут в
`results/generations/<ds>_<mdl>/sources/<sid>_sources.json`. Два режима по
config'у `evidence_mode`:

- `internal` — factuality сама строит файл по waterfall'у (cache → corpus.json
  → arxiv Atom API → опц. Semantic Scholar). Существующий файл — checkpoint.
- `external` — файл должен быть подготовлен заранее (например, скриптом
  `scripts/colab_bulk_fetch.py` в Colab — обходит SS rate limits через
  `/paper/batch` + waterfall arxiv/Crossref/OpenAlex). При отсутствии файла —
  явная `FileNotFoundError`.

Stage-чекпоинты переживают рестарты (resume автоматический). LLM-классификация
лежит в `tmp/factuality/classify/<ds>_<mdl>_<judge>/<sid>.json` и шарится между
вариантами evidence; AlignScore-чекпоинты остаются variant-specific:
`tmp/factuality/align/<ds>_<mdl>_<judge>_<variant>/<sid>.json`.

Для `evidence_source: full_text` и `full_text_or_abstract` можно включить
top-k отбор full-text chunk'ов (`full_text_top_k_chunks > 0` в
`metrics/factuality/config.yaml`). Chunk'и режутся так же, как внутри
AlignScore (~350 слов), выбираются по semantic similarity к паре `claim+ref`,
а `abstract`-only режим остаётся без изменений. Суффикс `topk<N>` добавляется
в output-dir, поэтому такие прогоны не перетирают полный full-text baseline.

### C. Экспертные качества письма — `metrics/expert/`

Четыре независимых per-claim суждения, считаются **одним общим LLM-вызовом**
(`judge_all`) для эффективности:

| Код | Что меряет |
|---|---|
| **C.1 `M_crit`** | доля критических claim'ов (ограничения, негативные результаты, противоречия, trade-offs) |
| **C.2 `M_comp`** | доля явных сравнений + check валидности |
| **C.3 `M_open`** | доля open questions / future work |
| **C.4 `M_mod`** | распределение epistemic modality (categorical 1 → explicit uncertainty 5) + Shannon entropy |

Гипотеза: human-written обзоры показывают больше C.1/C.3 и более сбалансированную
C.4 по сравнению с LLM-генерациями. Скрипт валидации против ручной разметки —
`metrics/expert/validate.py` (вход: `results/scores/expert_classes_test.json`,
`expert_modalities_test.json`).

### Diversity — `metrics/diversity/`

Не в A/B/C-таксономии — оценивает не то, *как* написан survey, а *что*
цитирует.

- SPECTER2-эмбеддинги цитируемых статей
- `citation_diversity` — средняя pairwise-cosine дистанция внутри survey'я
- `distribution_shift` — cosine-расстояние centroid'а survey'я от centroid'а
  всего corpus'а
- Опционально PCA-визуализация в одном 2D-пространстве для всех моделей

### SurGE — `metrics/surge/`

Враппер над `repos/SurGE/` (метрики **из оригинальной статьи** SurGE
benchmark'а). Это **не наш** baseline, а метрический suite авторов датасета,
который мы переиспользуем для контекста.

Поддержанные сейчас (через `eval_list` в config):
- `rouge_bleu` → `rouge_1`, `rouge_2`, `rouge_l`, `bleu`
- `sh_recall` — soft heading recall (FlagEmbedding)
- `structure_quality` — LLM-судья (0-5)
- `logic` — LLM-судья (0-5)
- `coverage` — overlap цитируемых статей с ground-truth survey'ём
- `citation_count` / `corpus_match_rate` / `reference_self_cited` — наши
  citation-метрики поверх их пайплайна

**Не подключены** из их статьи: три NLI-relevance метрики
(`Paper/Section/Sentence Level`).

---

## Быстрый старт

### 1. Клонировать репозиторий

```bash
git clone <this-repo> thesis && cd thesis

# Клонировать внешние репозитории
git clone https://github.com/weAIDB/SurveyForge   repos/SurveyForge
git clone https://github.com/weAIDB/SurGE          repos/SurGE
git clone https://github.com/xxx/SurveyGen-I       repos/SurveyGen-I
git clone https://github.com/yzha/AlignScore      repos/AlignScore   # для metrics/factuality

# .venv + полный набор зависимостей + spaCy/NLTK данные + патч AlignScore
make setup
```

`make setup` устанавливает `requirements.txt`, скачивает `spacy en_core_web_sm`
и NLTK `punkt_tab`, `punkt`, ставит AlignScore из `repos/` в editable-режиме и
применяет два совместимых патча через `scripts/patch_alignscore.py` (чинят
`AdamW` import + `load_from_checkpoint` под современные `transformers` /
`pytorch-lightning`). Идемпотентно.

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
make base                      # собирается один раз, кэшируется
make base NO_CACHE=1           # принудительная пересборка (после правок в docker/)
```

### 5. (Опц.) Скачать модели для метрик

```bash
make download-metric-models    # NER, DeBERTa-NLI, SPECTER2, AlignScore checkpoint
```

---

## Запуск без Docker (bare-metal venv)

Все таргеты `make generate` и `make evaluate` принимают флаг `DOCKER`:

```
DOCKER=1   (default) — собрать образ и запустить в контейнере
DOCKER=0             — выполнить напрямую через .venv/bin/python
```

Сценарии bare-metal:
- разработка / отладка на ноутбуке (нет docker daemon'а или дорого пересобирать)
- сервер где docker недоступен / запрещён, есть GPU и venv

Что нужно один раз, до первого `DOCKER=0`-прогона:

```bash
make setup                         # см. шаг 1 — устанавливает всё
# .env должен лежать в корне репо: переменные подхватятся автоматически
# (`set -a; . .env; set +a` встроен в Makefile-цели для DOCKER=0)
```

Дальше — те же команды, что и в docker-режиме, плюс `DOCKER=0`:

```bash
make generate MODEL=perplexity_dr DATASET=SurGE DOCKER=0

make evaluate METRIC=claimify    DATASET=SurGE MODEL=perplexity_dr DOCKER=0
make evaluate METRIC=expert      DATASET=SurGE MODEL=perplexity_dr DOCKER=0
make evaluate METRIC=factuality  DATASET=SurGE MODEL=perplexity_dr DOCKER=0
make evaluate METRIC=structural  DATASET=SurGE MODEL=perplexity_dr DOCKER=0
```

Поддерживаемые в bare-metal сейчас:
- модели: `perplexity_dr`, `surveygen_i` (для последнего ещё нужен
  `pip install -r models/surveygen_i/requirements.txt` — langchain/aiohttp/etc.)
- метрики: `claimify`, `veriscore`, `expert`, `factuality`, `structural`,
  `diversity`, `surge`

**Не поддерживается** в bare-metal:
- `surveyforge` (тяжёлая GPU-инсталляция, FAISS, отдельный `requirements`-сет —
  держим только в docker)

Логи, чекпоинты и кэши пишутся ровно в те же относительные пути, что и в
docker-режиме (`results/`, `tmp/`, `models_cache/`, `datasets/...`).
Корень для stage-чекпоинтов factuality задаётся переменной окружения
`FACTUALITY_CHECKPOINT_ROOT`: Makefile в `DOCKER=1` пробрасывает
`/tmp/factuality` (через volume mount), в `DOCKER=0` переменная не
задаётся и используется дефолт — `tmp/factuality/` относительно репо.

---

## Генерация

Универсальный таргет: `make generate MODEL=<name> DATASET=<id> [GPU=1]`.
`MODEL` выбирает директорию `models/<name>/`, остальное собирается оттуда.

```bash
# OpenRouter-based, без GPU
make generate MODEL=perplexity_dr DATASET=SurGE
make generate MODEL=surveygen_i   DATASET=SurGE

# С GPU (RAG + outline)
make generate MODEL=surveyforge   DATASET=SurGE GPU=1

# AutoSurvey — конвертация готовых baseline результатов (без Docker, локально)
make convert-autosurvey           DATASET=SurGE
```

Результаты — в `results/generations/<DATASET>_<MODEL>/<sid>.json`. Без
`LIMIT` обрабатываются все обзоры датасета. Чтобы ограничить — `LIMIT=N`
(id-based, инклюзивно: только обзоры с `int(sid) <= N`):

```bash
make generate MODEL=perplexity_dr DATASET=SurGE LIMIT=10
```

Resume автоматический: повторный запуск пропускает survey'и, у которых уже
есть `<sid>.json` с `success=true` и непустым `text`.

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

5. **`enrich_arxiv_ids.py`** — пост-процессинг. Для refs с `doc_id != null`,
   но `arxiv_id == null` достаёт чистый title из `corpus.json[doc_id]` и
   ищет его в arxiv Atom API с порогом `STRONG_MATCH_SCORE=0.98`
   (фактически exact-after-normalize). Закрывает гэп «SS знает статью, но не
   связал её с arxiv», который типичен для IEEE/ACM-primary публикаций.
   Идемпотентно; кэш hits/misses в
   `datasets/surge/latex_src/ref_arxiv_cache.json` — повторные прогоны
   мгновенные. Sanity-check: после этого шага выполняется инвариант
   `arxiv_id >= all_cites` и `doc_id >= all_cites` для всех обзоров.

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

# 5. пост-процессинг: добить arxiv_id через corpus-title → arxiv search
python3 scripts/enrich_arxiv_ids.py --limit 40
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

Универсальный таргет: `make evaluate METRIC=<name> DATASET=<id> MODEL=<m>
[LIMIT=N] [EXTRA_ARGS='--flag val']`.

**Порядок запуска метрик:**

```
generate → claimify (или veriscore) → factuality
                                   ↘ expert
        → structural   (независимо)
        → diversity    (независимо)
        → surge        (независимо)
```

```bash
# Шаг 1: извлечь atomic-claim'ы (выбери ОДИН экстрактор; оба пишут в общий
# results/scores/<ds>_<mdl>_claims/, не запускай оба последовательно)
make evaluate DATASET=SurGE MODEL=perplexity_dr METRIC=claimify
# или: METRIC=veriscore  (быстрее, но грубее)

# Шаг 2: B-метрика факт-чека
make evaluate DATASET=SurGE MODEL=perplexity_dr METRIC=factuality

# Шаг 2': C-метрики экспертного письма
make evaluate DATASET=SurGE MODEL=perplexity_dr METRIC=expert

# Независимые
make evaluate DATASET=SurGE MODEL=perplexity_dr METRIC=structural
make evaluate DATASET=SurGE MODEL=perplexity_dr METRIC=diversity
make evaluate DATASET=SurGE MODEL=perplexity_dr METRIC=surge

# Лимит по survey_id (≤ N)
make evaluate DATASET=SurGE MODEL=perplexity_dr METRIC=factuality LIMIT=10

# Произвольные флаги для main.py метрики
make evaluate DATASET=SurGE MODEL=perplexity_dr METRIC=factuality \
    EXTRA_ARGS='--debug-claim-idx 5'
```

### Валидация промптов

Для метрик `expert` и `factuality` есть скрипты валидации против ручной
разметки (precision/recall LLM-классификаторов):

```bash
make validate METRIC=expert       # vs expert_classes_test + expert_modalities_test
make validate METRIC=factuality   # vs factuality_classes_test
```

Результаты per-survey — в `results/scores/<DATASET>_<MODEL>_<METRIC>_<run_id>/`,
плюс `summary.csv` рядом. `<run_id>` собирается из judge_id + judge_comment +
variant из config'а соответствующей метрики.

---

## Colab: bulk-fetch evidence для factuality

`metrics/factuality` в `evidence_mode: external` ожидает готовые
`<sid>_sources.json` под `results/generations/<ds>_<mdl>/sources/`.
Самый надёжный способ их собрать — Colab (Semantic Scholar анонимно жёстко
лимитит сервер, но с Colab'овских IP отдаёт лучше).

Два standalone-скрипта в `scripts/` (без CLI, всё через константы наверху —
скопируй файл в ячейку Colab, поправь параметры, запусти):

| Файл | Что делает |
|---|---|
| `scripts/colab_bulk_fetch.py` | Принимает архив с генерациями (`.tar.gz` / `.zip` / директория), извлекает все ss_id/arxiv_id из refs, прогоняет SS `/paper/batch` + waterfall arxiv Atom / Crossref / OpenAlex / Unpaywall / arxiv-PDF (pymupdf). На выходе — папка с `<sid>_sources.json`. |
| `scripts/colab_fetch_stats.py` | Читает папку выхода, печатает coverage по абстрактам / full_text'у, breakdown по источникам и причинам провалов (`abs_errors` / `text_errors`), список refs которые не удалось найти. |

После прогона — скачай папку `<OUT_DIR>/` и положи её содержимое
в `results/generations/<dataset>_<model>/sources/` на сервере. Дальше
factuality в external-режиме подхватит без лишних сетевых вызовов.

Подробности схемы файлов и точек интеграции — в
[`metrics/factuality/sources_io.py`](metrics/factuality/sources_io.py).

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
from pathlib import Path
from src.datasets.base import DatasetInstance
from src.models.base import BaseModel

class MyModel(BaseModel):
    def __init__(self):
        super().__init__(Path(__file__).parent)   # читает config.yaml

    def generate(self, instance: DatasetInstance) -> dict:
        # доступно: instance.id, instance.query, instance.reference, instance.meta
        ...
        return {
            "text":    survey_text,
            "success": True,
            "meta":    {"model": ..., "latency_sec": ..., "references": [...]},
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    MyModel().run(args.dataset, limit=args.limit)
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
