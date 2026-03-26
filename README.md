# Survey Generation Benchmark

Экспериментальная база для магистерской диссертации:
**"Evaluating effectiveness of agentic systems for scientific survey generation"**

---

## Структура репозитория

```
├── src/
│   ├── generate.py       # генерация обзоров (все системы)
│   ├── evaluate.py       # LLM-judge оценка (SWR + CWR)
│   └── download.py       # скачка датасета с HuggingFace
├── experiments/
│   ├── Dockerfile.base               # базовый Docker-образ
│   ├── exp01_openai_dr/              # OpenAI o4-mini Deep Research
│   ├── exp02_perplexity_dr/          # Perplexity Sonar Deep Research
│   └── exp03_gemini_pro/             # Gemini 2.5 Pro (academic pipeline)
├── configs/
│   └── judges.json       # конфиг LLM-судей
├── repos/                # клонированные репозитории (не в git)
├── datasets/             # данные (не в git)
├── results/              # результаты генерации (не в git)
└── eval_results/         # результаты оценки (не в git)
```

Каждый эксперимент в `experiments/expXX_name/` содержит:
- `config.yaml` — параметры: система, топики, методы оценки
- `main.py` — полный пайплайн: генерация → оценка
- `Dockerfile` — запуск в изолированном контейнере

---

## Реализованные эксперименты

| ID | Система | Модель | Тип | Статус |
|----|---------|--------|-----|--------|
| exp01 | OpenAI Deep Research | `openai/o4-mini-deep-research` | Deep Research | ✅ |
| exp02 | Perplexity Sonar DR | `perplexity/sonar-deep-research` | Deep Research | ✅ |
| exp03 | Gemini 2.5 Pro | `google/gemini-2.5-pro` | Academic pipeline | ✅ |

Дополнительные системы, доступные в `src/generate.py`, но без отдельного эксперимента пока:

| ID в коде | Модель | Примечание |
|-----------|--------|------------|
| `autosurvey_gpt4o` | `openai/gpt-4o` | Academic pipeline |
| `autosurvey_local` | `LOCAL_MODEL` из `.env` | Локальный DeepSeek |

### Что делает каждый эксперимент

`main.py` запускает два шага последовательно:

1. **Генерация** (`src/generate.py`) — генерирует обзоры по N темам из SurveyBench. Сохраняет в `results/{experiment}/generated/{system}__{topic}.json`.

2. **Оценка** (`src/evaluate.py`) — сравнивает сгенерированные обзоры с человеческими эталонами двумя методами:
   - **SWR** (Score Win Rate) — судья независимо ставит балл 0–100 каждому тексту, победитель — у кого выше
   - **CWR** (Comparative Win Rate) — судья видит оба текста и выбирает лучший напрямую (A / B / TIE)
   - Оценка по двум измерениям: `outline` (структура) и `content` (полный текст)
   - Win rate = (wins + 0.5 × ties) / total

### Датасет — SurveyBench (SurveyForge)

10 тем из области CS:

- 3D Gaussian Splatting
- 3D Object Detection in Autonomous Driving
- Evaluation of Large Language Models
- LLM-based Multi-Agent
- Generative Diffusion Models
- Graph Neural Networks
- Hallucination in Large Language Models
- Multimodal Large Language Models
- Retrieval-Augmented Generation for Large Language Models
- Vision Transformers

---

## Быстрый старт

### 1. Клонировать репо и зависимости

```bash
git clone <this-repo> thesis && cd thesis

# Склонировать SurveyForge (нужен для тем и эталонных ссылок)
git clone https://github.com/weAIDB/SurveyForge repos/SurveyForge

# Создать окружение
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Настроить переменные окружения

```bash
cp .env.example .env
# Заполни .env:
#   OPENROUTER_API_KEY — ключ OpenRouter (нужен для exp01, exp02, exp03)
#   HF_TOKEN          — токен HuggingFace (для скачки датасета, если приватный)
```

### 3. Скачать датасет (опционально)

```bash
# Посмотреть структуру
make inspect

# Скачать
make download
# → datasets/human_surveys/{topic}.json
```

---

## Запуск экспериментов

### Вариант A — через Make + Docker (рекомендуется)

```bash
# Собрать базовый образ (один раз, или после изменений в src/)
make base

# Запустить эксперименты
make exp01   # OpenAI Deep Research
make exp02   # Perplexity Sonar DR
make exp03   # Gemini 2.5 Pro
```

Результаты монтируются в `results/` и `eval_results/` на хосте.

### Вариант B — локально без Docker

```bash
source .venv/bin/activate

# Один эксперимент
python experiments/exp01_openai_dr/main.py

# Или напрямую — только генерация
python src/generate.py --systems openai_o4_mini_dr --topics 5

# Только оценка
python src/evaluate.py --swr --cwr --eval outline,content \
    --systems openai_o4_mini_dr
```

---

## Настройка конфига эксперимента

Каждый `experiments/expXX/config.yaml`:

```yaml
experiment: exp01_openai_dr
system: openai_o4_mini_dr   # ключ из SYSTEMS в src/generate.py
topics: 5                   # сколько тем из SurveyBench

evaluation:
  methods: [swr, cwr]                   # методы оценки
  eval_types: [outline, content]        # что оцениваем
  judges: configs/judges.json           # путь к конфигу судей
```

### Конфиг судей (`configs/judges.json`)

Список LLM-судей. Каждый судья вызывается `n` раз, результаты усредняются:

```json
[
  {
    "name": "gemini-flash",
    "url": "https://openrouter.ai/api/v1",
    "model": "google/gemini-2.0-flash-001",
    "api_key_env": "OPENROUTER_API_KEY",
    "n": 1
  },
  {
    "name": "local-deepseek",
    "url": "http://localhost:8000/v1",
    "model": "deepseek-chat",
    "api_key_env": "LOCAL_API_KEY",
    "n": 1
  }
]
```

---

## Добавить новый эксперимент

```bash
cp -r experiments/exp01_openai_dr experiments/exp04_autosurvey_gpt4o
```

Отредактировать `experiments/exp04_autosurvey_gpt4o/config.yaml`:

```yaml
experiment: exp04_autosurvey_gpt4o
system: autosurvey_gpt4o
topics: 5
evaluation:
  methods: [swr, cwr]
  eval_types: [outline, content]
  judges: configs/judges.json
```

Добавить в `Makefile`:

```makefile
exp04: base
	docker build -f experiments/exp04_autosurvey_gpt4o/Dockerfile -t thesis-exp04 .
	docker run --rm $(EXP_VOLUMES) thesis-exp04
```

---

## Выходные данные

```
results/
└── exp01_openai_dr/
    ├── generated/
    │   ├── openai_o4_mini_dr__Graph_Neural_Networks.json
    │   └── ...
    └── eval/
        ├── swr__openai_o4_mini_dr__Graph_Neural_Networks__outline.json
        ├── cwr__openai_o4_mini_dr__Graph_Neural_Networks__content.json
        └── summary.csv    ← итоговая таблица win rates
```

`summary.csv` содержит win rate по каждой системе / методу / типу оценки.
