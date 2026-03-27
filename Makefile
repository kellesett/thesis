# Makefile
# Запускай из корня репозитория.
# Предварительно: cp .env.example .env && заполни ключи

.PHONY: help setup setup-sf base exp01 exp02 exp03 exp04 \
        download inspect models models-ping \
        sfdb sfdb-check sfmodel clean

PYTHON     = .venv/bin/python
EXP_VOLUMES = --env-file .env \
              -v "$(PWD)/results:/app/results" \
              -v "$(PWD)/eval_results:/app/eval_results" \
              -v "$(PWD)/datasets:/app/datasets"

## Справка
help:
	@echo ""
	@echo "  [Окружение]"
	@echo "  make setup       — .venv + requirements.txt (минимум для локальных команд)"
	@echo "  make setup-sf    — .venv + requirements exp04 (torch, faiss, sentence-transformers)"
	@echo ""
	@echo "  [Docker]"
	@echo "  make base        — собрать базовый образ thesis-base (для exp01-03)"
	@echo "  make exp01       — OpenAI Deep Research:   генерация + оценка"
	@echo "  make exp02       — Perplexity Sonar DR:    генерация + оценка"
	@echo "  make exp03       — Gemini 2.5 Pro:         генерация + оценка"
	@echo "  make exp04       — SurveyForge (GPU):      генерация + оценка"
	@echo ""
	@echo "  [Данные]"
	@echo "  make download    — скачать датасет U4R/SurveyBench"
	@echo "  make sfdb        — скачать БД SurveyForge (~6-10 GB)"
	@echo "  make sfdb-check  — проверить файлы БД SurveyForge"
	@echo "  make sfmodel     — скачать embedding-модель gte-large-en-v1.5 (~1.7 GB)"
	@echo ""
	@echo "  [Утилиты]"
	@echo "  make models      — список моделей на локальном сервере"
	@echo "  make models-ping — список моделей + пинг каждой"
	@echo "  make clean       — удалить Docker-образы проекта"
	@echo ""

## ── Локальное окружение ───────────────────────────────────────────────────────

# Для exp01-03 и локальных утилит (лёгкие зависимости)
setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip -q
	.venv/bin/pip install -r requirements.txt

# Для exp04 SurveyForge — устанавливает тяжёлые зависимости (torch, faiss, etc.)
# Запускай ПОСЛЕ make setup
setup-sf:
	.venv/bin/pip install -r experiments/exp04_surveyforge/requirements.txt

## ── Docker ───────────────────────────────────────────────────────────────────

# Базовый образ для exp01-03
# --no-cache гарантирует чистую сборку без устаревших pip-слоёв
base:
	docker build --no-cache -f experiments/Dockerfile.base -t thesis-base .

## exp01 — OpenAI Deep Research
exp01: base
	docker build -f experiments/exp01_openai_dr/Dockerfile -t thesis-exp01 .
	docker run --rm $(EXP_VOLUMES) thesis-exp01

## exp02 — Perplexity Sonar Deep Research
exp02: base
	docker build -f experiments/exp02_perplexity_dr/Dockerfile -t thesis-exp02 .
	docker run --rm $(EXP_VOLUMES) thesis-exp02

## exp03 — Gemini 2.5 Pro
exp03: base
	docker build -f experiments/exp03_gemini_pro/Dockerfile -t thesis-exp03 .
	docker run --rm $(EXP_VOLUMES) thesis-exp03

## exp04 — SurveyForge (GPU, отдельный образ на pytorch/pytorch base)
# HF кэш монтируется отдельно чтобы не скачивать модель каждый раз
exp04:
	docker build -f experiments/exp04_surveyforge/Dockerfile -t thesis-exp04 .
	docker run --rm --gpus all $(EXP_VOLUMES) \
		-v "$(PWD)/datasets/hf_cache:/root/.cache/huggingface" \
		thesis-exp04

# CPU-вариант exp04 (медленно, только для отладки)
exp04-cpu:
	docker build -f experiments/exp04_surveyforge/Dockerfile -t thesis-exp04 .
	docker run --rm $(EXP_VOLUMES) \
		-v "$(PWD)/datasets/hf_cache:/root/.cache/huggingface" \
		thesis-exp04

## ── Данные ───────────────────────────────────────────────────────────────────

download:
	$(PYTHON) src/download.py

inspect:
	$(PYTHON) src/download.py --inspect

sfdb:
	$(PYTHON) src/utils/download_surveyforge_db.py

sfdb-check:
	$(PYTHON) src/utils/download_surveyforge_db.py --check

# Скачать embedding-модель для SurveyForge (~1.7 GB)
sfmodel:
	$(PYTHON) src/utils/download_sf_model.py

## ── Утилиты ──────────────────────────────────────────────────────────────────

models:
	$(PYTHON) src/utils/check_local.py

models-ping:
	$(PYTHON) src/utils/check_local.py --ping

## ── Очистка ──────────────────────────────────────────────────────────────────

clean:
	-docker rmi thesis-base thesis-exp01 thesis-exp02 thesis-exp03 thesis-exp04 2>/dev/null || true
