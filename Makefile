# Makefile
# Запускай из корня репозитория.
# Предварительно: cp .env.example .env && заполни ключи

.PHONY: help base exp01 exp02 exp03 download inspect clean

EXP_VOLUMES = --env-file .env \
              -v "$(PWD)/results:/app/results" \
              -v "$(PWD)/eval_results:/app/eval_results" \
              -v "$(PWD)/datasets:/app/datasets"

## Справка
help:
	@echo ""
	@echo "  make base     — собрать базовый Docker-образ (thesis-base)"
	@echo "  make exp01    — OpenAI Deep Research: генерация + оценка"
	@echo "  make exp02    — Perplexity Sonar DR:  генерация + оценка"
	@echo "  make exp03    — Gemini 2.5 Pro:       генерация + оценка"
	@echo "  make download — скачать датасет с HuggingFace (локально)"
	@echo "  make inspect  — посмотреть структуру датасета"
	@echo "  make clean    — удалить Docker-образы проекта"
	@echo ""

## Базовый образ — пересобирать при изменениях в src/ или requirements.txt
base:
	docker build -f experiments/Dockerfile.base -t thesis-base .

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

## Датасет
download:
	.venv/bin/python src/download.py

inspect:
	.venv/bin/python src/download.py --inspect

## Очистка
clean:
	-docker rmi thesis-base thesis-exp01 thesis-exp02 thesis-exp03 2>/dev/null || true
