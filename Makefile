# Makefile
# Запускай из корня репозитория.
# Предварительно: cp .env.example .env && заполни ключи

.PHONY: help setup setup-sf base \
        generate evaluate validate \
        convert-autosurvey surge-reference \
        viewer enrich \
        download inspect models models-ping \
        sfdb sfdb-check sfmodel \
        download-metric-models clean

PYTHON  = .venv/bin/python
DATASET ?= SurGE
MODEL   ?= perplexity_dr
METRIC  ?= surge
GPU     ?= 0

# SurGE_reference pipeline knobs (used by `make surge-reference`):
#   MODE=string|llm|hybrid  (default hybrid — string top-K + LLM re-ranker)
#   LIMIT=N                 (default unset — process all surveys)
MODE     ?= hybrid
LIMIT_FLAG = $(if $(LIMIT),--limit $(LIMIT),)

# Shared volumes for all containers
VOLUMES = --env-file .env \
          -v "$(PWD)/tmp:/tmp" \
          -v "$(PWD)/repos:/app/repos" \
          -v "$(PWD)/datasets:/app/datasets" \
          -v "$(PWD)/results:/app/results" \
          -v "$(PWD)/models_cache:/app/models_cache"
# results/logs/ is inside the results volume, so logs survive container crashes

_GPU_FLAG = $(if $(filter 1,$(GPU)),--gpus all,)


## Справка
help:
	@echo ""
	@echo "  [Окружение]"
	@echo "  make setup          — .venv + requirements.txt"
	@echo "  make setup-sf       — .venv + requirements exp04 (torch, faiss, ...)"
	@echo ""
	@echo "  [Docker — base]"
	@echo "  make base              — собрать thesis-base (кэшируется)"
	@echo "  make base NO_CACHE=1   — пересобрать thesis-base без кэша"
	@echo ""
	@echo "  [Генерация]"
	@echo "  make generate MODEL=perplexity_dr  [DATASET=SurGE]"
	@echo "  make generate MODEL=surveyforge    [DATASET=SurGE] [GPU=1]"
	@echo "  make generate MODEL=surveygen_i   [DATASET=SurGE]"
	@echo "  make convert-autosurvey           [DATASET=SurGE]  — baseline (локально, без Docker)"
	@echo "  make surge-reference              [MODE=hybrid] [LIMIT=N] — human-written SurGE surveys → generation format"
	@echo ""
	@echo "  [Оценка]"
	@echo "  make evaluate DATASET=SurGE MODEL=perplexity_dr METRIC=surge"
	@echo "  make evaluate DATASET=SurGE MODEL=perplexity_dr METRIC=claimify     # извлечение claims (3-stage, нужен для factuality + expert)"
	@echo "  make evaluate DATASET=SurGE MODEL=perplexity_dr METRIC=veriscore   # извлечение claims (альтернатива claimify, быстрее)"
	@echo "  make evaluate DATASET=SurGE MODEL=perplexity_dr METRIC=diversity   # семантическое разнообразие"
	@echo "  make evaluate DATASET=SurGE MODEL=perplexity_dr METRIC=structural  # A.1 M_contr, A.2 M_term, A.3 M_rep"
	@echo "  make evaluate DATASET=SurGE MODEL=perplexity_dr METRIC=factuality  # B.1 CitCorrect_k"
	@echo "  make evaluate DATASET=SurGE MODEL=perplexity_dr METRIC=expert      # C.1-C.4 M_crit/comp/open/mod"
	@echo ""
	@echo "  [Валидация промптов]"
	@echo "  make validate METRIC=expert       # precision/recall на expert_classes_test + expert_modalities_test"
	@echo "  make validate METRIC=factuality   # precision/recall на factuality_classes_test"
	@echo ""
	@echo "  [Данные / Модели]"
	@echo "  make download / sfdb / sfdb-check / sfmodel"
	@echo "  make download-metric-models    — скачать NER/NLI/AlignScore модели в models_cache/"
	@echo ""
	@echo "  [Утилиты]"
	@echo "  make enrich / models / models-ping / clean"
	@echo ""

## ── Локальное окружение ───────────────────────────────────────────────────────

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip -q
	.venv/bin/pip install -r requirements.txt

setup-sf:
	.venv/bin/pip install -r experiments/exp04_surveyforge/requirements.txt

## ── Docker base ──────────────────────────────────────────────────────────────

# Собирает базовый образ (пропускает если уже существует)
NO_CACHE ?= 0

base:
	docker build $(if $(filter 1,$(NO_CACHE)),--no-cache,) -f docker/Dockerfile.base -t thesis-base .

## ── Генерация ────────────────────────────────────────────────────────────────

generate: base
	mkdir -p tmp models_cache results/logs
	docker build -f models/$(MODEL)/Dockerfile -t thesis-gen-$(MODEL) .
	docker run --rm $(_GPU_FLAG) $(VOLUMES) thesis-gen-$(MODEL) \
		python models/$(MODEL)/main.py --dataset $(DATASET)

## ── AutoSurvey (конвертация baseline) ───────────────────────────────────────

# Запускается локально (не в Docker — данные уже на диске, не нужен API)
convert-autosurvey:
	$(PYTHON) models/autosurvey/main.py --dataset $(DATASET)

## ── SurGE_reference — human-written surveys → generation format ─────────────
# End-to-end: fetch arxiv sources → merge LaTeX → match vs Semantic Scholar → assemble.
# Each step is resume-friendly (skips completed work). Run individual stages
# directly with python3 if you want tighter control — see README.
#
# Examples:
#   make surge-reference                         # all surveys, hybrid mode
#   make surge-reference MODE=string LIMIT=40    # first 40, string-only (no LLM cost)
#   make surge-reference MODE=hybrid LIMIT=40    # first 40, hybrid (default)
surge-reference:
	$(PYTHON) scripts/fetch_reference_latex.py $(LIMIT_FLAG)
	bash scripts/merge_latex.sh
	$(PYTHON) scripts/match_ss_to_bibitems.py --mode $(MODE) $(LIMIT_FLAG) --top-k 5 --parallel 50
	$(PYTHON) scripts/build_surge_reference.py --mode $(MODE) $(LIMIT_FLAG)
	$(PYTHON) scripts/enrich_arxiv_ids.py $(LIMIT_FLAG)

## ── Оценка ───────────────────────────────────────────────────────────────────
# Промежуточные кэши всех метрик живут в ./tmp/, которая монтируется в /tmp.
# Структура внутри: /tmp/claimify/<run_id>/, /tmp/veriscore/<run_id>/, ...
# Чтобы кэшировать HuggingFace-модели (diversity/structural/...) добавь:
#   -v "$(HOME)/.cache/huggingface:/root/.cache/huggingface"

evaluate: base
	mkdir -p tmp models_cache results/logs
	docker build -f metrics/$(METRIC)/Dockerfile -t thesis-eval-$(METRIC) .
	docker run --rm $(VOLUMES) thesis-eval-$(METRIC) \
		python metrics/$(METRIC)/main.py --dataset $(DATASET) --model $(MODEL) $(LIMIT_FLAG)

validate: base
	mkdir -p tmp
	docker build -f metrics/$(METRIC)/Dockerfile -t thesis-eval-$(METRIC) .
	docker run --rm $(VOLUMES) thesis-eval-$(METRIC) \
		python metrics/$(METRIC)/validate.py

## ── Viewer ───────────────────────────────────────────────────────────────────

viewer:
	$(PYTHON) -m streamlit run app/main.py

## ── Обогащение ссылок ────────────────────────────────────────────────────────

enrich:
	$(PYTHON) scripts/enrich_references.py --dir $(DATASET)_$(MODEL)

## ── Данные ───────────────────────────────────────────────────────────────────

download:
	$(PYTHON) src/download.py

inspect:
	$(PYTHON) src/download.py --inspect

sfdb:
	$(PYTHON) src/utils/download_surveyforge_db.py

sfdb-check:
	$(PYTHON) src/utils/download_surveyforge_db.py --check

sfmodel:
	$(PYTHON) src/utils/download_sf_model.py

## ── Утилиты ──────────────────────────────────────────────────────────────────

models:
	$(PYTHON) src/utils/check_local.py

models-ping:
	$(PYTHON) src/utils/check_local.py --ping

## ── Загрузка моделей для метрик ──────────────────────────────────────────────

download-metric-models:
	$(PYTHON) scripts/download_metric_models.py

## ── Очистка ──────────────────────────────────────────────────────────────────

clean:
	-docker rmi thesis-base thesis-gen-perplexity_dr thesis-gen-surveyforge \
	    thesis-gen-surveygen_i \
	    thesis-eval-surge thesis-eval-diversity \
	    thesis-eval-claimify thesis-eval-structural \
	    thesis-eval-factuality thesis-eval-expert \
	    thesis-eval-veriscore 2>/dev/null || true
