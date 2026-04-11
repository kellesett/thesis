# Makefile
# Запускай из корня репозитория.
# Предварительно: cp .env.example .env && заполни ключи

.PHONY: help setup setup-sf base \
        generate evaluate \
        generate-sf generate-sf-cpu generate-sgi \
        convert-autosurvey \
        evaluate-diversity \
        viewer enrich \
        download inspect models models-ping \
        sfdb sfdb-check sfmodel \
        download-metric-models clean

PYTHON     = .venv/bin/python
DATASET   ?= SurGE
MODEL     ?= perplexity_dr
METRIC    ?= surge

VOLUMES = --env-file .env \
          -v "$(PWD)/results:/app/results" \
          -v "$(PWD)/datasets:/app/datasets"

# Метрики групп A/B/C дополнительно монтируют локальные модели
METRIC_VOLUMES = $(VOLUMES) \
          -v "$(PWD)/models_cache:/app/models_cache"


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
	@echo "  make generate           [DATASET=SurGE] [MODEL=perplexity_dr]"
	@echo "  make generate-sf        [DATASET=SurGE]  — SurveyForge (GPU)"
	@echo "  make generate-sf-cpu    [DATASET=SurGE]  — SurveyForge (CPU, debug)"
	@echo "  make generate-sgi       [DATASET=SurGE]  — SurveyGen-I (CPU)"
	@echo "  make convert-autosurvey [DATASET=SurGE]  — конвертировать baseline AutoSurvey"
	@echo ""
	@echo "  [Оценка]"
	@echo "  make evaluate DATASET=SurGE MODEL=perplexity_dr METRIC=surge"
	@echo "  make evaluate DATASET=SurGE MODEL=perplexity_dr METRIC=claimify     # кэш claims (нужен для factuality + expert)"
	@echo "  make evaluate DATASET=SurGE MODEL=perplexity_dr METRIC=structural   # A.1 M_contr, A.2 M_term, A.3 M_rep"
	@echo "  make evaluate DATASET=SurGE MODEL=perplexity_dr METRIC=factuality   # B.1 CitCorrect_k"
	@echo "  make evaluate DATASET=SurGE MODEL=perplexity_dr METRIC=expert       # C.1-C.4 M_crit/comp/open/mod"
	@echo "  make evaluate-diversity [DATASET=SurGE] [MODEL=perplexity_dr]"
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
	docker build -f models/$(MODEL)/Dockerfile -t thesis-gen-$(MODEL) .
	docker run --rm $(VOLUMES) thesis-gen-$(MODEL) \
		python models/$(MODEL)/main.py --dataset $(DATASET)

## ── SurveyForge (GPU) ────────────────────────────────────────────────────────

SF_VOLUMES = --env-file .env \
             -v "$(PWD)/datasets:/app/datasets" \
             -v "$(PWD)/results:/app/results"

generate-sf:
	docker build -f models/surveyforge/Dockerfile -t thesis-gen-surveyforge .
	docker run --rm --gpus all $(SF_VOLUMES) thesis-gen-surveyforge --dataset $(DATASET)

# CPU-вариант (медленно, только для отладки)
generate-sf-cpu:
	docker build -f models/surveyforge/Dockerfile -t thesis-gen-surveyforge .
	docker run --rm $(SF_VOLUMES) thesis-gen-surveyforge --dataset $(DATASET)

## ── SurveyGen-I (CPU) ────────────────────────────────────────────────────────

generate-sgi: base
	docker build -f models/surveygen_i/Dockerfile -t thesis-gen-surveygen_i .
	docker run --rm $(VOLUMES) thesis-gen-surveygen_i \
		python models/surveygen_i/main.py --dataset $(DATASET)

## ── AutoSurvey (конвертация baseline) ───────────────────────────────────────

# Запускается локально (не в Docker — данные уже на диске, не нужен API)
convert-autosurvey:
	$(PYTHON) models/autosurvey/main.py --dataset $(DATASET)

## ── Оценка (универсальный target) ───────────────────────────────────────────
# Метрики surge/diversity используют VOLUMES (без models_cache).
# Метрики claimify/structural/factuality/expert используют METRIC_VOLUMES.
_METRIC_NEEDS_MODELS = claimify structural factuality expert
_USE_METRIC_VOLUMES  = $(filter $(METRIC),$(_METRIC_NEEDS_MODELS))

evaluate: base
	docker build -f metrics/$(METRIC)/Dockerfile -t thesis-eval-$(METRIC) .
	$(if $(_USE_METRIC_VOLUMES), \
		docker run --rm $(METRIC_VOLUMES) thesis-eval-$(METRIC) \
			python metrics/$(METRIC)/main.py --dataset $(DATASET) --model $(MODEL), \
		docker run --rm $(VOLUMES) thesis-eval-$(METRIC) \
			python metrics/$(METRIC)/main.py --dataset $(DATASET) --model $(MODEL))

## ── Оценка (diversity) ───────────────────────────────────────────────────────

# Опционально: кэш HuggingFace чтобы не качать SPECTER2 каждый раз:
#   DIVERSITY_VOLUMES += -v "$(HOME)/.cache/huggingface:/root/.cache/huggingface"

evaluate-diversity:
	docker build -f metrics/diversity/Dockerfile -t thesis-eval-diversity .
	docker run --rm $(VOLUMES) thesis-eval-diversity \
		--dataset $(DATASET) --model $(MODEL)

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
	    thesis-eval-factuality thesis-eval-expert 2>/dev/null || true
