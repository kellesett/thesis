# Makefile
# Запускай из корня репозитория.
# Предварительно: cp .env.example .env && заполни ключи

.PHONY: help setup setup-sf setup-scisage base \
        generate evaluate validate \
        convert-autosurvey surge-reference \
        viewer enrich \
        download inspect models models-ping \
        push-results pull-results pull-new-results \
        sfdb sfdb-check sfmodel \
        download-metric-models clean

PYTHON  = .venv/bin/python
DATASET ?= SurGE
MODEL   ?= perplexity_dr
METRIC  ?= surge
GPU     ?= 0

# DOCKER=1 (default) — собирать и запускать в контейнере.
# DOCKER=0           — запускать напрямую через .venv/bin/python (bare-metal).
#                      Для bare-metal нужно: `make setup` (полный — с torch,
#                      AlignScore, spaCy/NLTK данными), `.env` в корне репо.
DOCKER  ?= 1

# SurGE_reference pipeline knobs (used by `make surge-reference`):
#   MODE=string|llm|hybrid  (default hybrid — string top-K + LLM re-ranker)
#   LIMIT=N                 (default unset — process all surveys)
MODE     ?= hybrid
LIMIT_FLAG = $(if $(LIMIT),--limit $(LIMIT),)

# EXTRA_ARGS — generic pass-through for `make evaluate`. Any flags the
# underlying metric main.py accepts can be supplied here, e.g.:
#   EXTRA_ARGS="--debug-claim-idx 3 --evidence-aggregation per_ref"
# Unset by default. Useful for metric-specific knobs that aren't worth
# dedicated Makefile variables (factuality's --claim-scope,
# --evidence-source, etc.).
EXTRA_ARGS ?=

# Remote sync over plain ssh+tar (useful when the server has no rsync).
# SYNC_CONFLICT=overwrite — target files are overwritten.
# SYNC_CONFLICT=fail      — extraction uses tar -k and fails on existing files.
# SYNC_CONFLICT=skip      — pull only: keep existing local files, extract missing files.
REMOTE ?= admin@188.44.57.13
REMOTE_REPO ?= ~/dolgushevgleb/thesis
SYNC_CONFLICT ?= overwrite
ifneq ($(filter $(SYNC_CONFLICT),overwrite fail skip),$(SYNC_CONFLICT))
$(error SYNC_CONFLICT must be overwrite, fail, or skip)
endif
TAR_EXTRACT_FLAGS = $(if $(filter fail,$(SYNC_CONFLICT)),-xzkf,-xzf)

# Shared volumes for all containers
VOLUMES = --env-file .env \
          -e FACTUALITY_CHECKPOINT_ROOT=/tmp/factuality \
          -v "$(PWD)/tmp:/tmp" \
          -v "$(PWD)/tmp:/app/tmp" \
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
	@echo "  make setup          — .venv + requirements.txt + spaCy/NLTK данные + AlignScore patch"
	@echo "  make setup-sf       — .venv + requirements exp04 (torch, faiss, ...)"
	@echo "  make setup-scisage  — зависимости SciSage поверх текущего .venv"
	@echo ""
	@echo "  [Docker — base]"
	@echo "  make base              — собрать thesis-base (кэшируется)"
	@echo "  make base NO_CACHE=1   — пересобрать thesis-base без кэша"
	@echo ""
	@echo "  Любой target ниже принимает DOCKER=0 → запуск в bare-metal venv,"
	@echo "  DOCKER=1 (default) → запуск в контейнере."
	@echo ""
	@echo "  [Генерация]"
	@echo "  make generate MODEL=perplexity_dr  [DATASET=SurGE]"
	@echo "  make generate MODEL=surveyforge    [DATASET=SurGE] [GPU=1]"
	@echo "  make generate MODEL=surveygen_i   [DATASET=SurGE]"
	@echo "  make generate MODEL=scisage       [DATASET=SurGE]"
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
	@echo "  (add LIMIT=N to cap to survey_id<=N, EXTRA_ARGS='--flag val' for metric-specific knobs)"
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
	@echo "  make push-results [REMOTE=...] [REMOTE_REPO=...] [SYNC_CONFLICT=overwrite|fail]"
	@echo "  make pull-results [REMOTE=...] [REMOTE_REPO=...] [SYNC_CONFLICT=overwrite|fail|skip]"
	@echo "  make pull-new-results [REMOTE=...] [REMOTE_REPO=...]"
	@echo ""

## ── Локальное окружение ───────────────────────────────────────────────────────

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip -q
	.venv/bin/pip install -r requirements.txt
	# spaCy / NLTK данные нужны claimify, veriscore, factuality (sent_tokenize в AlignScore).
	.venv/bin/python -m spacy download en_core_web_sm
	.venv/bin/python -m nltk.downloader -d $(HOME)/nltk_data punkt_tab punkt
	# AlignScore (для factuality) ставится отдельно из repos/ — патч под
	# новую transformers/lightning применяется идемпотентно. Если репо
	# AlignScore ещё не клонировано — шаги тихо скипаются.
	@if [ -d repos/AlignScore ]; then \
	    .venv/bin/pip install --no-deps -e repos/AlignScore ; \
	    .venv/bin/python scripts/patch_alignscore.py ; \
	else \
	    echo "repos/AlignScore/ не найден — пропустил установку (factuality потребует его)" ; \
	fi

setup-sf:
	.venv/bin/pip install -r experiments/exp04_surveyforge/requirements.txt

setup-scisage:
	$(PYTHON) -m pip install -r models/scisage/requirements.txt

## ── Docker base ──────────────────────────────────────────────────────────────

# Собирает базовый образ (пропускает если уже существует)
NO_CACHE ?= 0

base:
	docker build $(if $(filter 1,$(NO_CACHE)),--no-cache,) -f docker/Dockerfile.base -t thesis-base .

## ── Генерация ────────────────────────────────────────────────────────────────
# DOCKER=1 (default) → собирает образ и запускает в контейнере.
# DOCKER=0           → исполняет напрямую через $(PYTHON) (bare-metal venv).
#                      `.env` подгружается inline через `set -a; . .env; set +a`.

generate: $(if $(filter 1,$(DOCKER)),base,)
	mkdir -p tmp models_cache results/logs
ifeq ($(DOCKER),0)
	@set -a; [ -f .env ] && . ./.env; set +a; \
	    $(PYTHON) models/$(MODEL)/main.py --dataset $(DATASET) $(LIMIT_FLAG)
else
	docker build -f models/$(MODEL)/Dockerfile -t thesis-gen-$(MODEL) .
	docker run --rm $(_GPU_FLAG) $(VOLUMES) thesis-gen-$(MODEL) \
		python models/$(MODEL)/main.py --dataset $(DATASET) $(LIMIT_FLAG)
endif

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

evaluate: $(if $(filter 1,$(DOCKER)),base,)
	mkdir -p tmp models_cache results/logs
ifeq ($(DOCKER),0)
	@set -a; [ -f .env ] && . ./.env; set +a; \
	    $(PYTHON) metrics/$(METRIC)/main.py \
	        --dataset $(DATASET) --model $(MODEL) \
	        $(LIMIT_FLAG) $(EXTRA_ARGS)
else
	docker build -f metrics/$(METRIC)/Dockerfile -t thesis-eval-$(METRIC) .
	# -t allocates a pseudo-TTY inside the container so tqdm can do in-place
	# bar redraws via \r and ANSI cursor codes. Without it every update
	# appends a new line and a two-bar display explodes on screen.
	# TTY detection lives in the recipe shell — not in a make $$(shell ..)
	# expansion, since there fd 1 is a pipe to make and the test would
	# always fail. The backslash-continued line keeps both assignment and
	# docker run in one shell so $$TTY propagates; falls back to empty
	# string under CI/nohup.
	TTY="$$(test -t 1 && echo -t || true)" ; \
	docker run --rm $$TTY $(VOLUMES) thesis-eval-$(METRIC) \
		python metrics/$(METRIC)/main.py --dataset $(DATASET) --model $(MODEL) $(LIMIT_FLAG) $(EXTRA_ARGS)
endif

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

## ── Синхронизация results без rsync ─────────────────────────────────────────

push-results:
	@echo "local results/ -> $(REMOTE):$(REMOTE_REPO)/results/  (conflict=$(SYNC_CONFLICT))"
	@if [ "$(SYNC_CONFLICT)" = "skip" ]; then \
	    echo "SYNC_CONFLICT=skip is supported for pull-results only; use overwrite or fail for push-results."; \
	    exit 2; \
	fi
	tar -C results -czf - . | ssh $(REMOTE) 'mkdir -p $(REMOTE_REPO)/results && tar -C $(REMOTE_REPO)/results $(TAR_EXTRACT_FLAGS) -'

pull-results:
	@echo "$(REMOTE):$(REMOTE_REPO)/results/ -> local results/  (conflict=$(SYNC_CONFLICT))"
	mkdir -p results
ifeq ($(SYNC_CONFLICT),skip)
	ssh $(REMOTE) 'cd $(REMOTE_REPO)/results && tar -czf - .' | $(PYTHON) scripts/extract_new_from_tar.py --dest results
else
	ssh $(REMOTE) 'cd $(REMOTE_REPO)/results && tar -czf - .' | tar -C results $(TAR_EXTRACT_FLAGS) -
endif

pull-new-results:
	$(MAKE) pull-results SYNC_CONFLICT=skip

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
