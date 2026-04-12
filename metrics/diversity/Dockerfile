# metrics/diversity/Dockerfile
#
# Сборка: из корня репозитория
#   docker build -f metrics/diversity/Dockerfile -t thesis-eval-diversity .
#
# Запуск:
#   docker run --rm --env-file .env \
#     -v $(pwd)/results:/app/results \
#     -v $(pwd)/datasets/SurGE/corpus.json:/app/datasets/SurGE/corpus.json:ro \
#     thesis-eval-diversity --dataset SurGE --model perplexity_dr
#
# Кэш HuggingFace (опционально, чтобы не качать SPECTER2 каждый раз):
#   -v $(HOME)/.cache/huggingface:/root/.cache/huggingface

FROM thesis-base

COPY metrics/diversity/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/                ./src/
COPY metrics/diversity/  ./metrics/diversity/

ENV TRANSFORMERS_CACHE="/root/.cache/huggingface"

CMD ["python", "metrics/diversity/main.py", "--dataset", "SurGE", "--model", "perplexity_dr"]
