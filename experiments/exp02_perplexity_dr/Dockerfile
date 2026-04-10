# experiments/exp02_perplexity_dr/Dockerfile
# Сборка: из корня репозитория
#   docker build -f experiments/exp02_perplexity_dr/Dockerfile -t thesis-exp02 .
# Запуск:
#   docker run --rm --env-file .env -v $(pwd)/results:/app/results thesis-exp02

FROM thesis-base

# gcc нужен для сборки C-расширений транзитивных зависимостей FlagEmbedding (zlib-state и др.)
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY experiments/exp02_perplexity_dr/requirements.txt ./exp_requirements.txt
RUN pip install --no-cache-dir -r exp_requirements.txt

COPY src/     ./src/
COPY configs/ ./configs/
COPY repos/SurveyForge/SurveyBench/ ./repos/SurveyForge/SurveyBench/
COPY repos/SurGE/src/               ./repos/SurGE/src/
COPY experiments/exp02_perplexity_dr/ ./experiments/exp02_perplexity_dr/

CMD ["python", "experiments/exp02_perplexity_dr/main.py"]
