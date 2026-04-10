# models/perplexity_dr/Dockerfile
# Сборка: из корня репозитория
#   docker build -f models/perplexity_dr/Dockerfile -t thesis-gen-perplexity_dr .
# Запуск:
#   docker run --rm --env-file .env -v $(pwd)/results:/app/results thesis-gen-perplexity_dr

FROM thesis-base

COPY src/     ./src/
COPY models/perplexity_dr/ ./models/perplexity_dr/

CMD ["python", "models/perplexity_dr/main.py", "--dataset", "SurGE"]
