# metrics/surge/Dockerfile
# Сборка: из корня репозитория
#   docker build -f metrics/surge/Dockerfile -t thesis-eval-surge .
# Запуск:
#   docker run --rm --env-file .env \
#     -v $(pwd)/results:/app/results \
#     -v $(pwd)/datasets:/app/datasets \
#     thesis-eval-surge --dataset SurGE --model perplexity_dr

FROM thesis-base

COPY src/              ./src/
COPY repos/SurGE/src/  ./repos/SurGE/src/
COPY metrics/surge/    ./metrics/surge/

CMD ["python", "metrics/surge/main.py"]
