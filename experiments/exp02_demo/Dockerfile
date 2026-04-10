# experiments/exp02_demo/Dockerfile
# Демо-версия exp02 без torch/FlagEmbedding — считаем только Structure_Quality и Logic.
# Сборка: из корня репозитория
#   docker build -f experiments/exp02_demo/Dockerfile -t thesis-exp02-demo .
# Запуск:
#   docker run --rm --env-file .env -v $(pwd)/results:/app/results thesis-exp02-demo

FROM thesis-base

COPY experiments/exp02_demo/requirements.txt ./exp_requirements.txt
RUN pip install --no-cache-dir -r exp_requirements.txt

COPY src/     ./src/
COPY configs/ ./configs/
COPY repos/SurveyForge/SurveyBench/ ./repos/SurveyForge/SurveyBench/
COPY repos/SurGE/src/               ./repos/SurGE/src/
COPY experiments/exp02_demo/        ./experiments/exp02_demo/

CMD ["python", "experiments/exp02_demo/main.py"]
