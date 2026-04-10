# experiments/exp01_openai_dr/Dockerfile
# Сборка: из корня репозитория
#   docker build -f experiments/exp01_openai_dr/Dockerfile -t thesis-exp01 .
# Запуск:
#   docker run --rm --env-file .env -v $(pwd)/results:/app/results thesis-exp01

FROM thesis-base

COPY experiments/exp01_openai_dr/requirements.txt ./exp_requirements.txt
RUN pip install --no-cache-dir -r exp_requirements.txt

COPY src/     ./src/
COPY configs/ ./configs/
COPY repos/SurveyForge/SurveyBench/ ./repos/SurveyForge/SurveyBench/
COPY experiments/exp01_openai_dr/   ./experiments/exp01_openai_dr/

CMD ["python", "experiments/exp01_openai_dr/main.py"]
