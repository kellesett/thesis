# experiments/exp03_gemini_pro/Dockerfile
# Сборка: из корня репозитория
#   docker build -f experiments/exp03_gemini_pro/Dockerfile -t thesis-exp03 .
# Запуск:
#   docker run --rm --env-file .env -v $(pwd)/results:/app/results thesis-exp03

FROM thesis-base

COPY experiments/exp03_gemini_pro/requirements.txt ./exp_requirements.txt
RUN pip install --no-cache-dir -r exp_requirements.txt

COPY experiments/exp03_gemini_pro/ ./experiments/exp03_gemini_pro/

CMD ["python", "experiments/exp03_gemini_pro/main.py"]
