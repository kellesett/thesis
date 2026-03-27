# experiments/exp04_surveyforge/Dockerfile
# Образ для запуска SurveyForge (GPU рекомендован).
#
# Сборка: из корня репозитория
#   docker build -f experiments/exp04_surveyforge/Dockerfile -t thesis-exp04 .
#
# Запуск (GPU):
#   docker run --rm --gpus all $(EXP_VOLUMES) \
#     -v "$(PWD)/datasets/hf_cache:/root/.cache/huggingface" \
#     thesis-exp04
#
# Запуск (CPU, только для отладки — embedding будет медленным):
#   docker run --rm $(EXP_VOLUMES) \
#     -v "$(PWD)/datasets/hf_cache:/root/.cache/huggingface" \
#     thesis-exp04
#
# ВАЖНО: SurveyForge в database.py использует .to(torch.device('cuda')).
# Для CPU-запуска замени на .to(torch.device('cpu')) в:
#   repos/SurveyForge/code/src/database.py

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# Python зависимости SurveyForge
COPY experiments/exp04_surveyforge/requirements.txt ./exp_requirements.txt
RUN pip install --no-cache-dir -r exp_requirements.txt

# Исходный код
COPY src/          ./src/
COPY configs/      ./configs/
COPY repos/        ./repos/

# Пустые папки (данные монтируются снаружи)
RUN mkdir -p datasets results eval_results

# Переменные окружения
ENV OPENROUTER_API_KEY=""
ENV HF_TOKEN=""
ENV TRANSFORMERS_CACHE="/root/.cache/huggingface"
ENV PYTHONUNBUFFERED=1

COPY experiments/exp04_surveyforge/ ./experiments/exp04_surveyforge/

CMD ["python", "experiments/exp04_surveyforge/main.py"]
