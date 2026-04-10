# models/surveyforge/Dockerfile
#
# Сборка: из корня репозитория
#   docker build -f models/surveyforge/Dockerfile -t thesis-gen-surveyforge .
#
# Запуск:
#   make generate-sf-cpu DATASET=SurGE
#
# ВАЖНО: SurveyForge database.py использует .to(torch.device('cuda')).
# Для CPU-запуска замени на .to(torch.device('cpu')) в:
#   repos/SurveyForge/code/src/database.py

FROM thesis-base

# SurveyForge-специфичные зависимости (langchain, faiss-cpu, tinydb и т.д.)
COPY models/surveyforge/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Исходный код
COPY src/                ./src/
COPY repos/SurveyForge/  ./repos/SurveyForge/
COPY models/surveyforge/ ./models/surveyforge/

# Пустые точки монтирования (данные приходят через volumes)
RUN mkdir -p datasets/surveyforge_db datasets/gte-large-en-v1.5 \
             datasets/SurGE results

ENTRYPOINT ["python", "models/surveyforge/main.py"]
CMD ["--dataset", "SurGE"]
