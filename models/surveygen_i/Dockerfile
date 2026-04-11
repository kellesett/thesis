FROM thesis-base

WORKDIR /app

# SurveyGen-I specific dependencies (torch already in base, faiss-cpu replaces faiss-gpu)
COPY models/surveygen_i/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY src/ ./src/
COPY models/surveygen_i/ ./models/surveygen_i/
# Copy SurveyGen-I repo (prompts, acm template, pipeline code)
COPY repos/SurveyGen-I/ ./repos/SurveyGen-I/

# results/ and datasets/ монтируются через volume при запуске
RUN mkdir -p results

CMD ["python", "models/surveygen_i/main.py"]
