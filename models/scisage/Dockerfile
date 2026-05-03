FROM thesis-base

WORKDIR /app

COPY models/scisage/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/scisage/ ./models/scisage/
COPY repos/SciSage/ ./repos/SciSage/

RUN mkdir -p tmp results

CMD ["python", "models/scisage/main.py", "--dataset", "SurGE"]
