FROM thesis-base
WORKDIR /app

COPY metrics/ metrics/
COPY src/ src/

RUN pip install --no-cache-dir -r metrics/structural/requirements.txt

CMD ["python", "metrics/structural/main.py"]
