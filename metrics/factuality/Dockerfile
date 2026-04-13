FROM thesis-base
WORKDIR /app

COPY metrics/ metrics/
COPY src/ src/

CMD ["python", "metrics/factuality/main.py"]
