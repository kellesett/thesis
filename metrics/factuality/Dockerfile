FROM thesis-base
WORKDIR /app

COPY metrics/factuality/ metrics/factuality/
COPY src/ src/

CMD ["python", "metrics/factuality/main.py"]
