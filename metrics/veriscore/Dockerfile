FROM thesis-base
WORKDIR /app

COPY src/               src/
# metrics/utils.py is imported by main.py (TokenCounter, llm helpers, etc.).
# Without this COPY Python finds only metrics/veriscore/ inside the image and
# `from metrics.utils import ...` fails with ModuleNotFoundError.
COPY metrics/utils.py   metrics/utils.py
COPY metrics/veriscore/ metrics/veriscore/
COPY repos/VeriScore/   repos/VeriScore/

CMD ["python", "metrics/veriscore/main.py"]
