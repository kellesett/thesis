FROM thesis-base
WORKDIR /app

COPY metrics/ metrics/
COPY src/ src/
# evidence_fetcher.py reuses helpers from scripts/ (fetch_with_retry,
# _clean_latex_fragment) — pull them into the image. Pure Python, no new
# system deps (pandoc isn't needed because we only use string helpers).
COPY scripts/ scripts/

CMD ["python", "metrics/factuality/main.py"]
