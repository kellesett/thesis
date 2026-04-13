FROM thesis-base
WORKDIR /app

COPY src/               src/
COPY metrics/veriscore/ metrics/veriscore/
COPY repos/VeriScore/   repos/VeriScore/

CMD ["python", "metrics/veriscore/main.py"]
