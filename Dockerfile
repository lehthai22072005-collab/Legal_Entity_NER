FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    USE_TF=0 \
    USE_FLAX=0 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/tmp/huggingface \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 ner

COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN python -m pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r /tmp/requirements-docker.txt \
    && python -m pip check

COPY app.py /app/app.py
COPY src/ /app/src/
COPY scripts/predict.py scripts/smoke_test.py /app/scripts/
COPY configs/ontology.json /app/configs/ontology.json
COPY resources/phobert-base/ /app/resources/phobert-base/
COPY checkpoints/phobert_span_all369_v1/ /app/checkpoints/phobert_span_all369_v1/
COPY reports/phobert_span_all369_v1/ACCEPTANCE_SNAPSHOT.json /app/reports/phobert_span_all369_v1/ACCEPTANCE_SNAPSHOT.json

# Refuse a missing/LFS-pointer/changed model before producing a runnable image.
RUN python -c "import hashlib,json,pathlib; r=pathlib.Path('/app'); s=json.loads((r/'reports/phobert_span_all369_v1/ACCEPTANCE_SNAPSHOT.json').read_text()); f=(r/'checkpoints/phobert_span_all369_v1/model.pt').open('rb'); assert hashlib.file_digest(f,'sha256').hexdigest()==s['model_sha256'], 'Checkpoint missing or SHA-256 mismatch'; print('CHECKPOINT_EPOCH_11_SHA256_PASS')"

USER ner
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).read()"
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--server.fileWatcherType=none", "--server.maxUploadSize=25", "--browser.gatherUsageStats=false"]
