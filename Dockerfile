# Lean image za Cloud Run, optimiziran za brz cold start:
# modeli i embedding cache se "zapeku" u image (build-time), pa se pri pokretanju
# ništa ne preuzima s interneta — samo učita s lokalnog diska.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models

WORKDIR /app

# CPU-only torch (puno manji od CUDA builda) prije ostalih ovisnosti.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend

# Zapeci modele u image (preuzmu se sad, ne pri svakom cold startu).
RUN python -c "import os; from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer(os.environ.get('EMBED_MODEL','intfloat/multilingual-e5-base')); \
    CrossEncoder(os.environ.get('RERANKER_MODEL','BAAI/bge-reranker-v2-m3'))"

# Zapeci i embedding cache (.npz) da se rute ne enkodiraju pri prvom upitu.
RUN python -c "from backend.search import SemanticIndex; from backend.data import load_qa_pairs; \
    SemanticIndex(load_qa_pairs())"

# Pri pokretanju nikad ne pokušavaj online dohvat modela.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Cloud Run prosljeđuje PORT (default 8080).
CMD exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8080}
