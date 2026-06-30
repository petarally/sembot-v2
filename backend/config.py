"""Sva konfiguracija na jednom mjestu (čita se iz env varijabli)."""

import os

# --- Modeli -----------------------------------------------------------------
# Bi-encoder za semantičko pretraživanje. e5-base je odabran mjerenjem (bolji od
# e5-small: 91% vs 84% hit@1, manje samouvjereno krivih). e5-small je lakša
# alternativa, e5-large skuplja nadogradnja. E5 modeli traže "query: "/"passage: ".
EMBED_MODEL = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-base")
QUERY_PREFIX = os.environ.get("QUERY_PREFIX", "query: ")
PASSAGE_PREFIX = os.environ.get("PASSAGE_PREFIX", "passage: ")

# Cross-encoder reranker (multijezičan). Precizniji, ali velik i spor; ISKLJUČEN
# po defaultu. Uključi (USE_RERANKER=true) tek ako mjerenje pokaže da pomaže.
USE_RERANKER = os.environ.get("USE_RERANKER", "false").lower() == "true"
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# --- Pretraga ---------------------------------------------------------------
# Koliko kandidata bi-encoder prosljeđuje rerankeru.
TOP_K = int(os.environ.get("TOP_K", 8))

# --- Pragovi odluke ---------------------------------------------------------
# Kalibrirano na testnom skupu (backend/eval_data.json) za EMBED_MODEL e5-base.
# PRAG OVISI O MODELU — promijeniš li EMBED_MODEL, ponovno pokreni evaluaciju
# (`python -m backend.evaluate`) i prilagodi ovu vrijednost.
#
# Ako je najbolja cosine sličnost ispod ovoga -> pitanje je izvan domene.
MIN_RETRIEVAL_SCORE = float(os.environ.get("MIN_RETRIEVAL_SCORE", 0.82))
# Minimalna razlika top-1 i top-2 (cosine) u bi-encoder putanji. Ako su preblizu,
# model nije siguran -> radije se suzdrži nego pogađa. Kalibrirano evaluacijom.
MIN_MARGIN = float(os.environ.get("MIN_MARGIN", 0.01))
# Reranker (sigmoid) prag ispod kojeg se suzdržavamo od odgovora.
RERANK_THRESHOLD = float(os.environ.get("RERANK_THRESHOLD", 0.50))
# Iznad ovoga smo "sigurni" i ne tražimo marginu.
RERANK_HIGH_CONF = float(os.environ.get("RERANK_HIGH_CONF", 0.80))
# Minimalna razlika top-1 i top-2 da bismo odgovorili kad nismo "sigurni".
RERANK_MARGIN = float(os.environ.get("RERANK_MARGIN", 0.15))

# --- Pohrana podataka i admin ----------------------------------------------
# "json" = lokalna datoteka (razvoj); "firestore" = Firebase Firestore (produkcija).
QA_STORE = os.environ.get("QA_STORE", "json")
FIRESTORE_COLLECTION = os.environ.get("FIRESTORE_COLLECTION", "qa_pairs")
# Token za admin endpointe. Prazno => admin je isključen (sigurnosni default).
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")
