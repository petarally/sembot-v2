"""Sva konfiguracija na jednom mjestu (čita se iz env varijabli)."""

import os

# --- Modeli -----------------------------------------------------------------
# Bi-encoder za semantičko pretraživanje. multilingual-e5-base dobro pokriva
# hrvatski; E5 modeli traže prefikse "query: " / "passage: ".
EMBED_MODEL = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-base")
QUERY_PREFIX = os.environ.get("QUERY_PREFIX", "query: ")
PASSAGE_PREFIX = os.environ.get("PASSAGE_PREFIX", "passage: ")

# Cross-encoder reranker (multijezičan). Skuplji, ali precizniji; rerankira
# samo top-k kandidata, nikad cijelu bazu.
USE_RERANKER = os.environ.get("USE_RERANKER", "true").lower() == "true"
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# --- Pretraga ---------------------------------------------------------------
# Koliko kandidata bi-encoder prosljeđuje rerankeru.
TOP_K = int(os.environ.get("TOP_K", 15))

# --- Pragovi odluke ---------------------------------------------------------
# POČETNE vrijednosti; OBAVEZNO kalibrirati na testnom skupu stvarnih upita.
#
# Ako je najbolja cosine sličnost ispod ovoga -> pitanje je izvan domene.
MIN_RETRIEVAL_SCORE = float(os.environ.get("MIN_RETRIEVAL_SCORE", 0.80))
# Reranker (sigmoid) prag ispod kojeg se suzdržavamo od odgovora.
RERANK_THRESHOLD = float(os.environ.get("RERANK_THRESHOLD", 0.50))
# Iznad ovoga smo "sigurni" i ne tražimo marginu.
RERANK_HIGH_CONF = float(os.environ.get("RERANK_HIGH_CONF", 0.80))
# Minimalna razlika top-1 i top-2 da bismo odgovorili kad nismo "sigurni".
RERANK_MARGIN = float(os.environ.get("RERANK_MARGIN", 0.15))
