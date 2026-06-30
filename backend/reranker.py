"""Cross-encoder reranker: precizno ocjenjuje relevantnost upita za kandidate."""

from __future__ import annotations

import threading

import numpy as np

from .config import RERANKER_MODEL, USE_RERANKER


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class Reranker:
    """Wrapper oko CrossEncodera.

    Model je velik (~2.3 GB), pa se učitava u pozadinskoj niti kako ne bi
    blokirao start aplikacije. Dok se ne učita, `available` je False i router
    se oslanja samo na bi-encoder (brži, malo manje precizan). Bitno za Cloud
    Run cold start: prvi korisnik dobije odgovor odmah, bez čekanja na model.
    """

    def __init__(self):
        self.model = None
        if USE_RERANKER:
            threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            from sentence_transformers import CrossEncoder

            print(f"Učitavanje reranker modela u pozadini: {RERANKER_MODEL}")
            model = CrossEncoder(RERANKER_MODEL)
            self.model = model  # tek sad postaje "available" (atomarno)
            print("Reranker spreman.")
        except Exception as e:
            print(f"Reranker se nije učitao ({e}); nastavljam samo s bi-encoderom.")

    @property
    def available(self) -> bool:
        return self.model is not None

    def score(self, query: str, questions: list[str]) -> np.ndarray:
        """Vrati sigmoid relevantnosti (0–1) za svako pitanje, poravnato po redu."""
        pairs = [[query, question] for question in questions]
        return _sigmoid(np.array(self.model.predict(pairs)))
