"""Cross-encoder reranker: precizno ocjenjuje relevantnost upita za kandidate."""

from __future__ import annotations

import numpy as np

from .config import RERANKER_MODEL, USE_RERANKER


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class Reranker:
    """Wrapper oko CrossEncodera. Ako se model ne učita, `available` je False."""

    def __init__(self):
        self.model = None
        if not USE_RERANKER:
            return
        try:
            from sentence_transformers import CrossEncoder

            print(f"Učitavanje reranker modela: {RERANKER_MODEL}")
            self.model = CrossEncoder(RERANKER_MODEL)
        except Exception as e:
            print(f"Reranker se nije učitao ({e}); nastavljam samo s bi-encoderom.")

    @property
    def available(self) -> bool:
        return self.model is not None

    def score(self, query: str, questions: list[str]) -> np.ndarray:
        """Vrati sigmoid relevantnosti (0–1) za svako pitanje, poravnato po redu."""
        pairs = [[query, question] for question in questions]
        return _sigmoid(np.array(self.model.predict(pairs)))
