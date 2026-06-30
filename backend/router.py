"""Orkestracija chatbota: pretraga -> rerank -> odluka -> odgovor."""

from __future__ import annotations

import numpy as np

from .config import (
    MIN_RETRIEVAL_SCORE,
    RERANK_HIGH_CONF,
    RERANK_MARGIN,
    RERANK_THRESHOLD,
    TOP_K,
)
from .data import load_qa_pairs
from .reranker import Reranker
from .search import Candidate, SemanticIndex
from .suggestions import SuggestionEngine


class ChatbotRouter:
    def __init__(self):
        self.qa_pairs = load_qa_pairs()
        self.index = SemanticIndex(self.qa_pairs)
        self.reranker = Reranker()
        self.suggestions = SuggestionEngine(self.qa_pairs, self.index)
        print(f"Router spreman: {len(self.qa_pairs)} QA parova.")

    async def get_response(self, query: str) -> dict:
        print(f"Obrada upita: '{query}'")
        return self.get_response_sync(query)

    def get_response_sync(self, query: str) -> dict:
        candidates = self.index.search(query, TOP_K)

        # Stage 1: očito izvan domene -> suzdrži se.
        if not candidates or candidates[0].score < MIN_RETRIEVAL_SCORE:
            return self._fallback()

        # Stage 2: reranker za precizan poredak i provjeru sigurnosti.
        chosen = self._decide(query, candidates)
        return self._answer_for(chosen) if chosen is not None else self._fallback()

    def _decide(self, query: str, candidates: list[Candidate]) -> Candidate | None:
        """Odaberi najboljeg kandidata ili None ako nismo dovoljno sigurni."""
        if not self.reranker.available:
            return candidates[0]  # već je prošao MIN_RETRIEVAL_SCORE

        questions = [self.qa_pairs[c.qa_index]["question"] for c in candidates]
        scores = self.reranker.score(query, questions)
        order = np.argsort(-scores)

        best = float(scores[order[0]])
        second = float(scores[order[1]]) if len(order) > 1 else 0.0

        confident = best >= RERANK_HIGH_CONF
        decent = best >= RERANK_THRESHOLD and (best - second) >= RERANK_MARGIN
        if confident or decent:
            return candidates[int(order[0])]
        return None

    def _answer_for(self, candidate: Candidate) -> dict:
        qa = self.qa_pairs[candidate.qa_index]
        return {
            "text": qa["answer"],
            "suggested_questions": self.suggestions.next_questions(candidate.qa_index),
        }

    def _fallback(self) -> dict:
        """Odgovor kad nismo sigurni: preusmjeri na studentsku službu."""
        return {
            "text": (
                "Žao mi je, na ovo ne mogu pouzdano odgovoriti. Za točne informacije "
                "kontaktirajte studentsku službu na ured-za-studente@unipu.hr ili "
                "telefonom na 052/377-006."
            ),
            "suggested_questions": [
                "Kako mogu kontaktirati studentsku službu?",
                "Gdje mogu pronaći akademski kalendar?",
                "Koji su rokovi za prijavu ispita?",
            ],
        }
