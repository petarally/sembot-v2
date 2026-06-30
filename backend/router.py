"""Orkestracija chatbota: pretraga -> rerank -> odluka -> odgovor."""

from __future__ import annotations

import numpy as np

from .config import (
    MIN_MARGIN,
    MIN_RETRIEVAL_SCORE,
    RERANK_HIGH_CONF,
    RERANK_MARGIN,
    RERANK_THRESHOLD,
    TOP_K,
)
from .reranker import Reranker
from .search import Candidate, SemanticIndex
from .store import get_store
from .suggestions import SuggestionEngine


class ChatbotRouter:
    def __init__(self):
        self.store = get_store()
        self.index = SemanticIndex(self.store.load())
        self.reranker = Reranker()
        self.suggestions = SuggestionEngine(self.index)
        print(f"Router spreman: {len(self.qa_pairs)} QA parova.")

    @property
    def qa_pairs(self) -> list[dict]:
        """Aktualni QA parovi (indeks je izvor istine u memoriji)."""
        return self.index.qa_pairs

    # ----------------------------------------------------------- admin izmjene
    def add_qa(self, question: str, answer: str, paraphrases: list[str] | None = None) -> dict:
        """Dodaj pitanje: spremi u store + inkrementalno u indeks (bez restarta)."""
        item = self.store.add(
            {"question": question, "answer": answer, "paraphrases": paraphrases or []}
        )
        self.index.add_qa(item)
        return item

    def add_qa_bulk(self, items: list[dict]) -> list[dict]:
        """Dodaj više pitanja odjednom: jedan upis u store + jedno enkodiranje."""
        new_items = self.store.add_many(items)
        self.index.add_many(new_items)
        return new_items

    def delete_qa(self, qa_id: str) -> bool:
        """Obriši pitanje iz storea i ponovno izgradi indeks iz svježeg stanja."""
        if not self.store.delete(qa_id):
            return False
        self.index.rebuild(self.store.load())
        return True

    def list_qa(self) -> list[dict]:
        return self.qa_pairs

    async def get_response(self, query: str) -> dict:
        print(f"Obrada upita: '{query}'")
        return self.get_response_sync(query)

    def get_response_sync(self, query: str) -> dict:
        qa_index = self.resolve(query)
        return self._answer_for(qa_index) if qa_index is not None else self._fallback()

    def resolve(self, query: str) -> int | None:
        """Vrati qa_index odabranog odgovora ili None (suzdržavanje). Cijeli pipeline."""
        candidates = self.index.search(query, TOP_K)
        # Stage 1: očito izvan domene -> suzdrži se.
        if not candidates or candidates[0].score < MIN_RETRIEVAL_SCORE:
            return None
        # Stage 2: reranker (ako je dostupan) ili margina za provjeru sigurnosti.
        chosen = self._decide(query, candidates)
        return chosen.qa_index if chosen is not None else None

    def _decide(self, query: str, candidates: list[Candidate]) -> Candidate | None:
        """Odaberi najboljeg kandidata ili None ako nismo dovoljno sigurni."""
        if not self.reranker.available:
            # Ako je drugi kandidat preblizu prvom, model nije siguran -> suzdrži se.
            if len(candidates) > 1 and (candidates[0].score - candidates[1].score) < MIN_MARGIN:
                return None
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

    def _answer_for(self, qa_index: int) -> dict:
        qa = self.qa_pairs[qa_index]
        return {
            "text": qa["answer"],
            "suggested_questions": self.suggestions.next_questions(qa_index),
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
