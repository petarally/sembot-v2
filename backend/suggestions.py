"""Prijedlozi sljedećih pitanja na temelju semantičke sličnosti.

Umjesto ručnih kategorija i ključnih riječi, koristi embeddinge koje indeks
ionako računa: predlaže najsličnija *druga* pitanja matchanom. Skalira se samo
s rastom baze i ne treba održavanje.
"""

from __future__ import annotations

from .search import SemanticIndex

# Pričuvna pitanja kad baza ima premalo srodnih unosa.
_GENERAL_QUESTIONS = [
    "Koje je službeno ime Sveučilišta u Puli?",
    "Kako mogu kontaktirati studentsku službu?",
    "Gdje mogu pronaći raspored predavanja?",
    "Koji su rokovi za prijavu ispita?",
    "Koje fakultete ima Sveučilište Jurja Dobrile u Puli?",
]


class SuggestionEngine:
    def __init__(self, qa_pairs: list[dict], index: SemanticIndex):
        self.qa_pairs = qa_pairs
        self.index = index

    def next_questions(self, qa_index: int, k: int = 3) -> list[str]:
        """Do k prijedloga: najsličnija pitanja, pa pričuvna ako ih nema dovoljno."""
        current = self.qa_pairs[qa_index]["question"]
        suggested = [self.qa_pairs[i]["question"] for i in self.index.similar_questions(qa_index, k)]

        for question in _GENERAL_QUESTIONS:
            if len(suggested) >= k:
                break
            if question != current and question not in suggested:
                suggested.append(question)

        return suggested[:k]
