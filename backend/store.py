"""Pohrana QA parova: lokalni JSON (razvoj) ili Firestore (produkcija).

Svaki par ima stabilni `id` (za brisanje), `question`, `answer` i opcionalno
`paraphrases`. Store je *izvor istine*; indeks i embeddingi se izvode iz njega.
"""

from __future__ import annotations

import json
import os
import uuid

from .config import FIRESTORE_COLLECTION, QA_STORE
from .data import QA_PATH, validate_qa_pairs


def _normalize(item: dict) -> dict:
    """Osiguraj id i paraphrases polja."""
    return {
        "id": item.get("id") or uuid.uuid4().hex,
        "question": item["question"],
        "answer": item["answer"],
        "paraphrases": item.get("paraphrases", []),
    }


class JsonStore:
    """Lokalna qa_data.json datoteka. Za razvoj; nije trajna na Cloud Run/Railway."""

    def load(self) -> list[dict]:
        with open(QA_PATH, "r", encoding="utf-8") as f:
            pairs = [_normalize(p) for p in json.load(f)["qa_pairs"]]
        validate_qa_pairs(pairs)
        return pairs

    def add(self, item: dict) -> dict:
        return self.add_many([item])[0]

    def add_many(self, items: list[dict]) -> list[dict]:
        pairs = self.load()
        news = [_normalize(i) for i in items]
        validate_qa_pairs(pairs + news)  # padne na duplikatu/neispravnom unosu (cijeli batch)
        pairs.extend(news)
        self._save(pairs)
        return news

    def delete(self, qa_id: str) -> bool:
        pairs = self.load()
        kept = [p for p in pairs if p["id"] != qa_id]
        if len(kept) == len(pairs):
            return False
        self._save(kept)
        return True

    @staticmethod
    def _save(pairs: list[dict]) -> None:
        with open(QA_PATH, "w", encoding="utf-8") as f:
            json.dump({"qa_pairs": pairs}, f, ensure_ascii=False, indent=2)


class FirestoreStore:
    """Firebase Firestore kolekcija. Trajna; preživljava restart i redeploy."""

    def __init__(self):
        from google.cloud import firestore

        self._db = firestore.Client()
        self._col = self._db.collection(FIRESTORE_COLLECTION)

    def load(self) -> list[dict]:
        pairs = []
        for doc in self._col.stream():
            data = doc.to_dict()
            pairs.append(_normalize({**data, "id": doc.id}))
        validate_qa_pairs(pairs)
        return pairs

    def add(self, item: dict) -> dict:
        return self.add_many([item])[0]

    def add_many(self, items: list[dict]) -> list[dict]:
        news = [_normalize(i) for i in items]
        validate_qa_pairs(self.load() + news)  # padne na duplikatu/neispravnom unosu (cijeli batch)
        # Firestore batch ima limit od 500 operacija po commitu.
        for start in range(0, len(news), 500):
            batch = self._db.batch()
            for n in news[start:start + 500]:
                batch.set(
                    self._col.document(n["id"]),
                    {"question": n["question"], "answer": n["answer"], "paraphrases": n["paraphrases"]},
                )
            batch.commit()
        return news

    def delete(self, qa_id: str) -> bool:
        ref = self._col.document(qa_id)
        if not ref.get().exists:
            return False
        ref.delete()
        return True


def get_store():
    """Odaberi store prema QA_STORE; padni na JSON ako Firestore nije dostupan."""
    if QA_STORE == "firestore":
        try:
            store = FirestoreStore()
            print(f"Pohrana: Firestore (kolekcija '{FIRESTORE_COLLECTION}').")
            return store
        except Exception as e:
            print(f"Firestore nedostupan ({e}); koristim lokalni JSON.")
    print("Pohrana: lokalni JSON.")
    return JsonStore()
