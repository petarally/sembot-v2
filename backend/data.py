"""Učitavanje i validacija pitanja i odgovora iz lokalne datoteke."""

from __future__ import annotations

import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
QA_PATH = os.path.join(DATA_DIR, "qa_data.json")


def load_qa_pairs() -> list[dict]:
    """Vrati listu QA parova. Svaki par: {"question", "answer", "paraphrases"?}."""
    print(f"Učitavanje podataka iz: {QA_PATH}")
    with open(QA_PATH, "r", encoding="utf-8") as file:
        qa_pairs = json.load(file)["qa_pairs"]
    validate_qa_pairs(qa_pairs)
    print(f"Učitano {len(qa_pairs)} pitanja i odgovora.")
    return qa_pairs


def validate_qa_pairs(qa_pairs: list[dict]) -> None:
    """Glasno padni na neispravnom unosu, imenujući problem (umjesto kasnijeg KeyError)."""
    if not isinstance(qa_pairs, list) or not qa_pairs:
        raise ValueError("qa_data.json: 'qa_pairs' mora biti neprazna lista.")

    seen: dict[str, int] = {}
    for i, qa in enumerate(qa_pairs):
        where = f"unos #{i}"
        if not isinstance(qa, dict):
            raise ValueError(f"{where}: mora biti objekt, a nije.")

        question, answer = qa.get("question"), qa.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{where}: nedostaje ili je prazan 'question'.")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"{where} ('{question}'): nedostaje ili je prazan 'answer'.")

        paraphrases = qa.get("paraphrases", [])
        if not isinstance(paraphrases, list) or any(not isinstance(p, str) for p in paraphrases):
            raise ValueError(f"{where} ('{question}'): 'paraphrases' mora biti lista stringova.")

        if question in seen:
            raise ValueError(f"{where}: duplikat pitanja (već u unosu #{seen[question]}): '{question}'")
        seen[question] = i
