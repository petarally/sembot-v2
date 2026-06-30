"""Učitavanje pitanja i odgovora iz lokalne datoteke."""

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
    print(f"Učitano {len(qa_pairs)} pitanja i odgovora.")
    return qa_pairs
