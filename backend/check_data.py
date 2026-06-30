"""Offline provjera kvalitete podataka.

Validira qa_data.json (kroz load_qa_pairs) i prijavljuje parove pitanja s vrlo
visokom semantičkom sličnošću — vjerojatne duplikate ili konflikte koje treba
očistiti prije nego baza naraste.

Pokretanje:  python -m backend.check_data [prag]
"""

from __future__ import annotations

import sys

from .data import load_qa_pairs
from .search import SemanticIndex

DEFAULT_THRESHOLD = 0.90


def main() -> int:
    threshold = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_THRESHOLD

    qa_pairs = load_qa_pairs()  # validacija se izvrši ovdje
    index = SemanticIndex(qa_pairs)

    questions = index.question_embeddings
    sims = questions @ questions.T

    flagged = []
    for i in range(len(qa_pairs)):
        for j in range(i + 1, len(qa_pairs)):
            if sims[i, j] >= threshold:
                flagged.append((float(sims[i, j]), i, j))
    flagged.sort(reverse=True)

    if not flagged:
        print(f"OK: nema parova pitanja sa sličnošću >= {threshold}.")
        return 0

    print(f"Sumnjivo slični parovi (>= {threshold}) — mogući duplikati/konflikti:\n")
    for score, i, j in flagged:
        print(f"  [{score:.3f}]")
        print(f"    #{i}: {qa_pairs[i]['question']}")
        print(f"    #{j}: {qa_pairs[j]['question']}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
