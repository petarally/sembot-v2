"""Evaluacija bi-encoder retrievala na ručno označenom testnom skupu.

Mjeri točnost dohvata i ponašanje suzdržavanja, te predlaže prag MIN_RETRIEVAL_SCORE.
Pokretanje:  python -m backend.evaluate

Format eval_data.json: [{"query": "...", "expected": "<točan tekst pitanja>" | null}]
gdje null znači da bot TREBA reći "ne znam" (izvan domene).
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from .config import MIN_MARGIN, MIN_RETRIEVAL_SCORE, TOP_K, USE_RERANKER
from .router import ChatbotRouter

EVAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_data.json")


def main() -> None:
    tests = json.load(open(EVAL_PATH, encoding="utf-8"))
    router = ChatbotRouter()

    # Reranker se učitava u pozadini — pričekaj ga prije mjerenja.
    if USE_RERANKER:
        print("Čekam da se reranker učita...")
        for _ in range(600):
            if router.reranker.available:
                break
            time.sleep(1)
        print(f"Reranker dostupan: {router.reranker.available}")

    qa_index = {qa["question"]: i for i, qa in enumerate(router.qa_pairs)}

    # Za svaki upit: očekivani indeks, STVARNI odabir pipelinea (resolve), te
    # bi-encoder dijagnostika (top-1/score/margina/rang).
    rows = []
    for t in tests:
        expected = t.get("expected")
        exp_idx = qa_index.get(expected) if expected else None
        if expected and exp_idx is None:
            print(f"  ! expected nije u bazi: {expected!r}")
        resolved = router.resolve(t["query"])
        cands = router.index.search(t["query"], TOP_K)
        rank = next((r for r, c in enumerate(cands) if c.qa_index == exp_idx), None)
        margin = cands[0].score - cands[1].score if len(cands) > 1 else 1.0
        rows.append({"q": t["query"], "exp": exp_idx, "resolved": resolved,
                     "top1": cands[0].qa_index, "score": cands[0].score,
                     "margin": margin, "rank": rank})

    in_rows = [r for r in rows if r["exp"] is not None]
    ood_rows = [r for r in rows if r["exp"] is None]

    # --- Kvaliteta dohvata (neovisno o pragu) ---
    hit1 = sum(r["rank"] == 0 for r in in_rows)
    hit3 = sum(r["rank"] is not None and r["rank"] < 3 for r in in_rows)
    print("\n=== Kvaliteta dohvata (in-domain, bez praga) ===")
    print(f"  hit@1: {hit1}/{len(in_rows)}   hit@3: {hit3}/{len(in_rows)}")

    # --- STVARNI pipeline (odražava EMBED_MODEL + USE_RERANKER + pragove) ---
    rr = "uključen" if USE_RERANKER else "isključen"
    correct = sum(r["resolved"] == r["exp"] for r in in_rows)
    wrong = sum(r["resolved"] is not None and r["resolved"] != r["exp"] for r in in_rows)
    abstain = sum(r["resolved"] is None for r in in_rows)
    ood_ok = sum(r["resolved"] is None for r in ood_rows)
    print(f"\n=== Stvarni pipeline (reranker {rr}) ===")
    print(f"  in-domain:  točno {correct}/{len(in_rows)}  | KRIVO {wrong}  | suzdržan {abstain}")
    print(f"  izvan teme: ispravno suzdržan {ood_ok}/{len(ood_rows)}")
    bad = [r for r in in_rows if r["resolved"] is not None and r["resolved"] != r["exp"]]
    if bad:
        print("  Samouvjereno krivi (pipeline je odgovorio, ali pogrešno):")
        for r in bad:
            print(f"    '{r['q']}'\n      dobio: {router.qa_pairs[r['resolved']]['question']}")

    if USE_RERANKER:
        return  # cosine sweepovi nemaju smisla kad reranker odlučuje

    # --- Sweep praga (margina fiksna na trenutnoj) ---
    print(f"\n=== Sweep praga (margina={MIN_MARGIN:.2f}) | ✓točno ✗krivo –suzdržan | OOD✓ ===")
    print(f"  {'prag':>5} {'točno':>6} {'krivo':>6} {'suzdr':>6} {'OOD✓':>6}")
    for thr in np.round(np.arange(0.74, 0.90, 0.01), 2):
        s = _stats(thr, MIN_MARGIN, in_rows, ood_rows)
        print(f"  {thr:>5.2f} {s['correct']:>6} {s['wrong']:>6} {s['abstain']:>6} {s['ood_ok']:>6}")

    # --- Sweep margine (prag fiksan) — tražimo gdje krivi padnu na ~0 ---
    print(f"\n=== Sweep margine (prag={MIN_RETRIEVAL_SCORE:.2f}) | ✓točno ✗krivo –suzdržan | OOD✓ ===")
    print(f"  {'marg':>5} {'točno':>6} {'krivo':>6} {'suzdr':>6} {'OOD✓':>6}")
    for m in np.round(np.arange(0.0, 0.10, 0.01), 2):
        s = _stats(MIN_RETRIEVAL_SCORE, m, in_rows, ood_rows)
        print(f"  {m:>5.2f} {s['correct']:>6} {s['wrong']:>6} {s['abstain']:>6} {s['ood_ok']:>6}")
    print("\n  Cilj 'sve sigurno točno': najmanja margina gdje 'krivo' = 0 (uz što više 'točno').")
    print("  'suzdržan' nije greška — to su pitanja na koja bot sigurno odgovori 'pitaj referadu'.")


def _answered(r, thr, margin_min):
    return r["score"] >= thr and r["margin"] >= margin_min


def _stats(thr, margin_min, in_rows, ood_rows):
    correct = sum(_answered(r, thr, margin_min) and r["top1"] == r["exp"] for r in in_rows)
    wrong = sum(_answered(r, thr, margin_min) and r["top1"] != r["exp"] for r in in_rows)
    abstain = sum(not _answered(r, thr, margin_min) for r in in_rows)
    ood_ok = sum(not _answered(r, thr, margin_min) for r in ood_rows)
    return {"correct": correct, "wrong": wrong, "abstain": abstain, "ood_ok": ood_ok}


def _report(label, thr, margin_min, in_rows, ood_rows):
    s = _stats(thr, margin_min, in_rows, ood_rows)
    print(f"\n=== {label}: prag={thr:.2f}, margina={margin_min:.2f} ===")
    print(f"  in-domain:  točno {s['correct']}/{len(in_rows)}  | "
          f"KRIVO {s['wrong']}  | suzdržan {s['abstain']}")
    print(f"  izvan teme: ispravno suzdržan {s['ood_ok']}/{len(ood_rows)}")


if __name__ == "__main__":
    main()
