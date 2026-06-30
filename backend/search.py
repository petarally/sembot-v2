"""Bi-encoder indeks: enkodira pitanja jednom, kešira ih i traži najsličnija."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBED_MODEL, PASSAGE_PREFIX, QUERY_PREFIX
from .data import DATA_DIR

_CACHE_PATH = os.path.join(DATA_DIR, ".embeddings_cache.npz")


@dataclass
class Candidate:
    """Jedan rezultat pretrage: indeks QA para i njegova cosine sličnost."""

    qa_index: int
    score: float


class SemanticIndex:
    def __init__(self, qa_pairs: list[dict]):
        self.qa_pairs = qa_pairs
        print(f"Učitavanje embedding modela: {EMBED_MODEL}")
        self.encoder = SentenceTransformer(EMBED_MODEL)
        # Svako pitanje može imati više formulacija (parafraza) radi boljeg recalla.
        self.utterances, self.utterance_to_qa = self._collect_utterances(qa_pairs)
        self.embeddings = self._build_or_load_embeddings()

    @staticmethod
    def _collect_utterances(qa_pairs: list[dict]) -> tuple[list[str], list[int]]:
        """Spljošti sva pitanja + parafraze u jednu listu uz mapu na QA par."""
        utterances, utterance_to_qa = [], []
        for qa_index, qa in enumerate(qa_pairs):
            seen = set()
            for variant in [qa["question"], *qa.get("paraphrases", [])]:
                if variant and variant not in seen:
                    seen.add(variant)
                    utterances.append(variant)
                    utterance_to_qa.append(qa_index)
        return utterances, utterance_to_qa

    def _config_hash(self) -> str:
        """Hash podataka + konfiguracije; promijeni li se išta, cache se odbacuje."""
        payload = json.dumps(
            {
                "utterances": self.utterances,
                "model": EMBED_MODEL,
                "query_prefix": QUERY_PREFIX,
                "passage_prefix": PASSAGE_PREFIX,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _build_or_load_embeddings(self) -> np.ndarray:
        current_hash = self._config_hash()
        if os.path.exists(_CACHE_PATH):
            try:
                cached = np.load(_CACHE_PATH, allow_pickle=False)
                if str(cached["meta"][0]) == current_hash:
                    print("Embeddingi učitani iz cachea.")
                    return cached["embeddings"]
            except Exception as e:
                print(f"Cache nečitljiv ({e}); ponovno enkodiram.")

        print(f"Enkodiranje {len(self.utterances)} formulacija (jednokratno)...")
        passages = [PASSAGE_PREFIX + u for u in self.utterances]
        embeddings = self.encoder.encode(
            passages, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)
        try:
            np.savez(_CACHE_PATH, embeddings=embeddings, meta=np.array([current_hash]))
            print(f"Embeddingi keširani u {_CACHE_PATH}")
        except Exception as e:
            print(f"Spremanje cachea nije uspjelo ({e}); nastavljam bez cachea.")
        return embeddings

    def search(self, query: str, top_k: int) -> list[Candidate]:
        """Vrati do top_k jedinstvenih QA parova, poredanih po cosine sličnosti."""
        query_vec = self.encoder.encode(
            QUERY_PREFIX + query, normalize_embeddings=True
        ).astype(np.float32)
        sims = self.embeddings @ query_vec  # normalizirani vektori -> dot = cosine

        candidates, seen = [], set()
        for utterance_index in np.argsort(-sims):
            qa_index = self.utterance_to_qa[int(utterance_index)]
            if qa_index in seen:
                continue
            seen.add(qa_index)
            candidates.append(Candidate(qa_index, float(sims[int(utterance_index)])))
            if len(candidates) >= top_k:
                break
        return candidates
