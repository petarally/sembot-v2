"""Bi-encoder indeks: enkodira pitanja jednom, kešira ih i traži najsličnija."""

from __future__ import annotations

import hashlib
import json
import os
import threading
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
        print(f"Učitavanje embedding modela: {EMBED_MODEL}")
        self.encoder = SentenceTransformer(EMBED_MODEL)
        self._lock = threading.RLock()  # štiti izmjene indeksa za vrijeme pretrage
        self._build(qa_pairs)

    def _build(self, qa_pairs: list[dict]) -> None:
        """(Pre)izgradi cijeli indeks iz danih QA parova."""
        self.qa_pairs = qa_pairs
        # Svako pitanje može imati više formulacija (parafraza) radi boljeg recalla.
        self.utterances, self.utterance_to_qa = self._collect_utterances(qa_pairs)
        self.embeddings = self._build_or_load_embeddings()
        # Redak embeddinga primarnog pitanja za svaki QA par (prva formulacija).
        self.qa_question_rows = self._index_primary_questions()

    @staticmethod
    def _qa_utterances(qa: dict) -> list[str]:
        """Deduplicirane formulacije jednog QA para: [pitanje, *parafraze]."""
        utterances, seen = [], set()
        for variant in [qa["question"], *qa.get("paraphrases", [])]:
            if variant and variant not in seen:
                seen.add(variant)
                utterances.append(variant)
        return utterances

    @classmethod
    def _collect_utterances(cls, qa_pairs: list[dict]) -> tuple[list[str], list[int]]:
        """Spljošti sva pitanja + parafraze u jednu listu uz mapu na QA par."""
        utterances, utterance_to_qa = [], []
        for qa_index, qa in enumerate(qa_pairs):
            for utterance in cls._qa_utterances(qa):
                utterances.append(utterance)
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

    def _index_primary_questions(self) -> np.ndarray:
        """Za svaki QA par zapamti redak embeddinga njegovog primarnog pitanja."""
        rows = [-1] * len(self.qa_pairs)
        for row, qa_index in enumerate(self.utterance_to_qa):
            if rows[qa_index] == -1:  # prva formulacija = samo pitanje
                rows[qa_index] = row
        return np.array(rows)

    @property
    def question_embeddings(self) -> np.ndarray:
        """Matrica embeddinga primarnih pitanja, poredana po qa_index."""
        return self.embeddings[self.qa_question_rows]

    # ----------------------------------------------------------- izmjene uživo
    def add_qa(self, qa: dict) -> None:
        """Dodaj jedan QA par, inkrementalno (bez rebuilda)."""
        self.add_many([qa])

    def add_many(self, qa_list: list[dict]) -> None:
        """Dodaj više QA parova odjednom; enkodira cijeli batch jednim pozivom."""
        if not qa_list:
            return
        with self._lock:
            new_utterances, new_map, primary_rows = [], [], []
            for qa in qa_list:
                qa_index = len(self.qa_pairs)
                self.qa_pairs.append(qa)
                utterances = self._qa_utterances(qa)
                # primarno pitanje = prva formulacija ovog para
                primary_rows.append(len(self.utterances) + len(new_utterances))
                new_utterances.extend(utterances)
                new_map.extend([qa_index] * len(utterances))

            passages = [PASSAGE_PREFIX + u for u in new_utterances]
            new_emb = np.atleast_2d(
                self.encoder.encode(passages, normalize_embeddings=True)
            ).astype(np.float32)

            self.utterances.extend(new_utterances)
            self.utterance_to_qa.extend(new_map)
            self.embeddings = np.vstack([self.embeddings, new_emb])
            self.qa_question_rows = np.append(self.qa_question_rows, primary_rows)

    def rebuild(self, qa_pairs: list[dict]) -> None:
        """Ponovno izgradi cijeli indeks (npr. nakon brisanja)."""
        with self._lock:
            self._build(qa_pairs)

    # --------------------------------------------------------------- čitanje
    def similar_questions(self, qa_index: int, k: int) -> list[int]:
        """Vrati k qa_indexa najsličnijih danom pitanju (bez njega samog)."""
        with self._lock:
            target = self.embeddings[self.qa_question_rows[qa_index]]
            sims = self.question_embeddings @ target
            order = np.argsort(-sims)
            return [int(i) for i in order if int(i) != qa_index][:k]

    def search(self, query: str, top_k: int) -> list[Candidate]:
        """Vrati do top_k jedinstvenih QA parova, poredanih po cosine sličnosti."""
        query_vec = self.encoder.encode(
            QUERY_PREFIX + query, normalize_embeddings=True
        ).astype(np.float32)

        with self._lock:
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
