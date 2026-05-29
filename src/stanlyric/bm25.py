"""A compact BM25 implementation for StanLyric."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class BM25Config:
    k1: float = 1.5
    b: float = 0.75
    epsilon: float = 0.25


class BM25Okapi:
    """BM25-Okapi ranker.

    This avoids an external rank_bm25 dependency and keeps the project self-contained.
    It expects already-tokenized documents.
    """

    def __init__(self, tokenized_corpus: Sequence[Sequence[str]], config: BM25Config | None = None):
        self.config = config or BM25Config()
        self.corpus_size = len(tokenized_corpus)
        if self.corpus_size == 0:
            raise ValueError("BM25 corpus is empty")
        self.doc_lens = np.array([len(doc) for doc in tokenized_corpus], dtype=np.float32)
        self.avgdl = float(np.mean(self.doc_lens)) if self.corpus_size else 0.0
        self.term_freqs: list[Counter[str]] = [Counter(doc) for doc in tokenized_corpus]
        self.idf = self._compute_idf()

    def _compute_idf(self) -> dict[str, float]:
        df = defaultdict(int)
        for freqs in self.term_freqs:
            for term in freqs.keys():
                df[term] += 1

        idf = {}
        idf_sum = 0.0
        negative_terms = []
        for term, freq in df.items():
            # Classic Robertson/Sparck Jones idf variant used by rank_bm25.
            value = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            idf[term] = value
            idf_sum += value
            if value < 0:
                negative_terms.append(term)

        average_idf = idf_sum / max(len(idf), 1)
        floor = self.config.epsilon * average_idf
        for term in negative_terms:
            idf[term] = floor
        return idf

    def get_scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        if not query_tokens:
            return scores

        k1 = self.config.k1
        b = self.config.b
        denom_const = k1 * (1.0 - b + b * self.doc_lens / max(self.avgdl, 1e-9))

        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            freqs = np.array([doc_freqs.get(term, 0) for doc_freqs in self.term_freqs], dtype=np.float32)
            numerator = freqs * (k1 + 1.0)
            denominator = freqs + denom_const
            scores += idf * (numerator / np.where(denominator == 0, 1.0, denominator))
        return scores

    def get_top_n(self, query_tokens: Sequence[str], n: int = 10) -> tuple[np.ndarray, np.ndarray]:
        scores = self.get_scores(query_tokens)
        n = min(n, len(scores))
        if n <= 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        top_idx = np.argpartition(-scores, n - 1)[:n]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return top_idx, scores[top_idx]
