"""High-level StanLyric search engine."""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from stanlyric.bm25 import BM25Config, BM25Okapi
from stanlyric.data import load_corpus
from stanlyric.text import normalize_text, tokenize, unique_preserve_order


@dataclass
class SearchResultConfig:
    snippet_chars: int = 420
    max_explanation_terms: int = 8
    include_full_lyrics: bool = False


class StanLyricSearchEngine:
    """BM25 lyric fragment search engine.

    Typical usage:
        engine = StanLyricSearchEngine.from_corpus_path("data/processed/corpus.parquet")
        results = engine.search("some lyric fragment", top_k=10)
    """

    def __init__(
        self,
        corpus: pd.DataFrame,
        bm25_config: BM25Config | None = None,
        result_config: SearchResultConfig | None = None,
    ):
        required = {"doc_id", "title", "artist", "lyrics"}
        missing = required - set(corpus.columns)
        if missing:
            raise ValueError(f"Corpus missing required columns: {sorted(missing)}")

        self.corpus = corpus.reset_index(drop=True).copy()
        self.bm25_config = bm25_config or BM25Config()
        self.result_config = result_config or SearchResultConfig()
        self.tokenized_corpus = [tokenize(x) for x in tqdm(self.corpus["lyrics"], desc="Tokenizing lyrics")]
        self.bm25 = BM25Okapi(self.tokenized_corpus, config=self.bm25_config)

    @classmethod
    def from_corpus_path(
        cls,
        path: str | Path,
        bm25_config: BM25Config | None = None,
        result_config: SearchResultConfig | None = None,
    ) -> "StanLyricSearchEngine":
        return cls(load_corpus(path), bm25_config=bm25_config, result_config=result_config)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
        return path

    @staticmethod
    def load(path: str | Path) -> "StanLyricSearchEngine":
        with Path(path).open("rb") as f:
            return pickle.load(f)

    def search(
        self,
        query: str,
        top_k: int = 10,
        include_full_lyrics: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Retrieve top-k songs for a lyric fragment."""
        include_full_lyrics = (
            self.result_config.include_full_lyrics if include_full_lyrics is None else include_full_lyrics
        )
        query_tokens = tokenize(query)
        top_idx, top_scores = self.bm25.get_top_n(query_tokens, n=top_k)
        all_scores = self.bm25.get_scores(query_tokens)
        confidence = _scores_to_confidence(top_scores)
        rows = []
        query_terms = set(query_tokens)

        for rank, (idx, score, conf) in enumerate(zip(top_idx, top_scores, confidence), start=1):
            row = self.corpus.iloc[int(idx)]
            lyric_text = row["lyrics"]
            matched_terms = self._matched_terms(int(idx), query_terms)
            rows.append(
                {
                    "rank": rank,
                    "doc_id": row.get("doc_id", ""),
                    "title": row.get("title", ""),
                    "artist": row.get("artist", ""),
                    "bm25_score": float(score),
                    "confidence": float(conf),
                    "score_percentile": float(_score_percentile(all_scores, score)),
                    "matched_terms": ", ".join(matched_terms[: self.result_config.max_explanation_terms]),
                    "snippet": self._best_snippet(lyric_text, query_terms),
                    "method": "BM25-Okapi",
                    "source": row.get("source", ""),
                    **({"full_lyrics": lyric_text} if include_full_lyrics else {}),
                }
            )
        return pd.DataFrame(rows)

    def explain_query(self, query: str, doc_id: str | None = None, rank: int = 1) -> dict:
        """Return a lightweight explanation for a retrieved result.

        If doc_id is omitted, the top-ranked result for the query is explained.
        """
        results = self.search(query, top_k=max(rank, 1), include_full_lyrics=False)
        if doc_id is not None:
            candidates = results[results["doc_id"] == doc_id]
            if candidates.empty:
                raise ValueError(f"doc_id={doc_id!r} was not in the retrieved top results")
            result = candidates.iloc[0]
        else:
            result = results.iloc[rank - 1]

        q_tokens = unique_preserve_order(tokenize(query))
        idx = int(self.corpus.index[self.corpus["doc_id"] == result["doc_id"]][0])
        doc_freqs = self.bm25.term_freqs[idx]
        term_rows = []
        for term in q_tokens:
            if term in self.bm25.idf and doc_freqs.get(term, 0) > 0:
                term_rows.append(
                    {
                        "term": term,
                        "doc_tf": int(doc_freqs[term]),
                        "idf": float(self.bm25.idf[term]),
                    }
                )
        term_rows = sorted(term_rows, key=lambda x: x["idf"], reverse=True)
        return {
            "query": query,
            "result": result.drop(labels=["snippet"], errors="ignore").to_dict(),
            "top_matching_terms_by_idf": term_rows,
            "snippet": result["snippet"],
        }

    def _matched_terms(self, doc_idx: int, query_terms: set[str]) -> list[str]:
        doc_terms = set(self.tokenized_corpus[doc_idx])
        terms = [t for t in query_terms if t in doc_terms]
        terms.sort(key=lambda t: self.bm25.idf.get(t, 0.0), reverse=True)
        return terms

    def _best_snippet(self, lyric_text: str, query_terms: set[str]) -> str:
        """Find a short lyric window around the highest-overlap lines."""
        if not lyric_text:
            return ""
        lines = [line.strip() for line in str(lyric_text).splitlines() if line.strip()]
        if not lines:
            text = str(lyric_text).strip()
            return text[: self.result_config.snippet_chars]

        best_i = 0
        best_score = -1.0
        for i, line in enumerate(lines):
            line_terms = set(tokenize(line))
            overlap = len(query_terms & line_terms)
            rare_bonus = sum(self.bm25.idf.get(t, 0.0) for t in query_terms & line_terms)
            score = overlap + 0.1 * rare_bonus
            if score > best_score:
                best_score = score
                best_i = i

        start = max(0, best_i - 2)
        end = min(len(lines), best_i + 3)
        snippet = "\n".join(lines[start:end])
        if len(snippet) > self.result_config.snippet_chars:
            snippet = snippet[: self.result_config.snippet_chars].rsplit(" ", 1)[0] + "..."
        return _highlight_terms(snippet, query_terms)


def _scores_to_confidence(scores: np.ndarray) -> np.ndarray:
    """Convert arbitrary BM25 scores into top-k relative confidence.

    This is not a calibrated probability. It is a local softmax-style confidence among
    the returned candidates, useful for UI ranking only.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return scores
    if np.allclose(scores, scores[0]):
        return np.ones_like(scores) / len(scores)
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    denom = np.sum(exp_scores)
    return exp_scores / denom if denom > 0 else np.zeros_like(scores)


def _score_percentile(all_scores: np.ndarray, score: float) -> float:
    all_scores = np.asarray(all_scores)
    if all_scores.size == 0:
        return 0.0
    return float(100.0 * np.mean(all_scores <= score))


def _highlight_terms(text: str, terms: set[str]) -> str:
    """Markdown-bold matched query terms in snippets."""
    if not terms:
        return text
    escaped = [re.escape(t) for t in terms if t]
    if not escaped:
        return text
    pattern = re.compile(r"\b(" + "|".join(escaped) + r")\b", flags=re.I)
    return pattern.sub(lambda m: f"**{m.group(0)}**", text)
