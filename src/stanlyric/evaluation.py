"""IR evaluation utilities for lyric fragment search."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from stanlyric.search import StanLyricSearchEngine
from stanlyric.text import tokenize


@dataclass
class EvalConfig:
    n_queries: int = 200
    min_query_tokens: int = 5
    max_query_tokens: int = 18
    random_state: int = 42
    top_k: int = 10


def hit_at_k(ranks: Sequence[int | None], k: int) -> float:
    return float(np.mean([rank is not None and rank <= k for rank in ranks])) if ranks else 0.0


def mrr_at_k(ranks: Sequence[int | None], k: int) -> float:
    vals = [1.0 / rank if rank is not None and rank <= k else 0.0 for rank in ranks]
    return float(np.mean(vals)) if vals else 0.0


def ndcg_at_k(ranks: Sequence[int | None], k: int) -> float:
    """Mean NDCG@k for one known relevant document per query."""
    vals = []
    for rank in ranks:
        if rank is not None and rank <= k:
            vals.append(1.0 / np.log2(rank + 1.0))
        else:
            vals.append(0.0)
    return float(np.mean(vals)) if vals else 0.0


def recall_at_k(ranks: Sequence[int | None], k: int) -> float:
    # With exactly one target document per query, recall@k equals hit@k.
    return hit_at_k(ranks, k)


def make_lyric_fragment(
    lyrics: str,
    min_query_tokens: int = 5,
    max_query_tokens: int = 18,
    rng: random.Random | None = None,
) -> str | None:
    """Sample a contiguous lyric fragment from one song."""
    rng = rng or random.Random()
    tokens = tokenize(lyrics)
    if len(tokens) < min_query_tokens:
        return None
    span_len = rng.randint(min_query_tokens, min(max_query_tokens, len(tokens)))
    start = rng.randint(0, len(tokens) - span_len)
    return " ".join(tokens[start : start + span_len])


def make_eval_queries(corpus: pd.DataFrame, config: EvalConfig | None = None) -> pd.DataFrame:
    """Create a synthetic benchmark by sampling lyric fragments from known songs.

    This gives an automatic sanity-check benchmark for retrieval. Later, you can add
    harder query sets by injecting typos, removing stopwords, or using user-written
    paraphrases.
    """
    config = config or EvalConfig()
    rng = random.Random(config.random_state)
    eligible = corpus[corpus["lyrics"].fillna("").map(lambda x: len(tokenize(x)) >= config.min_query_tokens)]
    if eligible.empty:
        raise ValueError("No songs have enough lyric tokens for evaluation queries")
    sample_size = min(config.n_queries, len(eligible))
    sampled = eligible.sample(n=sample_size, random_state=config.random_state)

    rows = []
    for _, row in sampled.iterrows():
        fragment = make_lyric_fragment(
            row["lyrics"],
            min_query_tokens=config.min_query_tokens,
            max_query_tokens=config.max_query_tokens,
            rng=rng,
        )
        if fragment:
            rows.append(
                {
                    "query": fragment,
                    "target_doc_id": row["doc_id"],
                    "target_title": row.get("title", ""),
                    "target_artist": row.get("artist", ""),
                }
            )
    return pd.DataFrame(rows)


class RetrievalEvaluator:
    """Evaluate StanLyric search results using standard IR metrics."""

    def __init__(self, engine: StanLyricSearchEngine):
        self.engine = engine

    def evaluate_queries(self, queries: pd.DataFrame, top_k: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
        required = {"query", "target_doc_id"}
        missing = required - set(queries.columns)
        if missing:
            raise ValueError(f"queries missing required columns: {sorted(missing)}")

        detail_rows = []
        ranks = []
        for _, qrow in tqdm(queries.iterrows(), total=len(queries), desc="Evaluating retrieval"):
            results = self.engine.search(qrow["query"], top_k=top_k, include_full_lyrics=False)
            target = qrow["target_doc_id"]
            match = results[results["doc_id"] == target]
            rank = int(match.iloc[0]["rank"]) if not match.empty else None
            ranks.append(rank)
            detail_rows.append(
                {
                    "query": qrow["query"],
                    "target_doc_id": target,
                    "target_title": qrow.get("target_title", ""),
                    "target_artist": qrow.get("target_artist", ""),
                    "rank": rank,
                    "found_in_top_k": rank is not None,
                    "top_prediction": results.iloc[0]["title"] if len(results) else "",
                    "top_artist": results.iloc[0]["artist"] if len(results) else "",
                    "top_doc_id": results.iloc[0]["doc_id"] if len(results) else "",
                    "top_score": results.iloc[0]["bm25_score"] if len(results) else 0.0,
                }
            )

        metrics = {
            "n_queries": len(ranks),
            "hit@1": hit_at_k(ranks, 1),
            "hit@3": hit_at_k(ranks, 3),
            "hit@5": hit_at_k(ranks, 5),
            "hit@10": hit_at_k(ranks, 10),
            "recall@10": recall_at_k(ranks, 10),
            "mrr@10": mrr_at_k(ranks, 10),
            "ndcg@10": ndcg_at_k(ranks, 10),
            "median_rank_found": _median_rank_found(ranks),
            "miss_rate@10": 1.0 - hit_at_k(ranks, 10),
        }
        metrics_df = pd.DataFrame([metrics])
        details_df = pd.DataFrame(detail_rows)
        return metrics_df, details_df

    def evaluate_synthetic(self, config: EvalConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        config = config or EvalConfig()
        queries = make_eval_queries(self.engine.corpus, config)
        metrics, details = self.evaluate_queries(queries, top_k=config.top_k)
        return metrics, details, queries


def _median_rank_found(ranks: Iterable[int | None]) -> float:
    found = [r for r in ranks if r is not None]
    return float(np.median(found)) if found else float("nan")
