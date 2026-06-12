#!/usr/bin/env python3
"""Evaluate BM25 lyric fragment retrieval."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation import EvalConfig, RetrievalEvaluator
from src.search import StanLyricSearchEngine
from src.visualization import plot_metric_summary, plot_rank_distribution


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/processed/stanlyric_bm25.pkl")
    parser.add_argument("--n-queries", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/evaluation")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = StanLyricSearchEngine.load(args.index)
    evaluator = RetrievalEvaluator(engine)
    metrics, details, queries = evaluator.evaluate_synthetic(
        EvalConfig(n_queries=args.n_queries, top_k=args.top_k, random_state=args.seed)
    )

    metrics.to_csv(output_dir / "metrics.csv", index=False)
    details.to_csv(output_dir / "query_details.csv", index=False)
    queries.to_csv(output_dir / "synthetic_queries.csv", index=False)
    plot_metric_summary(metrics, output_dir / "metric_summary.png")
    if details["rank"].notna().any():
        plot_rank_distribution(details, output_dir / "rank_distribution.png")

    print(metrics.to_string(index=False))
    print(f"Saved evaluation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
