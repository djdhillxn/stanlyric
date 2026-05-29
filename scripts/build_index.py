#!/usr/bin/env python3
"""Build and save a BM25 StanLyric search index."""

from __future__ import annotations

import argparse

from stanlyric.bm25 import BM25Config
from stanlyric.search import StanLyricSearchEngine


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/processed/corpus.parquet")
    parser.add_argument("--output", default="data/processed/stanlyric_bm25.pkl")
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    return parser.parse_args()


def main():
    args = parse_args()
    engine = StanLyricSearchEngine.from_corpus_path(
        args.corpus,
        bm25_config=BM25Config(k1=args.k1, b=args.b),
    )
    path = engine.save(args.output)
    print(f"Saved BM25 index: {path}")
    print(f"Indexed songs: {len(engine.corpus):,}")


if __name__ == "__main__":
    main()
