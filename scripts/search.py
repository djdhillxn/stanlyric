#!/usr/bin/env python3
"""Search StanLyric from the command line."""

from __future__ import annotations

import argparse

from src.search import StanLyricSearchEngine


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/processed/stanlyric_bm25.pkl")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--include-full-lyrics", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    engine = StanLyricSearchEngine.load(args.index)
    results = engine.search(args.query, top_k=args.top_k, include_full_lyrics=args.include_full_lyrics)
    cols = ["rank", "title", "artist", "bm25_score", "confidence", "matched_terms", "snippet"]
    print(results[cols].to_string(index=False))


if __name__ == "__main__":
    main()
