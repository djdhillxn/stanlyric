#!/usr/bin/env python3
"""Discover likely lyrics/metadata files in the Hugging Face dataset repo."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.hf_download import DEFAULT_REPO_ID, candidates_to_dataframe, rank_lyric_files


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--max-size-mb", type=float, default=900.0)
    parser.add_argument("--output", default="data/raw/hf_candidate_files.csv")
    parser.add_argument("--top", type=int, default=25)
    return parser.parse_args()


def main():
    args = parse_args()
    candidates = rank_lyric_files(args.repo_id, max_size_mb=args.max_size_mb)
    df = candidates_to_dataframe(candidates)
    if not df.empty:
        print(df.head(args.top).to_string(index=False))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\nSaved candidate manifest to {args.output}")
    else:
        print("No candidates found. Try increasing --max-size-mb or inspect the repo manually.")


if __name__ == "__main__":
    main()
