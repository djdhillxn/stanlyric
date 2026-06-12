#!/usr/bin/env python3
"""Download one lyrics/metadata file from Hugging Face without pulling the full dataset."""

from __future__ import annotations

import argparse

from src.hf_download import (
    DEFAULT_REPO_ID,
    auto_download_best_candidate,
    candidates_to_dataframe,
    download_file,
    rank_lyric_files,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--max-size-mb", type=float, default=900.0)
    parser.add_argument("--filename", default=None, help="Exact filename/path in the HF dataset repo.")
    parser.add_argument("--rank", type=int, default=0, help="Download the Nth ranked candidate if --filename is omitted.")
    parser.add_argument("--dry-run", action="store_true", help="Only show ranked candidates; do not download.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dry_run:
        candidates = rank_lyric_files(args.repo_id, max_size_mb=args.max_size_mb)
        print(candidates_to_dataframe(candidates).head(25).to_string(index=False))
        return

    if args.filename:
        local_path = download_file(args.filename, repo_id=args.repo_id, output_dir=args.output_dir)
        print(f"Downloaded: {local_path}")
        return

    local_path, candidates = auto_download_best_candidate(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        max_size_mb=args.max_size_mb,
        rank=args.rank,
    )
    print("Top candidates:")
    print(candidates_to_dataframe(candidates).head(10).to_string(index=False))
    print(f"\nDownloaded selected candidate: {local_path}")


if __name__ == "__main__":
    main()
