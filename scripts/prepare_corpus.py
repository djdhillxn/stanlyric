#!/usr/bin/env python3
"""Convert a raw lyrics file/directory into StanLyric's canonical corpus.parquet."""

from __future__ import annotations

import argparse

from stanlyric.data import CorpusConfig, load_raw_dataset, prepare_corpus, save_corpus


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Raw downloaded file or directory.")
    parser.add_argument("--output", default="data/processed/corpus.parquet")
    parser.add_argument("--min-lyric-chars", type=int, default=80)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--no-deduplicate", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    raw = load_raw_dataset(args.input)
    corpus = prepare_corpus(
        raw,
        CorpusConfig(
            min_lyric_chars=args.min_lyric_chars,
            max_docs=args.max_docs,
            deduplicate_lyrics=not args.no_deduplicate,
        ),
    )
    path = save_corpus(corpus, args.output)
    print(f"Prepared corpus: {path}")
    print(f"Rows: {len(corpus):,}")
    print(corpus[["doc_id", "title", "artist", "lyrics_char_len"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
