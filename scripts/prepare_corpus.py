#!/usr/bin/env python3
"""Convert a raw lyrics file/directory into StanLyric's canonical corpus.parquet."""

from __future__ import annotations

import argparse

from src.data import (
    CorpusCleaningConfig,
    CorpusConfig,
    clean_corpus,
    load_raw_dataset,
    prepare_corpus,
    save_corpus,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Raw downloaded file or directory.")
    parser.add_argument("--output", default="data/processed/corpus.parquet")
    parser.add_argument("--min-lyric-chars", type=int, default=80)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument(
        "--keep-unknown-artists",
        action="store_true",
        help="Retain rows whose artist is blank or marked unknown.",
    )
    parser.add_argument(
        "--keep-missing-titles",
        action="store_true",
        help="Retain rows with a blank song title.",
    )
    parser.add_argument(
        "--no-exact-lyric-deduplication",
        action="store_true",
        help="Do not collapse exact lyric copies.",
    )
    parser.add_argument(
        "--no-song-identity-deduplication",
        action="store_true",
        help="Do not keep one representative per normalized artist-title identity.",
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    raw = load_raw_dataset(args.input)
    corpus = prepare_corpus(
        raw,
        CorpusConfig(
            min_lyric_chars=args.min_lyric_chars,
            max_docs=args.max_docs,
            deduplicate_lyrics=False,
        ),
    )
    corpus, report = clean_corpus(
        corpus,
        CorpusCleaningConfig(
            drop_unknown_artists=not args.keep_unknown_artists,
            drop_missing_titles=not args.keep_missing_titles,
            deduplicate_exact_lyrics=not (
                args.no_deduplicate or args.no_exact_lyric_deduplication
            ),
            deduplicate_song_identities=not (
                args.no_deduplicate or args.no_song_identity_deduplication
            ),
        ),
    )
    path = save_corpus(corpus, args.output)
    print(f"Prepared corpus: {path}")
    print("Cleaning report:")
    for name, value in report.to_dict().items():
        print(f"  {name}: {value:,}")
    print(corpus[["doc_id", "title", "artist", "lyrics_char_len"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
