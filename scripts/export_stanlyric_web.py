#!/usr/bin/env python3
"""Export a StanLyric corpus into a static browser-search artifact.

This script is intended for the portfolio repository, not the StanLyric package.
It reads the prepared StanLyric corpus from CSV/Parquet/JSON/Pickle and writes
an index JSON that can be loaded by assets/js/stanlyric/stanlyric.js on GitHub Pages.

Public-site default: metadata + BM25 index only, no full lyrics. For private/local
development, pass --include-full-lyrics to enable matched snippets and collapsible lyrics.
Output filenames are tagged with ``_with_lyrics`` or ``_without_lyrics``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

LYRIC_COLUMNS = ["lyrics", "lyric", "song_lyrics", "clean_lyrics", "text", "body", "content"]
TITLE_COLUMNS = ["title", "song", "song_title", "track", "track_name", "name"]
ARTIST_COLUMNS = ["artist", "artist_name", "singer", "performer", "band"]
SOURCE_COLUMNS = ["source", "dataset", "path", "file", "filename", "url", "source_path"]
OUTPUT_VARIANTS = ("with_lyrics", "without_lyrics")


def canonical(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def infer_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {canonical(col): col for col in df.columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    for candidate in candidates:
        for key, original in lookup.items():
            if candidate in key:
                return original
    return None


def variant_output_path(path: Path, include_full_lyrics: bool) -> Path:
    """Add an explicit lyrics-content tag to a JSON output filename."""
    suffix = path.suffix or ".json"
    stem = path.stem if path.suffix else path.name
    for variant in OUTPUT_VARIANTS:
        tag = f"_{variant}"
        if stem.endswith(tag):
            stem = stem[: -len(tag)]
            break
    variant = "with_lyrics" if include_full_lyrics else "without_lyrics"
    return path.with_name(f"{stem}_{variant}{suffix}")


def read_any(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            with path.open("r", encoding="utf-8") as handle:
                obj = json.load(handle)
            if isinstance(obj, dict) and "docs" in obj:
                return pd.DataFrame(obj["docs"])
            return pd.DataFrame(obj)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported input file type: {path}")


def normalize_text(text: Any) -> str:
    return (
        str(text or "")
        .lower()
        .replace("’", "")
        .replace("'", "")
    )


def tokenize(text: Any) -> list[str]:
    normalized = normalize_text(text)
    normalized = re.sub(r"[^a-z0-9\s]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.split() if normalized else []


def to_records(
    df: pd.DataFrame,
    max_docs: int | None = None,
    min_lyric_chars: int = 80,
) -> pd.DataFrame:
    lyric_col = infer_column(df, LYRIC_COLUMNS)
    if lyric_col is None:
        raise ValueError(f"Could not infer lyrics column. Available columns: {list(df.columns)}")
    title_col = infer_column(df, TITLE_COLUMNS)
    artist_col = infer_column(df, ARTIST_COLUMNS)
    source_col = infer_column(df, SOURCE_COLUMNS)

    corpus = pd.DataFrame()
    corpus["doc_id"] = (
        df["doc_id"].astype(str)
        if "doc_id" in df.columns
        else [f"song_{i:07d}" for i in range(len(df))]
    )
    corpus["title"] = df[title_col].fillna("").astype(str) if title_col else ""
    corpus["artist"] = df[artist_col].fillna("").astype(str) if artist_col else ""
    corpus["lyrics"] = df[lyric_col].fillna("").astype(str)
    corpus["source"] = df[source_col].fillna("").astype(str) if source_col else ""
    corpus["lyrics_char_len"] = corpus["lyrics"].str.len()
    corpus = corpus[corpus["lyrics_char_len"] >= min_lyric_chars].copy()
    corpus = corpus.drop_duplicates(subset=["lyrics"]).reset_index(drop=True)
    if max_docs is not None:
        corpus = corpus.head(max_docs).copy()
    corpus["doc_id"] = [f"song_{i:07d}" for i in range(len(corpus))]
    return corpus


def build_index(corpus: pd.DataFrame, *, k1: float, b: float, epsilon: float) -> dict[str, Any]:
    tokenized = [tokenize(text) for text in corpus["lyrics"].tolist()]
    doc_lens = [len(tokens) for tokens in tokenized]
    corpus_size = len(tokenized)
    if corpus_size == 0:
        raise ValueError("Corpus is empty after filtering.")
    avgdl = sum(doc_lens) / corpus_size

    term_freqs = [Counter(tokens) for tokens in tokenized]
    df_counts: dict[str, int] = defaultdict(int)
    for freqs in term_freqs:
        for term in freqs:
            df_counts[term] += 1

    idf: dict[str, float] = {}
    idf_sum = 0.0
    negative_terms = []
    for term, freq in df_counts.items():
        value = math.log(corpus_size - freq + 0.5) - math.log(freq + 0.5)
        idf[term] = value
        idf_sum += value
        if value < 0:
            negative_terms.append(term)
    average_idf = idf_sum / max(len(idf), 1)
    floor = epsilon * average_idf
    for term in negative_terms:
        idf[term] = floor

    postings: dict[str, list[list[int]]] = defaultdict(list)
    for doc_index, freqs in enumerate(term_freqs):
        for term, tf in freqs.items():
            postings[term].append([doc_index, int(tf)])

    return {
        "avgdl": avgdl,
        "doc_lens": doc_lens,
        "idf": {term: round(float(value), 8) for term, value in idf.items()},
        "postings": dict(postings),
        "metadata": {
            "n_docs": corpus_size,
            "vocabulary_size": len(idf),
            "avgdl": avgdl,
        },
    }


def read_metrics(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".json"}:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep).to_dict(orient="records")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(f"Unsupported metrics file type: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        required=True,
        type=Path,
        help="Prepared StanLyric corpus: parquet/csv/json/pickle.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/json/stanlyric/stanlyric_web_index.json"),
        help=(
            "Base output path. The exporter appends _with_lyrics or "
            "_without_lyrics before .json."
        ),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Optional metrics file from offline evaluation.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("assets/json/stanlyric/stanlyric_eval_metrics.json"),
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional limit for quick local testing.",
    )
    parser.add_argument("--min-lyric-chars", type=int, default=80)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--epsilon", type=float, default=0.25)
    parser.add_argument(
        "--include-full-lyrics",
        action="store_true",
        help="Includes full lyric text in the exported JSON.",
    )
    parser.add_argument(
        "--attribution-url",
        default="https://huggingface.co/datasets/asigalov61/Lyrics-MIDI-Dataset",
    )
    args = parser.parse_args()
    output_path = variant_output_path(args.output, args.include_full_lyrics)

    raw = read_any(args.corpus)
    corpus = to_records(raw, max_docs=args.max_docs, min_lyric_chars=args.min_lyric_chars)
    index = build_index(corpus, k1=args.k1, b=args.b, epsilon=args.epsilon)

    docs = []
    for _, row in corpus.iterrows():
        source_id = str(row.get("source", "") or "")
        metadata_key = source_id or str(row["doc_id"])
        doc = {
            "doc_id": row["doc_id"],
            "metadata_key": metadata_key,
            "title": row.get("title", ""),
            "artist": row.get("artist", ""),
            "source": source_id,
            "lyrics_char_len": int(row.get("lyrics_char_len", 0)),
        }
        if args.include_full_lyrics:
            doc["lyrics"] = row["lyrics"]
        docs.append(doc)

    payload = {
        "metadata": {
            "project": "StanLyric",
            "description": "Static BM25 lyric-fragment search artifact for GitHub Pages.",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_docs": len(docs),
            "vocabulary_size": index["metadata"]["vocabulary_size"],
            "avgdl": index["metadata"]["avgdl"],
            "include_full_lyrics": bool(args.include_full_lyrics),
            "song_metadata_join_field": "metadata_key",
            "attribution_url": args.attribution_url,
            "license_note": (
                "Lyrics-MIDI-Dataset is listed as CC-BY-NC-SA 4.0; original "
                "lyrics belong to source datasets and creators."
            ),
        },
        "bm25": {"k1": args.k1, "b": args.b, "epsilon": args.epsilon},
        "avgdl": index["avgdl"],
        "doc_lens": index["doc_lens"],
        "docs": docs,
        "idf": index["idf"],
        "postings": index["postings"],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    print(
        f"Wrote {output_path} "
        f"({len(docs):,} docs, {index['metadata']['vocabulary_size']:,} terms)"
    )

    if args.metrics:
        metrics_payload = read_metrics(args.metrics)
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_output.open("w", encoding="utf-8") as handle:
            json.dump(metrics_payload, handle, ensure_ascii=False, indent=2)
        print(f"Wrote {args.metrics_output}")


if __name__ == "__main__":
    main()
