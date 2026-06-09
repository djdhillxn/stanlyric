#!/usr/bin/env python3
"""Export StanLyric song metadata as a browser-friendly JSON lookup."""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def read_corpus(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("Reading a parquet corpus requires pyarrow.") from exc
        return pq.read_table(path).to_pandas()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported corpus file type: {path}")


def parse_source_label(label: Any) -> tuple[str, str, str]:
    text = str(label)
    parts = [part.strip() for part in text.split(" --- ")]
    if len(parts) >= 3:
        return parts[0], " --- ".join(parts[1:-1]), parts[-1]
    if len(parts) == 2:
        return parts[0], parts[1], text
    return text, "", text


def normalize_keywords(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist") and not isinstance(value, str):
        value = value.tolist()
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(keyword).strip() for keyword in value if str(keyword).strip()]


def read_source_metadata(path: Path) -> dict[str, dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix not in {".pkl", ".pickle"}:
        raise ValueError("Source metadata currently expects the processed Lyrics-MIDI pickle.")

    with path.open("rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a dictionary in {path}, found {type(raw).__name__}.")

    records: dict[str, dict[str, Any]] = {}
    for label, value in raw.items():
        title, artist, source_id = parse_source_label(label)
        if not source_id:
            continue
        fields = value if isinstance(value, dict) else {}
        records[source_id] = {
            "title": title,
            "artist": artist,
            "keywords": normalize_keywords(fields.get("keywords")),
        }
    return records


def build_metadata_payload(
    corpus: pd.DataFrame,
    source_records: dict[str, dict[str, Any]],
    *,
    source_filename: str,
    max_docs: int | None = None,
) -> dict[str, Any]:
    required = {"doc_id", "title", "artist"}
    missing = required - set(corpus.columns)
    if missing:
        raise ValueError(f"Corpus is missing required columns: {sorted(missing)}")

    selected = corpus.head(max_docs) if max_docs is not None else corpus
    songs: dict[str, dict[str, Any]] = {}
    keyword_matches = 0

    for _, row in selected.iterrows():
        doc_id = str(row["doc_id"])
        source_id = clean_string(row.get("source", ""))
        metadata_key = source_id or doc_id
        if metadata_key in songs:
            raise ValueError(f"Duplicate metadata key: {metadata_key}")

        source_record = source_records.get(source_id, {})
        keywords = normalize_keywords(source_record.get("keywords"))
        if keywords:
            keyword_matches += 1

        songs[metadata_key] = {
            "doc_id": doc_id,
            "source_id": source_id,
            "title": clean_string(row.get("title")) or source_record.get("title", ""),
            "artist": clean_string(row.get("artist")) or source_record.get("artist", ""),
            "keywords": keywords,
        }

    return {
        "metadata": {
            "project": "StanLyric",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "schema_version": 1,
            "n_songs": len(songs),
            "songs_with_keywords": keyword_matches,
            "key_field": "metadata_key",
            "key_strategy": "source_id with doc_id fallback",
            "source_filename": source_filename,
            "available_fields": ["doc_id", "source_id", "title", "artist", "keywords"],
        },
        "songs": songs,
    }


def clean_string(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument(
        "--source-data",
        required=True,
        type=Path,
        help="Original Lyrics_MIDI_Dataset_Processed_Corpus...pickle file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/json/stanlyric/song_metadata.json"),
    )
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus = read_corpus(args.corpus)
    source_records = read_source_metadata(args.source_data)
    payload = build_metadata_payload(
        corpus,
        source_records,
        source_filename=args.source_data.name,
        max_docs=args.max_docs,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        if args.pretty:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        else:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))

    metadata = payload["metadata"]
    print(
        f"Wrote {args.output} "
        f"({metadata['n_songs']:,} songs, "
        f"{metadata['songs_with_keywords']:,} with keywords)"
    )


if __name__ == "__main__":
    main()
