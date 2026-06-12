"""Dataset loading, column inference, and corpus preparation."""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd


LYRIC_COLUMN_CANDIDATES = [
    "lyrics",
    "lyric",
    "song_lyrics",
    "clean_lyrics",
    "text",
    "body",
    "content",
]
TITLE_COLUMN_CANDIDATES = ["title", "song", "song_title", "track", "track_name", "name"]
ARTIST_COLUMN_CANDIDATES = ["artist", "artist_name", "singer", "performer", "band"]
SOURCE_COLUMN_CANDIDATES = ["source", "dataset", "path", "file", "filename", "url"]


@dataclass
class CorpusConfig:
    min_lyric_chars: int = 80
    max_docs: Optional[int] = None
    deduplicate_lyrics: bool = True


def _canonical_column_name(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(col).lower()).strip("_")


def infer_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Find the best matching column from a candidate list."""
    canonical_to_original = {_canonical_column_name(c): c for c in df.columns}
    for cand in candidates:
        if cand in canonical_to_original:
            return canonical_to_original[cand]
    # Fuzzy containment fallback: useful for columns like "cleaned_lyrics".
    for cand in candidates:
        for canon, original in canonical_to_original.items():
            if cand in canon:
                return original
    return None


def read_any_table(path: str | Path) -> pd.DataFrame:
    """Read CSV/TSV/JSON/JSONL/Parquet/Pickle into a DataFrame.

    Pickle files from public datasets can contain either DataFrames, dicts, or lists of
    records. This function handles all three common patterns.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    if suffix in {".parquet", ".pq"}:
        return _read_parquet(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            return _object_to_dataframe(obj)
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as f:
            obj = pickle.load(f)
        return _object_to_dataframe(obj)

    raise ValueError(f"Unsupported table file type: {path}")


def _object_to_dataframe(obj: Any) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    if isinstance(obj, dict):
        # Common patterns: {"data": [...]}, {id: record}, or dict of columns.
        for key in ["data", "train", "records", "items"]:
            if key in obj and isinstance(obj[key], list):
                return pd.DataFrame(obj[key])
        try:
            return pd.DataFrame(obj)
        except ValueError:
            return pd.DataFrame.from_dict(obj, orient="index").reset_index(names="source_id")
    raise ValueError(f"Cannot convert object of type {type(obj)} to DataFrame")


def load_txt_directory(directory: str | Path) -> pd.DataFrame:
    """Load a directory of .txt lyric files into a DataFrame."""
    directory = Path(directory)
    rows = []
    for path in sorted(directory.rglob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        stem = path.stem
        title, artist = _parse_title_artist_from_stem(stem)
        rows.append(
            {
                "doc_id": str(path.relative_to(directory)),
                "title": title,
                "artist": artist,
                "lyrics": text,
                "source_path": str(path),
            }
        )
    if not rows:
        raise ValueError(f"No .txt files found under {directory}")
    return pd.DataFrame(rows)


def _parse_title_artist_from_stem(stem: str) -> tuple[str, str]:
    # Handles common patterns like "artist - title" or "artist_title".
    for sep in [" - ", "--", "__"]:
        if sep in stem:
            left, right = stem.split(sep, 1)
            return right.strip(), left.strip()
    return stem.replace("_", " ").strip(), ""


def load_raw_dataset(path: str | Path) -> pd.DataFrame:
    """Load either a single table file or a directory of lyric .txt files."""
    path = Path(path)
    if path.is_dir():
        # Prefer tabular files if present; otherwise read .txt files.
        table_files = []
        for ext in [
            "*.parquet",
            "*.pq",
            "*.csv",
            "*.tsv",
            "*.jsonl",
            "*.json",
            "*.pkl",
            "*.pickle",
        ]:
            table_files.extend(path.rglob(ext))
        if table_files:
            frames = [read_any_table(p) for p in sorted(table_files)]
            return pd.concat(frames, ignore_index=True)
        return load_txt_directory(path)
    return read_any_table(path)


def prepare_corpus(df: pd.DataFrame, config: CorpusConfig | None = None) -> pd.DataFrame:
    """Normalize arbitrary lyric data into StanLyric's canonical corpus schema.

    Output columns:
    - doc_id: stable row-level id
    - title: song title if available
    - artist: artist if available
    - lyrics: full lyric text
    - source: source/path/file if available
    """
    config = config or CorpusConfig()
    df = _maybe_reshape_wide_lyrics_table(df)
    lyric_col = infer_column(df, LYRIC_COLUMN_CANDIDATES)
    if lyric_col is None:
        raise ValueError(
            "Could not infer a lyrics/text column. Available columns: "
            + ", ".join(map(str, df.columns))
        )

    title_col = infer_column(df, TITLE_COLUMN_CANDIDATES)
    artist_col = infer_column(df, ARTIST_COLUMN_CANDIDATES)
    source_col = infer_column(df, SOURCE_COLUMN_CANDIDATES)

    corpus = pd.DataFrame()
    corpus["doc_id"] = [f"song_{i:07d}" for i in range(len(df))]
    corpus["title"] = df[title_col].fillna("").astype(str) if title_col else ""
    corpus["artist"] = df[artist_col].fillna("").astype(str) if artist_col else ""
    corpus["lyrics"] = df[lyric_col].fillna("").astype(str)
    corpus["source"] = df[source_col].fillna("").astype(str) if source_col else ""

    corpus["lyrics_char_len"] = corpus["lyrics"].str.len()
    corpus = corpus[corpus["lyrics_char_len"] >= config.min_lyric_chars].copy()
    if corpus.empty:
        max_len = df[lyric_col].fillna("").astype(str).str.len().max()
        raise ValueError(
            f"No corpus rows remain after filtering lyrics shorter than "
            f"{config.min_lyric_chars} characters. Inferred lyrics column: {lyric_col!r}; "
            f"max observed lyric length: {max_len}; input shape: {df.shape}."
        )
    if config.deduplicate_lyrics:
        corpus["_dedupe_key"] = (
            corpus["lyrics"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
        )
        corpus = corpus.drop_duplicates("_dedupe_key").drop(columns=["_dedupe_key"])
    if config.max_docs is not None:
        corpus = corpus.head(config.max_docs).copy()

    corpus = corpus.reset_index(drop=True)
    corpus["doc_id"] = [f"song_{i:07d}" for i in range(len(corpus))]
    return corpus[["doc_id", "title", "artist", "lyrics", "source", "lyrics_char_len"]]


def _maybe_reshape_wide_lyrics_table(df: pd.DataFrame) -> pd.DataFrame:
    """Convert tables with one song per column into one song per row.

    Some lyrics datasets store records transposed, with columns like
    "Title --- Artist --- id" and index rows such as "keywords" and "lyrics".
    """
    if df.empty or df.shape[0] > 20 or _has_exact_column(df, LYRIC_COLUMN_CANDIDATES):
        return df

    canonical_index = {_canonical_column_name(label): label for label in df.index}
    lyric_row = None
    for candidate in LYRIC_COLUMN_CANDIDATES:
        if candidate in canonical_index:
            lyric_row = canonical_index[candidate]
            break
    if lyric_row is None:
        return df

    rows = []
    for column in df.columns:
        lyrics = df.at[lyric_row, column]
        title, artist, source = _parse_wide_song_column(column)
        rows.append(
            {
                "title": title,
                "artist": artist,
                "lyrics": "" if _is_missing_value(lyrics) else str(lyrics),
                "source": source,
            }
        )
    return pd.DataFrame(rows)


def _has_exact_column(df: pd.DataFrame, candidates: Iterable[str]) -> bool:
    canonical_columns = {_canonical_column_name(column) for column in df.columns}
    return any(candidate in canonical_columns for candidate in candidates)


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, set)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _parse_wide_song_column(column: Any) -> tuple[str, str, str]:
    source = str(column)
    parts = [part.strip() for part in source.split(" --- ")]
    if len(parts) >= 3:
        return parts[0], " --- ".join(parts[1:-1]), parts[-1]
    if len(parts) == 2:
        return parts[0], parts[1], source
    title, artist = _parse_title_artist_from_stem(source)
    return title, artist, source


def save_corpus(corpus: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in {".parquet", ".pq"}:
        _write_parquet(corpus, output_path)
    else:
        corpus.to_csv(output_path, index=False)
    return output_path


def load_corpus(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return _read_parquet(path)
    return read_any_table(path)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write parquet without pandas' parquet-engine shim.

    Some pandas/pyarrow version combinations raise ArrowKeyError while pandas
    patches pyarrow extension types before writing. Calling pyarrow directly
    avoids that engine setup path while keeping the same parquet output format.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Writing parquet requires pyarrow. Install pyarrow or use a .csv output path."
        ) from exc

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path)


def _read_parquet(path: Path) -> pd.DataFrame:
    """Read parquet through pyarrow directly for symmetry with _write_parquet."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Reading parquet requires pyarrow. Install pyarrow or use a CSV input path."
        ) from exc

    return pq.read_table(path).to_pandas()
