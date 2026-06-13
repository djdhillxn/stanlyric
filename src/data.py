"""Dataset loading, column inference, and corpus preparation."""

from __future__ import annotations

import json
import math
import pickle
import re
import unicodedata
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
CANONICAL_CORPUS_COLUMNS = [
    "doc_id",
    "title",
    "artist",
    "lyrics",
    "source",
    "lyrics_char_len",
]
UNKNOWN_ARTIST_KEYS = {
    "",
    "artist unknown",
    "n a",
    "na",
    "nan",
    "none",
    "not available",
    "null",
    "unknown",
    "unknown artist",
}
VERSION_TERMS = {
    "acoustic",
    "cover",
    "demo",
    "edit",
    "instrumental",
    "karaoke",
    "live",
    "mix",
    "mono",
    "remaster",
    "remastered",
    "remix",
    "stereo",
    "unplugged",
    "version",
}
WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
PLACEHOLDER_RE = re.compile(
    r"(?:\[(?:\?|x+|censored|inaudible|unintelligible)[^\]]*\]|\*{2,}|\?{3,}|_{3,}|\ufffd)",
    re.IGNORECASE,
)


@dataclass
class CorpusConfig:
    min_lyric_chars: int = 80
    max_docs: Optional[int] = None
    deduplicate_lyrics: bool = False


@dataclass(frozen=True)
class CorpusCleaningConfig:
    """Controls the explicit quality and song-identity cleaning stage."""

    drop_unknown_artists: bool = True
    drop_missing_titles: bool = True
    deduplicate_exact_lyrics: bool = True
    deduplicate_song_identities: bool = True
    preserve_explicit_versions: bool = True


@dataclass(frozen=True)
class CorpusCleaningReport:
    input_rows: int
    unknown_artist_rows_removed: int
    missing_title_rows_removed: int
    exact_lyric_duplicate_groups: int
    exact_lyric_rows_removed: int
    song_identity_duplicate_groups: int
    song_identity_rows_removed: int
    output_rows: int

    @property
    def total_rows_removed(self) -> int:
        return self.input_rows - self.output_rows

    def to_dict(self) -> dict[str, int]:
        return {
            "input_rows": self.input_rows,
            "unknown_artist_rows_removed": self.unknown_artist_rows_removed,
            "missing_title_rows_removed": self.missing_title_rows_removed,
            "exact_lyric_duplicate_groups": self.exact_lyric_duplicate_groups,
            "exact_lyric_rows_removed": self.exact_lyric_rows_removed,
            "song_identity_duplicate_groups": self.song_identity_duplicate_groups,
            "song_identity_rows_removed": self.song_identity_rows_removed,
            "total_rows_removed": self.total_rows_removed,
            "output_rows": self.output_rows,
        }


def _canonical_column_name(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(col).lower()).strip("_")


def normalize_song_component(value: Any) -> str:
    """Normalize title/artist text for stable identity comparisons."""
    text = unicodedata.normalize(
        "NFKD", "" if value is None else str(value)
    ).casefold()
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.replace("&", " and ").replace("\u2019", "").replace("'", "")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def song_identity_key(title: Any, artist: Any) -> str:
    """Return the normalized artist-title key used for song deduplication."""
    return f"{normalize_song_component(artist)}\x1f{normalize_song_component(title)}"


def is_unknown_artist(value: Any) -> bool:
    return normalize_song_component(value) in UNKNOWN_ARTIST_KEYS


def explicit_version_key(title: Any) -> str:
    """Identify named versions that should remain separate search documents."""
    title_tokens = set(normalize_song_component(title).split())
    return " ".join(sorted(title_tokens & VERSION_TERMS))


def score_lyric_quality(lyrics: Any) -> float:
    """Score transcription completeness and cleanliness for representative selection."""
    text = (
        ("" if lyrics is None else str(lyrics))
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )
    tokens = WORD_RE.findall(text.casefold())
    if not tokens:
        return float("-inf")

    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    substantial_lines = [
        re.sub(r"\s+", " ", line.casefold())
        for line in nonempty_lines
        if len(WORD_RE.findall(line)) >= 3
    ]
    repeated_lines = len(substantial_lines) - len(set(substantial_lines))
    repetition_ratio = repeated_lines / max(len(substantial_lines), 1)
    unique_tokens = len(set(tokens))
    visible_chars = [char for char in text if not char.isspace()]
    alphanumeric_ratio = (
        sum(char.isalnum() for char in visible_chars) / max(len(visible_chars), 1)
    )
    placeholder_count = len(PLACEHOLDER_RE.findall(text))

    word_score = min(math.log1p(len(tokens)) / math.log1p(700), 1.0) * 40.0
    vocabulary_score = min(math.log1p(unique_tokens) / math.log1p(350), 1.0) * 25.0
    structure_score = min(len(nonempty_lines) / 60.0, 1.0) * 10.0
    character_score = alphanumeric_ratio * 10.0
    placeholder_penalty = min(placeholder_count * 3.0, 24.0)
    repetition_penalty = max(repetition_ratio - 0.30, 0.0) * 30.0

    return round(
        word_score
        + vocabulary_score
        + structure_score
        + character_score
        - placeholder_penalty
        - repetition_penalty,
        6,
    )


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
    return corpus[CANONICAL_CORPUS_COLUMNS]


def clean_corpus(
    corpus: pd.DataFrame,
    config: CorpusCleaningConfig | None = None,
) -> tuple[pd.DataFrame, CorpusCleaningReport]:
    """Filter weak metadata and retain one high-quality row per song identity.

    The stage is deliberately separate from :func:`prepare_corpus` so callers can
    inspect the normalized-but-unclean corpus and audit exactly what was removed.
    Explicitly named versions such as live, remix, or acoustic recordings remain
    separate when ``preserve_explicit_versions`` is enabled.
    """
    config = config or CorpusCleaningConfig()
    required = {"title", "artist", "lyrics"}
    missing = required - set(corpus.columns)
    if missing:
        raise ValueError(
            "Corpus cleaning requires columns: "
            + ", ".join(sorted(required))
            + f". Missing: {', '.join(sorted(missing))}."
        )

    working = corpus.copy()
    if "doc_id" not in working:
        working["doc_id"] = [f"song_{i:07d}" for i in range(len(working))]
    if "source" not in working:
        working["source"] = ""
    working["title"] = working["title"].fillna("").astype(str)
    working["artist"] = working["artist"].fillna("").astype(str)
    working["lyrics"] = working["lyrics"].fillna("").astype(str)
    working["lyrics_char_len"] = working["lyrics"].str.len()
    working["_original_order"] = range(len(working))
    working["_artist_key"] = working["artist"].map(normalize_song_component)
    working["_title_key"] = working["title"].map(normalize_song_component)

    input_rows = len(working)
    unknown_artist_rows_removed = 0
    if config.drop_unknown_artists:
        unknown_mask = working["_artist_key"].isin(UNKNOWN_ARTIST_KEYS)
        unknown_artist_rows_removed = int(unknown_mask.sum())
        working = working.loc[~unknown_mask].copy()

    missing_title_rows_removed = 0
    if config.drop_missing_titles:
        missing_title_mask = working["_title_key"].eq("")
        missing_title_rows_removed = int(missing_title_mask.sum())
        working = working.loc[~missing_title_mask].copy()

    exact_lyric_duplicate_groups = 0
    exact_lyric_rows_removed = 0
    if config.deduplicate_exact_lyrics and not working.empty:
        working["_lyric_key"] = working["lyrics"].map(_normalized_lyric_key)
        if config.preserve_explicit_versions:
            working["_version_key"] = working["title"].map(explicit_version_key)
            working["_exact_content_key"] = (
                working["_lyric_key"] + "\x1f" + working["_version_key"]
            )
        else:
            working["_exact_content_key"] = working["_lyric_key"]

        exact_counts = working["_exact_content_key"].value_counts()
        exact_lyric_duplicate_groups = int((exact_counts > 1).sum())
        before_exact = len(working)

        working["_artist_exact_frequency"] = working.groupby(
            ["_exact_content_key", "_artist_key"]
        )["_original_order"].transform("size")
        working["_title_artifact_penalty"] = working["title"].map(
            _title_artifact_penalty
        )
        working["_title_key_length"] = working["_title_key"].str.len()
        working = (
            working.sort_values(
                [
                    "_exact_content_key",
                    "_artist_exact_frequency",
                    "_title_artifact_penalty",
                    "_title_key_length",
                    "_original_order",
                ],
                ascending=[True, False, True, True, True],
                kind="stable",
            )
            .drop_duplicates("_exact_content_key", keep="first")
            .sort_values("_original_order", kind="stable")
        )
        exact_lyric_rows_removed = before_exact - len(working)

    song_identity_duplicate_groups = 0
    song_identity_rows_removed = 0
    if config.deduplicate_song_identities and not working.empty:
        working["_song_identity"] = (
            working["_artist_key"] + "\x1f" + working["_title_key"]
        )
        identity_counts = working["_song_identity"].value_counts()
        song_identity_duplicate_groups = int((identity_counts > 1).sum())
        before_identity = len(working)

        working["_lyric_quality"] = working["lyrics"].map(score_lyric_quality)
        working = (
            working.sort_values(
                [
                    "_song_identity",
                    "_lyric_quality",
                    "lyrics_char_len",
                    "_original_order",
                ],
                ascending=[True, False, False, True],
                kind="stable",
            )
            .drop_duplicates("_song_identity", keep="first")
            .sort_values("_original_order", kind="stable")
        )
        song_identity_rows_removed = before_identity - len(working)

    if working.empty:
        raise ValueError("No corpus rows remain after corpus cleaning.")

    working = working.reset_index(drop=True)
    working["doc_id"] = [f"song_{i:07d}" for i in range(len(working))]
    cleaned = working[CANONICAL_CORPUS_COLUMNS].copy()
    report = CorpusCleaningReport(
        input_rows=input_rows,
        unknown_artist_rows_removed=unknown_artist_rows_removed,
        missing_title_rows_removed=missing_title_rows_removed,
        exact_lyric_duplicate_groups=exact_lyric_duplicate_groups,
        exact_lyric_rows_removed=exact_lyric_rows_removed,
        song_identity_duplicate_groups=song_identity_duplicate_groups,
        song_identity_rows_removed=song_identity_rows_removed,
        output_rows=len(cleaned),
    )
    return cleaned, report


def _normalized_lyric_key(lyrics: Any) -> str:
    return re.sub(r"\s+", " ", str(lyrics).casefold()).strip()


def _title_artifact_penalty(title: Any) -> int:
    """Penalize obvious source-generated title suffixes during exact deduplication."""
    text = str(title).strip()
    penalty = 0
    if re.search(r"\.\s*\d+$", text):
        penalty += 3
    if re.search(r"[_-]+$", text):
        penalty += 2
    if re.search(r"\s{2,}", text):
        penalty += 1
    return penalty


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
