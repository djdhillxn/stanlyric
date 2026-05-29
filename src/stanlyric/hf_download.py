"""Selective Hugging Face dataset discovery and download helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download


DEFAULT_REPO_ID = "asigalov61/Lyrics-MIDI-Dataset"
DEFAULT_REPO_TYPE = "dataset"

LYRIC_DATA_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".ndjson",
    ".parquet",
    ".pq",
    ".pkl",
    ".pickle",
    ".txt",
}
HEAVY_OR_IRRELEVANT_EXTENSIONS = {
    ".mid",
    ".midi",
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".npy",
    ".npz",
    ".safetensors",
    ".pt",
}


@dataclass
class CandidateFile:
    path: str
    size_bytes: Optional[int]
    score: float
    reason: str

    @property
    def size_mb(self) -> Optional[float]:
        if self.size_bytes is None:
            return None
        return self.size_bytes / (1024 * 1024)


def list_repo_files_with_sizes(repo_id: str = DEFAULT_REPO_ID, repo_type: str = DEFAULT_REPO_TYPE):
    """Return Hugging Face repo siblings with filename and optional file size."""
    api = HfApi()
    info = api.repo_info(repo_id=repo_id, repo_type=repo_type, files_metadata=True)
    return info.siblings


def rank_lyric_files(
    repo_id: str = DEFAULT_REPO_ID,
    repo_type: str = DEFAULT_REPO_TYPE,
    max_size_mb: float = 900.0,
) -> list[CandidateFile]:
    """Rank likely lyric/metadata files while avoiding MIDI/audio/huge archives.

    The linked dataset is evolving, so this function uses filename heuristics instead
    of hard-coding one path. It prefers clean/deduped/Genius lyric tables and avoids
    .mid, audio, embedding arrays, and massive archives by default.
    """
    siblings = list_repo_files_with_sizes(repo_id, repo_type)
    candidates: list[CandidateFile] = []

    for sibling in siblings:
        path = sibling.rfilename
        lower = path.lower()
        suffix = Path(path).suffix.lower()
        size = getattr(sibling, "size", None)
        size_mb = size / (1024 * 1024) if size else None

        if suffix in HEAVY_OR_IRRELEVANT_EXTENSIONS:
            continue
        if size_mb is not None and size_mb > max_size_mb:
            continue
        if suffix not in LYRIC_DATA_EXTENSIONS:
            continue

        score = 0.0
        reasons = []
        for token, weight in [
            ("lyrics", 8),
            ("lyric", 8),
            ("genius", 6),
            ("clean", 5),
            ("dedup", 5),
            ("corpus", 4),
            ("metadata", 3),
            ("songs", 3),
            ("standard", 2),
        ]:
            if token in lower:
                score += weight
                reasons.append(token)
        for token, penalty in [
            ("summary", 4),
            ("summaries", 4),
            ("embedding", 8),
            ("self_similarity", 8),
            ("chords", 3),
            ("midi", 2),
        ]:
            if token in lower:
                score -= penalty
        if suffix in {".parquet", ".pq", ".csv", ".jsonl", ".pkl", ".pickle"}:
            score += 3
        if size_mb is not None:
            # Prefer manageable files. Very tiny files are often docs/manifests.
            if 1 <= size_mb <= max_size_mb:
                score += 2
            elif size_mb < 0.05:
                score -= 2

        if score > 0:
            candidates.append(CandidateFile(path, size, score, ", ".join(reasons) or "heuristic"))

    candidates.sort(key=lambda c: (c.score, -(c.size_bytes or 0)), reverse=True)
    return candidates


def candidates_to_dataframe(candidates: Iterable[CandidateFile]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "path": c.path,
                "size_mb": None if c.size_mb is None else round(c.size_mb, 2),
                "score": round(c.score, 2),
                "reason": c.reason,
            }
            for c in candidates
        ]
    )


def download_file(
    filename: str,
    repo_id: str = DEFAULT_REPO_ID,
    repo_type: str = DEFAULT_REPO_TYPE,
    output_dir: str | Path = "data/raw",
) -> Path:
    """Download one selected file from Hugging Face into output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=filename,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
    )
    return Path(local_path)


def auto_download_best_candidate(
    repo_id: str = DEFAULT_REPO_ID,
    repo_type: str = DEFAULT_REPO_TYPE,
    output_dir: str | Path = "data/raw",
    max_size_mb: float = 900.0,
    rank: int = 0,
) -> tuple[Path, list[CandidateFile]]:
    """Rank lyric files and download the selected candidate."""
    candidates = rank_lyric_files(repo_id=repo_id, repo_type=repo_type, max_size_mb=max_size_mb)
    if not candidates:
        raise RuntimeError(
            "No suitable small lyric/metadata file found. Run scripts/discover_hf_files.py "
            "to inspect available files, or pass --filename manually."
        )
    if rank >= len(candidates):
        raise IndexError(f"rank={rank} but only {len(candidates)} candidate files were found")
    local_path = download_file(candidates[rank].path, repo_id, repo_type, output_dir)
    return local_path, candidates
