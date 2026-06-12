"""Text normalization and tokenization utilities for lyric retrieval."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List

_BRACKETED_SECTION_RE = re.compile(r"\[(verse|chorus|bridge|intro|outro|hook|pre-chorus).*?\]", re.I)
_NON_WORD_RE = re.compile(r"[^a-z0-9'\s]+")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str, keep_apostrophes: bool = True) -> str:
    """Normalize lyric text for BM25 indexing and search.

    The normalization is intentionally conservative. We remove common section headers,
    lowercase text, normalize unicode, and collapse whitespace. Apostrophes are kept by
    default because lyric fragments often contain words like "don't" and "I'm".
    """
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = _BRACKETED_SECTION_RE.sub(" ", text)
    text = text.lower()
    if keep_apostrophes:
        text = _NON_WORD_RE.sub(" ", text)
    else:
        text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> List[str]:
    """Tokenize normalized text into word tokens."""
    text = normalize_text(text)
    return text.split() if text else []


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    """Return unique strings while preserving original order."""
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
