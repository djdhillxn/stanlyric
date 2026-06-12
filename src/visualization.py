"""Notebook-friendly visualizations for StanLyric retrieval."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_top_scores(results: pd.DataFrame, title: str = "Top BM25 matches", save_path: str | Path | None = None):
    """Plot top retrieved songs by BM25 score.

    This is useful because it shows whether the top candidate is clearly separated or
    whether the query is ambiguous across many songs.
    """
    if results.empty:
        raise ValueError("results DataFrame is empty")
    labels = []
    for _, row in results.iterrows():
        song = row.get("title", "") or row.get("doc_id", "")
        artist = row.get("artist", "")
        labels.append(f"{song}\n{artist}" if artist else str(song))

    fig, ax = plt.subplots(figsize=(10, max(4, 0.55 * len(results))))
    y = range(len(results))
    ax.barh(y, results["bm25_score"])
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("BM25 score")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=160)
    return fig, ax


def plot_metric_summary(metrics: pd.DataFrame, save_path: str | Path | None = None):
    """Plot core retrieval metrics from one evaluation run."""
    if metrics.empty:
        raise ValueError("metrics DataFrame is empty")
    row = metrics.iloc[0]
    metric_names = [m for m in ["hit@1", "hit@3", "hit@5", "hit@10", "mrr@10", "ndcg@10"] if m in row]
    values = [float(row[m]) for m in metric_names]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(metric_names, values)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("StanLyric BM25 retrieval metrics")
    for i, value in enumerate(values):
        ax.text(i, value + 0.02, f"{value:.3f}", ha="center")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=160)
    return fig, ax


def plot_rank_distribution(details: pd.DataFrame, save_path: str | Path | None = None):
    """Plot distribution of target ranks for found queries."""
    found = details[details["rank"].notna()].copy()
    if found.empty:
        raise ValueError("No target documents were found in the evaluated top-k results")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(found["rank"].astype(int), bins=range(1, int(found["rank"].max()) + 2), align="left")
    ax.set_xlabel("rank of correct song")
    ax.set_ylabel("number of queries")
    ax.set_title("Distribution of correct-song ranks")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=160)
    return fig, ax
