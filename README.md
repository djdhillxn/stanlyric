# StanLyric

**StanLyric** is an offline-first lyric fragment search and song identification project.

The v1 goal is simple:

> Type a few lyric lines or a partial remembered lyric, and StanLyric retrieves the most likely songs using BM25.

The project is intentionally separate from the Spotify dashboard. Later, it can become the lyrics intelligence module behind Spotify-based playlist insights and lyric-aware recommendations.

## Current scope: v1 BM25 song identifier

StanLyric currently supports:

- selective Hugging Face dataset discovery and download, avoiding MIDI/audio/huge ZIP files by default;
- corpus preparation from CSV, JSONL, JSON, Parquet, Pickle, or directories of `.txt` lyric files;
- a self-contained BM25-Okapi implementation;
- top-k lyric fragment search;
- result snippets, matched terms, confidence-like scores, and source metadata;
- offline full-lyrics access for development notebooks;
- synthetic IR evaluation with Hit@1, Hit@3, Hit@5, Hit@10, MRR@10, NDCG@10, Recall@10, and miss rate;
- notebook-friendly visualizations for score gaps, metric summaries, and rank distributions.

## Repository layout

```text
stanlyric/
├── data/
│   ├── raw/                    # downloaded HF files, ignored by git
│   ├── processed/              # corpus.parquet and BM25 index, ignored by git
│   └── sample_lyrics.csv        # tiny smoke-test sample
├── notebooks/
│   └── 01_bm25_song_identifier.ipynb
├── outputs/                    # evaluation plots/results, ignored by git
├── scripts/
│   ├── discover_hf_files.py
│   ├── download_data.py
│   ├── prepare_corpus.py
│   ├── build_index.py
│   ├── search.py
│   ├── evaluate.py
│   ├── export_stanlyric_web.py
│   └── export_song_metadata.py
├── src/stanlyric/
│   ├── bm25.py
│   ├── data.py
│   ├── evaluation.py
│   ├── hf_download.py
│   ├── search.py
│   ├── text.py
│   └── visualization.py
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Setup

```bash
cd stanlyric
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For Colab, upload or clone the repo, then run:

```python
!pip install -e .
```

## Quick smoke test with the included tiny sample

```bash
python scripts/prepare_corpus.py \
  --input data/sample_lyrics.csv \
  --output data/processed/sample_corpus.parquet

python scripts/build_index.py \
  --corpus data/processed/sample_corpus.parquet \
  --output data/processed/sample_bm25.pkl

python scripts/search.py \
  --index data/processed/sample_bm25.pkl \
  --query "one shot one opportunity seize everything" \
  --top-k 3
```

## Main workflow with the Hugging Face dataset

The linked Lyrics-MIDI dataset is large, so do not blindly download the entire repo or the main archive. First inspect likely lyric files:

```bash
python scripts/discover_hf_files.py --top 25
```

Then dry-run the downloader:

```bash
python scripts/download_data.py --dry-run
```

Download the top-ranked candidate:

```bash
python scripts/download_data.py
```

Or download a specific candidate shown by the dry run:

```bash
python scripts/download_data.py --filename "PATH/SHOWN/BY/DRY_RUN.parquet"
```

Prepare the canonical corpus:

```bash
python scripts/prepare_corpus.py \
  --input data/raw/YOUR_DOWNLOADED_FILE \
  --output data/processed/corpus.parquet
```

Build the BM25 index:

```bash
python scripts/build_index.py \
  --corpus data/processed/corpus.parquet \
  --output data/processed/stanlyric_bm25.pkl
```

Search:

```bash
python scripts/search.py \
  --index data/processed/stanlyric_bm25.pkl \
  --query "look if you had one shot or one opportunity" \
  --top-k 10
```

Evaluate:

```bash
python scripts/evaluate.py \
  --index data/processed/stanlyric_bm25.pkl \
  --n-queries 500 \
  --top-k 10
```

Evaluation artifacts are saved to `outputs/evaluation/`.

## Portfolio exports

`export_stanlyric_web.py` builds the static BM25 artifact used by a browser-based
portfolio app:

```bash
python scripts/export_stanlyric_web.py \
  --corpus data/processed/corpus.parquet \
  --output assets/json/stanlyric/stanlyric_web_index.json
```

The output name always states whether lyric text is present:

- `stanlyric_web_index_without_lyrics.json` by default
- `stanlyric_web_index_with_lyrics.json` with `--include-full-lyrics`

The browser app should load the matching explicit filename. Evaluation metrics can
optionally be exported with `--metrics` and `--metrics-output`.

Export the separate song metadata lookup with:

```bash
python scripts/export_song_metadata.py \
  --corpus data/processed/corpus.parquet \
  --source-data data/raw/Lyrics_MIDI_Dataset_Processed_Corpus_CC_BY_NC_SA.pickle \
  --output assets/json/stanlyric/song_metadata.json
```

Each search document includes a `metadata_key`. Use it to read
`song_metadata.json["songs"][metadata_key]`. The lookup currently contains title,
artist, source ID, document ID, and available lyric keywords. The stable source ID
is preferred; `doc_id` is used only when a source ID is unavailable. Album, year,
artwork, playlist membership, and external service IDs can be added later without
rebuilding the BM25 artifact.

## Notebook workflow

Open:

```text
notebooks/01_bm25_song_identifier.ipynb
```

The notebook walks through:

1. Hugging Face file inspection;
2. selective download;
3. corpus preparation;
4. BM25 index building;
5. lyric-fragment search;
6. result explanation;
7. score visualization;
8. synthetic benchmark evaluation.

## Result fields

Search returns a DataFrame with columns like:

- `rank`
- `doc_id`
- `title`
- `artist`
- `bm25_score`
- `confidence`
- `score_percentile`
- `matched_terms`
- `snippet`
- `method`
- `source`
- `full_lyrics`, only if `include_full_lyrics=True`

`confidence` is not a calibrated probability. It is a top-k relative softmax score useful for comparing retrieved candidates for one query.

## Evaluation metrics

StanLyric v1 uses synthetic retrieval evaluation:

1. sample lyric fragments from known songs;
2. hide the source song ID;
3. retrieve top-k songs;
4. check whether the original song appears in the result list.

Metrics:

- **Hit@1**: fraction of queries where the correct song is rank 1.
- **Hit@5 / Hit@10**: fraction where the correct song appears in top 5 / top 10.
- **MRR@10**: average reciprocal rank of the correct song within top 10.
- **NDCG@10**: ranking-quality metric; with one known relevant song, higher means the target appears closer to rank 1.
- **Recall@10**: same as Hit@10 in this one-target setup.

## Development note on full lyrics

The notebook can keep full lyrics in variables for offline development and debugging. For any future public portfolio demo, prefer showing short matched snippets and metadata rather than publishing full lyric text.

## Planned v2 upgrades

- character n-gram TF-IDF for typo-heavy or misremembered lyric fragments;
- dense sentence-transformer retrieval using the dataset's existing embeddings;
- BM25 + dense hybrid reranking;
- Spotify playlist bridge: recommend songs absent from a playlist but lyrically similar to the user's liked tracks.
