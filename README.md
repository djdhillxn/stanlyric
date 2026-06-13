# StanLyric

**StanLyric** is an offline-first lyric fragment search and song identification
project. Give it a remembered line, even an incomplete one, and it ranks the most
likely songs with BM25.

The project follows the full retrieval story: selectively download lyric data,
prepare a canonical corpus, build and evaluate an index, search from Python or a
notebook, and export the same search engine as a static browser artifact for a
portfolio site.

## What It Does

- Discovers and selectively downloads likely lyric files from Hugging Face.
- Reads CSV, JSONL, JSON, Parquet, Pickle, and directories of lyric text files.
- Normalizes raw lyrics, audits duplicate song identities, and writes a clean canonical
  Parquet corpus.
- Builds a self-contained BM25-Okapi index.
- Returns ranked songs, matched terms, snippets, scores, and source information.
- Evaluates retrieval with synthetic lyric fragments and standard ranking metrics.
- Exports a static JSON index that runs entirely in the browser.
- Supports public metadata-only and local full-lyrics export modes.

## Repository Layout

```text
stanlyric/
|-- assets/
|   `-- json/stanlyric/             # portfolio-ready web index copies
|-- data/
|   |-- raw/                        # downloaded source data
|   |-- processed/
|   |   |-- corpus.parquet
|   |   |-- stanlyric_bm25.pkl
|   |   |-- stanlyric_web_index_without_lyrics.json
|   |   `-- stanlyric_web_index_with_lyrics.json
|   `-- sample_lyrics.csv           # tiny smoke-test corpus
|-- notebooks/
|   `-- 01_bm25_song_identifier.ipynb
|-- outputs/evaluation/             # evaluation tables and plots
|-- scripts/
|   |-- discover_hf_files.py
|   |-- download_data.py
|   |-- prepare_corpus.py
|   |-- build_index.py
|   |-- search.py
|   |-- evaluate.py
|   `-- export_stanlyric_web.py
|-- src/
|   |-- bm25.py
|   |-- data.py
|   |-- evaluation.py
|   |-- hf_download.py
|   |-- search.py
|   |-- text.py
|   `-- visualization.py
|-- tests/
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

## Setup

Run commands from the repository root so the scripts and notebook resolve the
local `src` modules directly:

```bash
cd stanlyric
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Restart the notebook kernel after changing source modules so imports reflect the
latest code.

## Quick Smoke Test

The included sample exercises corpus preparation, indexing, and search:

```bash
python -m scripts.prepare_corpus \
  --input data/sample_lyrics.csv \
  --output data/processed/sample_corpus.parquet

python -m scripts.build_index \
  --corpus data/processed/sample_corpus.parquet \
  --output data/processed/sample_bm25.pkl

python -m scripts.search \
  --index data/processed/sample_bm25.pkl \
  --query "one shot one opportunity seize everything" \
  --top-k 3
```

## Main Workflow

The Lyrics-MIDI repository is large, so begin by inspecting likely lyric files
instead of downloading the complete repository:

```bash
python -m scripts.discover_hf_files --top 25
python -m scripts.download_data --dry-run
```

Download the highest-ranked candidate, or select a file reported by the dry run:

```bash
python -m scripts.download_data

python -m scripts.download_data \
  --filename "PATH/SHOWN/BY/DRY_RUN.parquet"
```

Prepare the canonical corpus and build the index:

```bash
python -m scripts.prepare_corpus \
  --input data/raw/YOUR_DOWNLOADED_FILE \
  --output data/processed/corpus.parquet

python -m scripts.build_index \
  --corpus data/processed/corpus.parquet \
  --output data/processed/stanlyric_bm25.pkl
```

Corpus preparation is deliberately split into two stages. `prepare_corpus`
normalizes the source into the six-column StanLyric schema without hiding duplicate
records. `clean_corpus` then:

- removes blank/unknown artists and blank titles;
- collapses exact lyric copies using the most credible artist/title label;
- groups case-, punctuation-, and accent-normalized artist-title identities;
- keeps the most complete, least malformed transcription in each identity group;
- preserves explicitly named live, remix, acoustic, demo, and similar versions.

For the current Lyrics-MIDI source file, this reduces 47,474 prepared rows to
36,545 clean song documents. The CLI prints an audit report for every removal
category before saving `corpus.parquet`.

The cleaning policy can be relaxed for experiments:

```bash
python -m scripts.prepare_corpus \
  --input data/raw/YOUR_DOWNLOADED_FILE \
  --keep-unknown-artists \
  --keep-missing-titles \
  --no-exact-lyric-deduplication \
  --no-song-identity-deduplication
```

Search and evaluate it:

```bash
python -m scripts.search \
  --index data/processed/stanlyric_bm25.pkl \
  --query "look if you had one shot or one opportunity" \
  --top-k 10

python -m scripts.evaluate \
  --index data/processed/stanlyric_bm25.pkl \
  --n-queries 500 \
  --top-k 10
```

Evaluation tables and plots are written to `outputs/evaluation/`.

## Notebook Workflow

Open `notebooks/01_bm25_song_identifier.ipynb` from the repository root. It walks
through dataset discovery, selective download, corpus preparation, an explicit
deduplication audit, index building, search explanation, visualization, and
synthetic evaluation.

Search results include:

- rank, document ID, title, and artist;
- BM25 score and corpus score percentile;
- matched terms and a lyric snippet;
- source and retrieval method;
- full lyrics when explicitly requested.

The reported `confidence` is a top-k relative softmax score. It is useful for
comparing candidates for one query, but it is not a calibrated probability.

## Evaluation

StanLyric samples fragments from known songs, hides the source song ID, retrieves
the top candidates, and measures where the original song reappears.

The evaluation reports Hit@1, Hit@3, Hit@5, Hit@10, Recall@10, MRR@10, NDCG@10,
median found rank, and miss rate. In this one-relevant-song setup, Recall@10 and
Hit@10 are equivalent.

## Browser Portfolio Export

`export_stanlyric_web.py` packages the corpus and BM25 structures into a static
JSON artifact. A portfolio page can fetch that file and run lyric-fragment search
entirely in the browser, with no search server.

Export the recommended metadata-only version:

```bash
python -m scripts.export_stanlyric_web \
  --corpus data/processed/corpus.parquet \
  --output data/processed/stanlyric_web_index.json
```

The exporter makes the content mode explicit in the final filename:

- `stanlyric_web_index_without_lyrics.json` is the default.
- `stanlyric_web_index_with_lyrics.json` is produced with
  `--include-full-lyrics`.

To generate the local full-lyrics version:

```bash
python -m scripts.export_stanlyric_web \
  --corpus data/processed/corpus.parquet \
  --output data/processed/stanlyric_web_index.json \
  --include-full-lyrics
```

### GitHub Pages Integration

The portfolio can keep its page, styling, search code, and exported data separated:

```text
your-github-io/
|-- _projects/stanlyric.md
`-- assets/
    |-- css/stanlyric/stanlyric.css
    |-- js/stanlyric/stanlyric.js
    `-- json/stanlyric/
        |-- stanlyric_web_index_without_lyrics.json
        `-- stanlyric_eval_metrics.json          # optional
```

Export directly into the portfolio repository:

```bash
python -m scripts.export_stanlyric_web \
  --corpus data/processed/corpus.parquet \
  --output /path/to/your-github-io/assets/json/stanlyric/stanlyric_web_index.json \
  --metrics outputs/evaluation/metrics.csv \
  --metrics-output /path/to/your-github-io/assets/json/stanlyric/stanlyric_eval_metrics.json
```

The metrics arguments are optional, and the exporter accepts metrics stored as
CSV, JSON, or Parquet. The portfolio JavaScript must fetch the explicit generated
filename, normally:

```text
assets/json/stanlyric/stanlyric_web_index_without_lyrics.json
```

The browser UI can use the artifact to show the top candidate, ranked result
cards, BM25 scores, relative confidence, corpus percentile, first-to-second score
gap, matched and missing terms, per-term BM25 contributions, a score chart, and
optional offline metrics. Snippets and collapsible lyrics require the
`with_lyrics` artifact.

For a Jekyll portfolio, verify the integration locally:

```bash
bundle exec jekyll serve
```

Then open `/projects/stanlyric/`. A small synthetic fallback artifact can be useful
during portfolio development, but it is optional and separate from StanLyric's
generated data.

## Export Modes

Use the `without_lyrics` artifact for a compact public portfolio: it contains the
search structures and essential result metadata, but not lyric text. Use the
`with_lyrics` artifact for local or private development when snippets and full
lyrics are needed.

The source dataset is the
[Lyrics-MIDI-Dataset](https://huggingface.co/datasets/asigalov61/Lyrics-MIDI-Dataset),
listed as CC BY-NC-SA 4.0. Review dataset and lyric redistribution terms before
publishing generated artifacts.

## Planned Improvements

- Character n-gram TF-IDF for typo-heavy or misremembered fragments.
- Dense sentence-transformer retrieval using available embeddings.
- BM25 and dense hybrid reranking.
- A Spotify bridge for lyric-aware recommendations beyond an existing playlist.
