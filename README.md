# MergePatternMining

Pipeline for building **document-level hybrid AMR + CoBaLD graphs**, merging them with coreference-aware consolidation, and exporting **Gaston / gSofia** flat graph files for pattern mining.

This repository contains the **graph construction and export** code only. Classifiers and gSofia mining outputs live elsewhere.

## Pipeline overview

Run all commands from the **project root**.

| Step | Script | Input CSV columns | Output columns / artifacts |
|------|--------|-------------------|---------------------------|
| 1a. AMR parse | `amr/amr_parse.py` | `category`, `content` | `graph` → `data-temp/amr_{i}.pkl` (+ optional PNG) |
| 1b. CoBaLD parse | `cobald/cobald_parse.py` | `category`, `content` | `conllu` → `data-temp/cobald_{i}.pkl` (+ optional PNG) |
| 2. Enrich | `enrich/enrich.py` | `graph`, `conllu` | `hybrid` → `data-temp/hybrid_{i}.pkl` (+ optional PNG) |
| 3. Merge | `merge/merge.py` | `hybrid` | `merged` → `data-temp/merged_{i}.pkl` (+ optional PNG) |
| 4. Flat export | `flat/export_gaston.py` | `category`, `merged` | `flat/train/{amr,enr-node,cobald}/*.txt`, vocab JSON |

Each step updates the CSV in place (or `-o`) and supports **`--rewrite`** to recompute rows. Rows with filled target columns are skipped by default (resume-safe).

### Example (small test document)

```bash
python amr/amr_parse.py -i data/test-set.csv
python cobald/cobald_parse.py -i data/test-set.csv
python enrich/enrich.py -i data/test-set.csv
python merge/merge.py -i data/test-set.csv
python flat/export_gaston.py -i data/test-set.csv --train-ratio 0.8
```

### Example (BBC News corpus)

```bash
python amr/amr_parse.py -i data/bbc-news-data.csv --no-png
python cobald/cobald_parse.py -i data/bbc-news-data.csv --no-png
python enrich/enrich.py -i data/bbc-news-data.csv --no-png
python merge/merge.py -i data/bbc-news-data.csv --no-png
python flat/export_gaston.py -i data/bbc-news-data.csv
```

## Directory layout

```
├── data/                 # Input TSV/CSV (category, content, …)
├── data-temp/            # Runtime pickles (gitignored); PNG previews may be committed
├── data-research/        # One-off research figures (single sentence, diluted AMR, …)
├── amr/                  # AMR parsing + Graphviz render helpers
├── cobald/               # CoBaLD HF pipeline → NetworkX
├── enrich/               # Anchor-based AMR ← CoBaLD attribute projection
├── merge/                # FastCoref merge + name cleanup + triple dedup
├── flat/                 # Gaston flat export + vocab + optional test split .txt
├── model/                # amrlib StoG config stubs (download weights separately)
└── utils.py              # Shared CSV read/write
```

## Module details

### `amr/`

- **`amr_parse.py`** — Sentence-level AMR (amrlib StoG + RBW alignment), document union under a `document` node, PENMAN → NetworkX.
- **`amr_save.py`** — Pickle I/O and AMR Graphviz PNG rendering.

Default AMR model directory: `model/model_parse_xfm_bart_large-v0_1_0` (see [amrlib](https://github.com/bjascob/amrlib) for downloading full weights).

### `cobald/`

- **`cobald_parse.py`** — Hugging Face `token-classification` pipeline for `CoBaLD/xlm-roberta-base-cobald-parser`, document-level CoBaLD graph.
- **`graph_construct.py`** — Sentence graph nodes/edges from parser JSON.
- **`cobald_save.py`** — Pickle + Graphviz render for CoBaLD graphs.
- **`diagnose_cobald_row.py`** — Debug one CSV row / pipeline output.

### `enrich/`

- **`enrich.py`** — Maps CoBaLD `semclass` / `deepslot` onto AMR nodes by character `anchor` overlap.

### `merge/`

- **`merge_graph.py`** — Coref mapping (FastCoref + spaCy heads), entity/predicate hubs, WordNet hypernym lift, recursive triple deduplication.
- **`name_cleanup.py`** — Collapse duplicate `:name` branches and strip literal `/name` nodes after merge.
- **`merge.py`** — CLI over hybrid pickles → merged pickles.

Requires: `fastcoref`, `spacy` (`en_core_web_sm`), NLTK WordNet (`wordnet`, `omw-1.4`).

### `flat/`

- **`export_gaston.py`** — Main exporter: three label strategies (`amr`, `enr-node`, `cobald`), train/test split per class, global vocabularies.
- **`visualize_flat.py`** — Render one transaction from a `.txt` flat file.
- **`diagnose_gaston_input.py`** — Validate flat files before gSofia.
- **`strip_enr_nodes_by_label_id.py`** — Post-process enr-node flat files by label id.

**Committed under `flat/` for gSofia input:** `train/` and `test/` `*.txt`, `vocab_*.json`.  
**Not committed:** `flat/**/out/` (`.OUT`, run `.json` from gSofia).

## Flat label strategies

| Folder | Node label |
|--------|------------|
| `flat/train/amr/` | AMR concept / PropBank frame |
| `flat/train/enr-node/` | `(concept, semclass)` |
| `flat/train/cobald/` | CoBaLD `semclass` (fallback: AMR concept) |

Edges use AMR relation names in all strategies.

## Auxiliary scripts (not in the main pipeline)

| Script | Purpose |
|--------|---------|
| `visualize_graph.py` | PNG for any stage pickle (`--type amr\|cobald\|hybrid\|merged`) |
| `research_sentence_outputs.py` | Single-sentence AMR + CoBaLD for papers (`data/test-research.csv`) |
| `amr_diluted.py` | Research demo: naive merge of identical AMR triples across sentences |

## Models and dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

- **AMR:** place the full StoG checkpoint under `model/model_parse_xfm_bart_large-v0_1_0/` (JSON configs in-repo are not the weights).
- **CoBaLD:** downloaded from Hugging Face on first run (`CoBaLD/xlm-roberta-base-cobald-parser`).
- **Graphviz:** system install required for PNG/PDF rendering.

## What to commit / what to ignore

**Include**

- `amr/`, `cobald/`, `enrich/`, `merge/`, `flat/` (scripts + `train/`/`test/` `.txt` + vocab JSON)
- `utils.py`, `visualize_graph.py`, `requirements.txt`, `.gitignore`, this `README.md`
- `data/test-research.csv`, `data/test-set.csv` (small samples)
- `data/bbc-news-data.csv` (optional, ~5 MB full corpus)
- `data-research/` (paper examples: penman, conllu, PNG/PDF)
- `data-temp/**/*.png` only (preview images; pickles stay local)
- `model/model_parse_xfm_bart_large-v0_1_0/*.json` (config stubs)

**Exclude** (see `.gitignore`)

- `classifier/` — cloned [stable-AMR-graphs](https://github.com/ericparakal) repos
- `cobald_parser/`, `src/` — CoBaLD training code, not used at inference
- `data-temp/**/*.pkl` and other large intermediates
- `flat/**/out/`, `*.OUT` — gSofia outputs
- Local docs: `PARSE.md`, `ENRICH.md`, `MERGE.md`, `FLAT.md` (kept locally; pipeline is described here)
- `main.tex`, notebooks, `flat.zip`

## CSV format

Minimum input:

```text
category	content
business	Some document text…
```

After the pipeline, optional path columns are added: `graph`, `conllu`, `hybrid`, `merged` (relative paths into `data-temp/`). For a clean public clone, use `data/test-research.csv` (text only) and rerun the steps above.

## gSofia (external)

This repo stops at **flat graph files**. Run gSofia/Gaston separately on `flat/train/` (and `flat/test/` for hold-out), writing results under `flat/**/out/` — those directories are gitignored.
