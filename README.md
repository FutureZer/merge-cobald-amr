# MergePatternMining

Pipeline for building **document-level hybrid AMR + CoBaLD graphs**, merging them with coreference-aware consolidation, and exporting **Gaston / gSofia** (https://github.com/AlekseyBuzmakov/FCAPS/tree/master/FCAPS/EXAMPLES/gSofia) flat graph files for pattern mining.

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

## Models and dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

- **AMR:** place the full AMR model in `model/` folder. For example, `model/model_parse_xfm_bart_large-v0_1_0/`.
- **CoBaLD:** downloaded from Hugging Face on first run (`CoBaLD/xlm-roberta-base-cobald-parser`). Also sourse code required in project root https://github.com/CobaldAnnotation/CobaldParser
- **Graphviz:** system install required for PNG/PDF rendering.
