"""
Per-sentence CoBaLD probe for one CSV row (same NLTK + HF pipeline as cobald_parse).

From project root:
  python cobald/diagnose_cobald_row.py -i data/bbc-news-data.csv --row-index 86
"""

from __future__ import annotations

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import utils  # noqa: E402
from nltk.tokenize import sent_tokenize  # noqa: E402

import cobald_parse  # noqa: E402


def _abspath(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def main() -> None:
    default_model = "CoBaLD/xlm-roberta-base-cobald-parser"
    parser = argparse.ArgumentParser(description="Find first CoBaLD-failing sentence in one row")
    parser.add_argument("-i", "--input", required=True, help="Input TSV/CSV (utils.read_documents_csv)")
    parser.add_argument("--row-index", type=int, default=None, help="Pandas row label (index after read)")
    parser.add_argument("--filename", type=str, default=None, help="Match e.g. 087.txt (first match if set)")
    parser.add_argument("--model", type=str, default=default_model)
    parser.add_argument("--max-chars", type=int, default=320, help="Preview length per sentence")
    args = parser.parse_args()

    inp = _abspath(args.input)
    df = utils.read_documents_csv(inp)

    if args.filename:
        m = df["filename"].astype(str) == args.filename
        if not m.any():
            print(f"No row with filename={args.filename!r}", file=sys.stderr)
            sys.exit(1)
        hit_idx = df.loc[m].index
        idx = int(hit_idx[0])
        if len(hit_idx) > 1:
            print(f"Note: {len(hit_idx)} rows match filename; using first index {idx}")
    elif args.row_index is not None:
        idx = args.row_index
        if idx not in df.index:
            print(f"Row index {idx} not in dataframe index", file=sys.stderr)
            sys.exit(1)
    else:
        print("Provide --row-index or --filename", file=sys.stderr)
        sys.exit(1)

    row = df.loc[idx]
    content = str(row.get("content", "")).strip()
    title = str(row.get("title", ""))
    fn = str(row.get("filename", ""))
    cat = str(row.get("category", ""))

    cobald_parse.ensure_nltk_punkt()
    sentences = sent_tokenize(content, "english")
    print(f"row_index={idx} category={cat!r} filename={fn!r} title={title!r}")
    print(f"content_chars={len(content)} sentences={len(sentences)} model={args.model}")
    print("---")

    pipe = cobald_parse.get_cobald_pipeline(args.model)

    for si, s in enumerate(sentences):
        preview = s.strip().replace("\n", " ")
        if len(preview) > args.max_chars:
            preview = preview[: args.max_chars] + "..."
        try:
            out = pipe([s])
            cobald_parse.normalize_cobald_sentence_annotation(out[0])
            print(f"OK   snt[{si}] len={len(s)} words~={len(s.split())} | {preview!r}")
        except Exception as e:
            print(f"FAIL snt[{si}] len={len(s)} | {preview!r}")
            print(f"       {type(e).__name__}: {e}")
            sys.exit(1)

    print("---")
    print("All sentences passed CoBaLD when run one-by-one.")


if __name__ == "__main__":
    main()
