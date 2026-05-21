"""
CSV (TSV) -> CoBaLD (HF pipeline) -> merged document graph -> data-temp (.pkl, .png) -> ``conllu`` column.

Run from project root:
  python cobald/cobald_parse.py
  python cobald/cobald_parse.py --debug -i data/test-set.csv
"""

from __future__ import annotations

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cobald_save  # noqa: E402
import utils  # noqa: E402
from graph_construct import build_cobald_edges, build_cobald_nodes  # noqa: E402

import networkx as nx
import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize

_PIPE_CACHE: dict[str, object] = {}


def _abspath(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def ensure_nltk_punkt() -> None:
    nltk.download("punkt", quiet=True)
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass


def sentence_char_spans(full_text: str, sentences: list[str]) -> list[tuple[int, int]]:
    """Map each sentence string to [start, end) character offsets in ``full_text``."""
    text = full_text.strip()
    spans: list[tuple[int, int]] = []
    search = 0
    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            spans.append((search, search))
            continue
        j = text.find(s_clean, search)
        if j < 0:
            j = search
        end = j + len(s_clean)
        spans.append((j, end))
        search = end
    return spans


def token_char_spans_in_sentence(sentence_text: str, words: list[str]) -> list[tuple[int, int]]:
    """Approximate [start, end) spans of each token within the sentence text."""
    spans: list[tuple[int, int]] = []
    pos = 0
    st = sentence_text
    for w in words:
        j = st.find(w, pos)
        if j < 0:
            spans.append((pos, pos))
        else:
            spans.append((j, j + len(w)))
            pos = j + len(w)
    return spans


def normalize_cobald_sentence_annotation(raw: object) -> dict:
    """
    HF / CoBaLD pipeline may return a sentence-level dict, or wrap it in a list/tuple
    (e.g. ``[annotation_dict]``). Reduce to the dict expected by ``graph_construct``.
    """
    x: object = raw
    for _ in range(32):
        if isinstance(x, tuple):
            x = list(x)
        if isinstance(x, list):
            if len(x) == 0:
                raise ValueError("CoBaLD pipeline returned an empty list for a sentence.")
            if len(x) == 1:
                x = x[0]
                continue
            for item in x:
                if isinstance(item, dict) and "words" in item and "deps_eud" in item:
                    return item
            raise ValueError(
                "CoBaLD pipeline returned a multi-element list without a sentence-level dict. "
                f"First element type: {type(x[0])!r}."
            )
        break

    if not isinstance(x, dict):
        raise ValueError(f"CoBaLD annotation must be a dict after unwrapping, got {type(x)!r}.")

    if "words" not in x or "deps_eud" not in x:
        raise ValueError(
            "CoBaLD dict missing 'words' or 'deps_eud'. "
            f"Keys present: {list(x.keys())[:40]!r}."
        )
    return x


def sentence_annotation_to_graph(
    ann: dict,
    sent_idx: int,
    sent_char_start: int,
) -> nx.DiGraph:
    """
    Build NetworkX graph from one CoBaLD pipeline dict; node/edge attrs per PARSE.md
    (lemma, deepslot, semclass, sent_idx, idx, anchor; edges: relation).
    """
    sentence_text = ann.get("text", "")
    words = ann.get("words", [])
    ids = ann.get("ids", [])
    lemmas = ann.get("lemmas", [])
    deepslots = ann.get("deepslots", [])
    semclasses = ann.get("semclasses", [])

    G = build_cobald_nodes(ann)
    G = build_cobald_edges(G, ann)

    local_spans = token_char_spans_in_sentence(sentence_text, words)

    for i, cob_id in enumerate(ids):
        nid = str(cob_id)
        if nid not in G.nodes:
            continue
        lo, hi = local_spans[i] if i < len(local_spans) else (0, 0)
        anchor = [sent_char_start + lo, sent_char_start + hi]
        G.nodes[nid]["lemma"] = lemmas[i] if i < len(lemmas) else ""
        G.nodes[nid]["deepslot"] = deepslots[i] if i < len(deepslots) else "_"
        G.nodes[nid]["semclass"] = semclasses[i] if i < len(semclasses) else "_"
        G.nodes[nid]["sent_idx"] = sent_idx
        G.nodes[nid]["idx"] = i
        G.nodes[nid]["anchor"] = anchor
        G.nodes[nid].pop("ds", None)
        G.nodes[nid].pop("sc", None)

    if "0" in G.nodes:
        G.nodes["0"]["lemma"] = "<ROOT>"
        G.nodes["0"]["deepslot"] = "<ROOT>"
        G.nodes["0"]["semclass"] = "<ROOT>"
        G.nodes["0"]["sent_idx"] = sent_idx
        G.nodes["0"]["idx"] = -1
        G.nodes["0"]["anchor"] = None
        G.nodes["0"].pop("ds", None)
        G.nodes["0"].pop("sc", None)

    for _u, _v, ed in G.edges(data=True):
        ed["relation"] = ed.get("eud_rel", "")

    return G


def merge_cobald_document_graph(graphs: list[nx.DiGraph]) -> nx.DiGraph:
    """``document`` node and ``:snt1``, ``:snt2``, ... to each sentence ROOT (node ``0``)."""
    out = nx.DiGraph()
    out.add_node(
        "document",
        lemma="document",
        deepslot="document",
        semclass="document",
        sent_idx=-1,
        idx=-1,
        anchor=None,
        token="DOCUMENT",
    )
    for si, g in enumerate(graphs):
        pfx = f"s{si}_"
        m = {n: pfx + str(n) if not str(n).startswith(pfx) else str(n) for n in g.nodes()}

        for n, data in g.nodes(data=True):
            out.add_node(m[n], **dict(data))

        for u, v, ed in g.edges(data=True):
            out.add_edge(m[u], m[v], **dict(ed))

        out.add_edge("document", m["0"], relation=f":snt{si + 1}")

    return out


def get_cobald_pipeline(model_id: str):
    if model_id in _PIPE_CACHE:
        return _PIPE_CACHE[model_id]

    from nltk.tokenize import word_tokenize
    from transformers import pipeline

    ensure_nltk_punkt()
    sentenizer = lambda text: sent_tokenize(text, "english")
    tokenizer = lambda sentence: word_tokenize(sentence, preserve_line=True)

    pipe = pipeline(
        "token-classification",
        model=model_id,
        trust_remote_code=True,
        sentenizer=sentenizer,
        tokenizer=tokenizer,
    )
    _PIPE_CACHE[model_id] = pipe
    return pipe


def build_document_cobald(text: str, *, model_id: str, debug: bool) -> nx.DiGraph:
    ensure_nltk_punkt()
    full = text.strip()
    sentences = sent_tokenize(full, "english")
    if not sentences:
        raise ValueError("No sentences after NLTK segmentation.")

    spans = sentence_char_spans(full, sentences)
    if debug:
        print(f"[debug] CoBaLD sentences: {len(sentences)}")

    pipe = get_cobald_pipeline(model_id)
    result = pipe(sentences)
    if len(result) != len(sentences):
        raise ValueError(f"CoBaLD returned {len(result)} graphs for {len(sentences)} sentences.")

    graphs: list[nx.DiGraph] = []
    for i, raw in enumerate(result):
        ann = normalize_cobald_sentence_annotation(raw)
        if not str(ann.get("text", "")).strip():
            ann = {**ann, "text": sentences[i]}
        g = sentence_annotation_to_graph(ann, i, spans[i][0] if i < len(spans) else 0)
        graphs.append(g)
        if debug:
            print(f"[debug] CoBaLD snt {i}: {len(ann.get('words', []))} tokens")

    merged = merge_cobald_document_graph(graphs)
    if debug:
        print(
            f"[debug] CoBaLD document graph: nodes={merged.number_of_nodes()}, "
            f"edges={merged.number_of_edges()}"
        )
    return merged


def process_dataframe(
    df: pd.DataFrame,
    *,
    model_id: str,
    data_temp: str,
    project_root: str,
    debug: bool,
    render_png: bool,
    path_style: str = "relative",
    rewrite: bool = False,
    checkpoint_csv: str | None = None,
) -> pd.DataFrame:
    """
    For each row: build CoBaLD graph, save artifacts, fill ``conllu`` column (path to .pkl).
    If ``rewrite`` is False, rows with a non-empty ``conllu`` cell are skipped.
    If ``checkpoint_csv`` is set, the table is written after each row that updates ``conllu``.
    """
    out = df.copy()
    if "conllu" not in out.columns:
        out["conllu"] = ""

    data_temp = _abspath(data_temp)
    os.makedirs(data_temp, exist_ok=True)
    project_root = _abspath(project_root)
    ckpt = _abspath(checkpoint_csv) if checkpoint_csv else None

    def _flush_checkpoint() -> None:
        if ckpt:
            utils.write_documents_csv(out, ckpt)

    n_rows = len(out)
    for pos, (i, row) in enumerate(out.iterrows()):
        print(f"[progress] {pos + 1}/{n_rows} (row index {i})", flush=True)
        content = str(row.get("content", "")).strip()
        if not content:
            out.loc[i, "conllu"] = ""
            if debug:
                print(f"[debug] Empty content, row index {i}")
            _flush_checkpoint()
            continue

        if not rewrite and utils.csv_cell_has_value(row.get("conllu")):
            if debug:
                print(f"[debug] Skip row {i}: conllu column already set")
            continue

        pkl_path = os.path.join(data_temp, f"cobald_{i}.pkl")
        png_path = os.path.join(data_temp, f"cobald_{i}.png")

        graph = build_document_cobald(content, model_id=model_id, debug=debug)

        bundle = {
            "graph": graph,
            "category": row.get("category", ""),
            "content": content,
            "filename": row.get("filename", ""),
            "title": row.get("title", ""),
        }
        cobald_save.save_graph_pickle(bundle, pkl_path)

        if render_png:
            try:
                cobald_save.render_cobald_graph(graph, png_path, title=f"CoBaLD cobald_{i}")
            except (ImportError, OSError, RuntimeError) as e:
                print(f"Warning: CoBaLD PNG not written ({e}).", file=sys.stderr)

        abs_pkl = _abspath(pkl_path)
        if path_style == "absolute":
            path_str = abs_pkl
        else:
            try:
                path_str = os.path.relpath(abs_pkl, project_root)
            except ValueError:
                path_str = abs_pkl
        out.loc[i, "conllu"] = os.path.normpath(path_str)
        if debug:
            print(f"[debug] conllu[{i}] = {out.loc[i, 'conllu']}")
        _flush_checkpoint()

    return out


def main() -> None:
    default_input = os.path.join(PROJECT_ROOT, "data", "test-set.csv")
    default_out_dir = os.path.join(PROJECT_ROOT, "data-temp")
    default_hf_model = "CoBaLD/xlm-roberta-base-cobald-parser"

    parser = argparse.ArgumentParser(
        description="Document CoBaLD -> NetworkX -> data-temp + conllu column"
    )
    parser.add_argument("-i", "--input", type=str, default=default_input, help="Input CSV (pandas)")
    parser.add_argument(
        "-o",
        "--output-csv",
        type=str,
        default=None,
        help="Output table with conllu column (default: overwrite input)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=default_hf_model,
        help="Hugging Face model id for CoBaLD parser",
    )
    parser.add_argument(
        "--data-temp",
        type=str,
        default=default_out_dir,
        help="Directory for .pkl and PNG",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Re-parse every row; default: only rows with empty conllu column",
    )
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument(
        "--path-style",
        choices=("relative", "absolute"),
        default="relative",
        help="Path style for the conllu column",
    )
    args = parser.parse_args()

    inp = _abspath(args.input)
    if not os.path.isfile(inp):
        print(f"File not found: {inp}", file=sys.stderr)
        sys.exit(1)

    if args.debug:
        print(f"[debug] HF model: {args.model}")

    out_csv = _abspath(args.output_csv) if args.output_csv else inp

    df = utils.read_documents_csv(inp)
    df_out = process_dataframe(
        df,
        model_id=args.model,
        data_temp=args.data_temp,
        project_root=PROJECT_ROOT,
        debug=args.debug,
        render_png=not args.no_png,
        path_style=args.path_style,
        rewrite=args.rewrite,
        checkpoint_csv=out_csv,
    )

    utils.write_documents_csv(df_out, out_csv)
    print(f"Wrote table: {out_csv}")


if __name__ == "__main__":
    main()
