"""
CSV (TSV) -> merged document AMR graph -> data-temp (.pkl, .png) -> ``graph`` column in the table.

Run from project root:
  python amr/amr_parse.py
  python amr/amr_parse.py --debug -i data/test-set.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import amr_save  # noqa: E402
import utils  # noqa: E402

import networkx as nx
import pandas as pd
import penman
from penman import surface


def _abspath(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _anchor_for_tokens(doc, sent_idx: int, token_indices: list[int]) -> list[int] | None:
    if not token_indices:
        return None
    sents = list(doc.sents)
    if sent_idx < 0 or sent_idx >= len(sents):
        return None
    tokens = list(sents[sent_idx])
    spans: list[tuple[int, int]] = []
    for ti in token_indices:
        if 0 <= ti < len(tokens):
            t = tokens[ti]
            spans.append((int(t.idx), int(t.idx) + len(t.text)))
    if not spans:
        return None
    return [min(s for s, _ in spans), max(e for _, e in spans)]


def penman_to_nx(p_str: str, *, sent_idx: int, doc) -> tuple[nx.DiGraph, str]:
    """
    PENMAN string -> DiGraph. Node attrs: name, concept, sent_idx, idx, anchor; edges: relation.
    Also: is_constant (for visualization).
    """
    g = penman.decode(p_str)
    dg = nx.DiGraph()
    alignments_map = surface.alignments(g)
    triples = sorted(g.triples, key=lambda t: (0 if t[1] == ":instance" else 1))
    inst_vars = {t[0] for t in triples if t[1] == ":instance"}

    for source, role, target in triples:
        al = alignments_map.get((source, role, target))
        raw_idx = list(al.indices) if al else []
        sent_i = sent_idx
        tok_i = raw_idx[0] if raw_idx else -1
        anchor = _anchor_for_tokens(doc, sent_idx, raw_idx)

        if role == ":instance":
            var = str(source)
            concept = str(target)
            dg.add_node(
                var,
                name=f"{var}/{concept}",
                concept=concept,
                sent_idx=sent_i,
                idx=tok_i,
                anchor=anchor,
                is_constant=False,
            )
        else:
            tgt = str(target)
            if target not in inst_vars:
                if tgt not in dg.nodes:
                    dg.add_node(
                        tgt,
                        name=tgt,
                        concept=tgt,
                        sent_idx=sent_i,
                        idx=tok_i if raw_idx else -1,
                        anchor=anchor,
                        is_constant=True,
                    )
            dg.add_edge(str(source), tgt, relation=str(role))

    top = g.top
    if top is None:
        raise ValueError("PENMAN graph has no top (root) node.")
    return dg, str(top)


def merge_document_graph(graphs: list[nx.DiGraph], roots: list[str]) -> nx.DiGraph:
    """``document`` node and ``:snt1``, ``:snt2``, ... edges; variable prefixes ``s0_``, ``s1_``, ..."""
    out = nx.DiGraph()
    out.add_node(
        "document",
        name="document",
        concept="document",
        sent_idx=-1,
        idx=-1,
        anchor=None,
        is_constant=False,
    )
    for i, (g, root) in enumerate(zip(graphs, roots)):
        pfx = f"s{i}_"
        m = {n: pfx + str(n) if not str(n).startswith(pfx) else str(n) for n in g.nodes()}

        for n, data in g.nodes(data=True):
            data = dict(data)
            data.pop("lemma", None)
            nid = m[n]
            if data.get("is_constant"):
                data["name"] = data.get("concept", str(n))
            else:
                sense = data.get("concept") or ""
                data["name"] = f"{nid}/{sense}" if sense else nid
            out.add_node(nid, **data)

        for u, v, ed in g.edges(data=True):
            out.add_edge(m[u], m[v], **dict(ed))

        out.add_edge("document", m[root], relation=f":snt{i + 1}")

    return out


def build_document_amr(
    text: str,
    *,
    model_dir: str,
    spacy_model: str,
    debug: bool,
) -> nx.DiGraph:
    import amrlib
    import spacy
    from amrlib.alignments.rbw_aligner import RBWAligner
    from amrlib.graph_processing.annotator import add_lemmas

    nlp = spacy.load(spacy_model)
    doc = nlp(text.strip())
    sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    if not sents:
        raise ValueError("No sentences after segmentation.")

    if debug:
        print(f"[debug] Sentences: {len(sents)}")

    stog = amrlib.load_stog_model(model_dir=model_dir)
    raw_graphs = stog.parse_sents(sents)
    aligned: list[str] = []
    for i, (snt, gs) in enumerate(zip(sents, raw_graphs)):
        with_lemmas = add_lemmas(gs, snt_key="snt")
        ag = RBWAligner.from_penman_w_json(with_lemmas).get_graph_string()
        aligned.append(ag)
        if debug:
            snip = ag[:500] + ("..." if len(ag) > 500 else "")
            print(f"[debug] snt {i}: {snip}\n")

    graphs: list[nx.DiGraph] = []
    roots: list[str] = []
    for i, ps in enumerate(aligned):
        dg, r = penman_to_nx(ps, sent_idx=i, doc=doc)
        graphs.append(dg)
        roots.append(r)

    merged = merge_document_graph(graphs, roots)
    if debug:
        print(
            f"[debug] Document graph: nodes={merged.number_of_nodes()}, "
            f"edges={merged.number_of_edges()}"
        )
    return merged


def process_dataframe(
    df: pd.DataFrame,
    *,
    model_dir: str,
    data_temp: str,
    project_root: str,
    spacy_model: str,
    debug: bool,
    render_png: bool,
    graph_path_style: str = "relative",
    rewrite: bool = False,
    checkpoint_csv: str | None = None,
) -> pd.DataFrame:
    """
    For each row: build graph, save artifacts, fill ``graph`` column.
    ``graph_path_style``: ``relative`` to project root, or ``absolute``.
    If ``rewrite`` is False, rows with a non-empty ``graph`` cell are skipped.
    If ``checkpoint_csv`` is set, the table is written to that path after each
    row that updates ``graph`` (resume-safe on interrupt).
    """
    out = df.copy()
    if "graph" not in out.columns:
        out["graph"] = ""

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
            out.loc[i, "graph"] = ""
            if debug:
                print(f"[debug] Empty content, row index {i}")
            _flush_checkpoint()
            continue

        if not rewrite and utils.csv_cell_has_value(row.get("graph")):
            print(f"Skip row {i}: graph column already set")
            continue

        pkl_path = os.path.join(data_temp, f"amr_{i}.pkl")
        png_base = os.path.join(data_temp, f"amr_{i}")

        t_parse = time.perf_counter()
        graph = build_document_amr(
            content,
            model_dir=model_dir,
            spacy_model=spacy_model,
            debug=debug,
        )
        parse_sec = time.perf_counter() - t_parse
        print(f"[timing] row {i} AMR document parse: {parse_sec:.1f}s", flush=True)

        bundle = {
            "graph": graph,
            "category": row.get("category", ""),
            "content": content,
            "filename": row.get("filename", ""),
            "title": row.get("title", ""),
        }
        amr_save.save_graph_pickle(bundle, pkl_path)

        if render_png:
            try:
                amr_save.render_amr_graph(graph, png_base, title=f"AMR amr_{i}")
            except (ImportError, RuntimeError, OSError) as e:
                print(f"Warning: PNG not written ({e}).", file=sys.stderr)

        abs_pkl = _abspath(pkl_path)
        if graph_path_style == "absolute":
            path_str = abs_pkl
        else:
            try:
                path_str = os.path.relpath(abs_pkl, project_root)
            except ValueError:
                path_str = abs_pkl
        path_str = os.path.normpath(path_str)
        out.loc[i, "graph"] = path_str
        if debug:
            print(f"[debug] graph[{i}] = {out.loc[i, 'graph']}")
        _flush_checkpoint()

    return out


def main() -> None:
    default_input = os.path.join(PROJECT_ROOT, "data", "test-set.csv")
    default_out_dir = os.path.join(PROJECT_ROOT, "data-temp")
    default_model = os.path.join(PROJECT_ROOT, "model", "model_parse_xfm_bart_large-v0_1_0")

    parser = argparse.ArgumentParser(
        description="Document AMR -> NetworkX -> data-temp + graph column"
    )
    parser.add_argument("-i", "--input", type=str, default=default_input, help="Input CSV (pandas)")
    parser.add_argument(
        "-o",
        "--output-csv",
        type=str,
        default=None,
        help="Output table with graph column (default: overwrite input)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=default_model,
        help="amrlib StoG model directory",
    )
    parser.add_argument(
        "--data-temp",
        type=str,
        default=default_out_dir,
        help="Directory for .pkl and PNG",
    )
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Re-parse every row; default: only rows with empty graph column",
    )
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument(
        "--graph-path",
        choices=("relative", "absolute"),
        default="relative",
        help="Path style for the graph column",
    )
    args = parser.parse_args()

    inp = _abspath(args.input)
    if not os.path.isfile(inp):
        print(f"File not found: {inp}", file=sys.stderr)
        sys.exit(1)

    model_dir = _abspath(args.model_dir)
    if not os.path.isdir(model_dir):
        print(
            f"Model directory not found: {model_dir}\n"
            "Set --model-dir to the folder that contains your StoG checkpoint.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.debug:
        print(f"[debug] Model directory: {model_dir}")

    out_csv = _abspath(args.output_csv) if args.output_csv else inp

    df = utils.read_documents_csv(inp)
    df_out = process_dataframe(
        df,
        model_dir=model_dir,
        data_temp=args.data_temp,
        project_root=PROJECT_ROOT,
        spacy_model=args.spacy_model,
        debug=args.debug,
        render_png=not args.no_png,
        graph_path_style=args.graph_path,
        rewrite=args.rewrite,
        checkpoint_csv=out_csv,
    )

    utils.write_documents_csv(df_out, out_csv)
    print(f"Wrote table: {out_csv}")


if __name__ == "__main__":
    main()
