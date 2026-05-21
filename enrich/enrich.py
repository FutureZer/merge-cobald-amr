"""
Load AMR + CoBaLD pickles from CSV (columns ``graph``, ``conllu``), merge CoBaLD node
attributes into AMR nodes by character ``anchor`` overlap, save hybrid pickle + PNG.

Run from project root:
  python enrich/enrich.py -i data/test-set.csv
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from typing import Any, Mapping

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
_AMR_DIR = os.path.join(PROJECT_ROOT, "amr")
if _AMR_DIR not in sys.path:
    sys.path.insert(0, _AMR_DIR)

import amr_save  # noqa: E402
import utils  # noqa: E402

import networkx as nx
import pandas as pd


def _abspath(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _cell_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(x).strip()
    if s.lower() in ("nan", "none", ""):
        return ""
    return s


def resolve_artifact_path(cell: str, project_root: str) -> str:
    """Resolve path from CSV (relative to project root or absolute)."""
    p = _cell_str(cell)
    if not p:
        return ""
    p = p.strip('"').strip("'")
    p = os.path.normpath(p)
    if os.path.isabs(p):
        return _abspath(p)
    return _abspath(os.path.join(project_root, p))


def load_graph_bundle(pkl_path: str) -> tuple[nx.DiGraph, dict[str, Any]]:
    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)
    g = bundle.get("graph")
    if not isinstance(g, nx.DiGraph):
        raise TypeError(f"Pickle {pkl_path} has no DiGraph under key 'graph'.")
    return g, bundle


def char_span_overlap(a: list | tuple | None, b: list | tuple | None) -> int:
    """Overlap length of two [start, end) character intervals (end exclusive)."""
    if a is None or b is None:
        return 0
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
        return 0
    if len(a) < 2 or len(b) < 2:
        return 0
    a0, a1 = int(a[0]), int(a[1])
    b0, b1 = int(b[0]), int(b[1])
    return max(0, min(a1, b1) - max(a0, b0))


def _is_cobald_token_node(data: Mapping[str, Any]) -> bool:
    if data.get("idx", -1) < 0:
        return False
    if str(data.get("lemma", "")) == "<ROOT>":
        return False
    if str(data.get("token", "")) == "DOCUMENT":
        return False
    anch = data.get("anchor")
    if not anch or not isinstance(anch, (list, tuple)) or len(anch) < 2:
        return False
    return True


def cobald_token_index(cobald: nx.DiGraph) -> list[tuple[tuple[int, int], dict[str, Any]]]:
    """List of (anchor_tuple, node_data) for CoBaLD token nodes."""
    out: list[tuple[tuple[int, int], dict[str, Any]]] = []
    for _n, d in cobald.nodes(data=True):
        if not _is_cobald_token_node(d):
            continue
        anch = d.get("anchor")
        t = (int(anch[0]), int(anch[1]))
        out.append((t, dict(d)))
    return out


def enrich_amr_from_cobald(amr: nx.DiGraph, cobald: nx.DiGraph) -> nx.DiGraph:
    """
    Copy AMR graph and add ``deepslot`` / ``semclass`` on each node by best anchor overlap
    with CoBaLD token nodes (ENRICH.md).
    """
    G = amr.copy()
    cob_idx = cobald_token_index(cobald)

    for nid, data in G.nodes(data=True):
        d = dict(data)
        d.pop("lemma", None)
        anch = d.get("anchor")
        if not anch or not isinstance(anch, (list, tuple)) or len(anch) < 2:
            d.setdefault("deepslot", "_")
            d.setdefault("semclass", "_")
            G.add_node(nid, **d)
            continue

        best_ov = -1
        best_ds: str | None = None
        best_sc: str | None = None
        best_span = 10**9

        for (b0, b1), cd in cob_idx:
            ov = char_span_overlap(anch, (b0, b1))
            span_len = b1 - b0
            if ov > best_ov:
                best_ov = ov
                best_ds = cd.get("deepslot")
                best_sc = cd.get("semclass")
                best_span = span_len
            elif ov == best_ov and ov > 0 and span_len < best_span:
                best_ds = cd.get("deepslot")
                best_sc = cd.get("semclass")
                best_span = span_len

        if best_ov > 0:
            d["deepslot"] = str(best_ds) if best_ds not in (None, "") else "_"
            d["semclass"] = str(best_sc) if best_sc not in (None, "") else "_"
        else:
            d["deepslot"] = "_"
            d["semclass"] = "_"

        G.add_node(nid, **d)

    return G


def hybrid_artifact_paths(output_dir: str, row_index: int) -> tuple[str, str]:
    output_dir = _abspath(output_dir)
    base = f"hybrid_{row_index}"
    pkl = os.path.join(output_dir, f"{base}.pkl")
    png_base = os.path.join(output_dir, base)
    return pkl, png_base


def process_dataframe(
    df: pd.DataFrame,
    *,
    project_root: str,
    data_temp: str,
    path_style: str,
    render_png: bool,
    debug: bool,
    rewrite: bool = False,
    checkpoint_csv: str | None = None,
) -> pd.DataFrame:
    """
    For each row: merge CoBaLD attrs into AMR, save hybrid artifacts, fill ``hybrid`` column.
    If ``rewrite`` is False, rows with a non-empty ``hybrid`` cell are skipped.
    If ``checkpoint_csv`` is set, the table is written after each row that updates ``hybrid``
    (resume-safe on interrupt).
    """
    out = df.copy()
    if "hybrid" not in out.columns:
        out["hybrid"] = ""

    project_root = _abspath(project_root)
    data_temp = _abspath(data_temp)
    os.makedirs(data_temp, exist_ok=True)

    ckpt = _abspath(checkpoint_csv) if checkpoint_csv else None

    def _flush_checkpoint() -> None:
        if ckpt:
            utils.write_documents_csv(out, ckpt)

    n_rows = len(out)
    for pos, (i, row) in enumerate(out.iterrows()):
        print(f"[progress] {pos + 1}/{n_rows} (row index {i})", flush=True)

        if not rewrite and utils.csv_cell_has_value(row.get("hybrid")):
            print(f"Skip row {i}: hybrid column already set")
            continue

        g_amr = _cell_str(row.get("graph", ""))
        g_co = _cell_str(row.get("conllu", ""))
        if not g_amr or not g_co:
            out.loc[i, "hybrid"] = ""
            if debug:
                print(f"[debug] Skip row {i}: missing graph or conllu path")
            _flush_checkpoint()
            continue

        p_amr = resolve_artifact_path(g_amr, project_root)
        p_co = resolve_artifact_path(g_co, project_root)
        if not os.path.isfile(p_amr) or not os.path.isfile(p_co):
            if debug:
                print(f"[debug] Skip row {i}: file not found amr={p_amr} cobald={p_co}")
            out.loc[i, "hybrid"] = ""
            _flush_checkpoint()
            continue

        t_enrich = time.perf_counter()
        amr_g, _ = load_graph_bundle(p_amr)
        cob_g, _ = load_graph_bundle(p_co)

        hybrid_g = enrich_amr_from_cobald(amr_g, cob_g)
        enrich_sec = time.perf_counter() - t_enrich
        print(f"[timing] row {i} enrichment: {enrich_sec:.2f}s", flush=True)

        pkl_path, png_base = hybrid_artifact_paths(data_temp, int(i))
        bundle = {
            "graph": hybrid_g,
            "category": row.get("category", ""),
            "content": row.get("content", ""),
            "filename": row.get("filename", ""),
            "title": row.get("title", ""),
            "source_amr_pkl": p_amr,
            "source_cobald_pkl": p_co,
        }
        amr_save.save_graph_pickle(bundle, pkl_path)

        if render_png:
            try:
                amr_save.render_amr_graph(
                    hybrid_g,
                    png_base,
                    title=f"Hybrid AMR+CoBaLD hybrid_{i}",
                    show_cobald_attrs=True,
                )
            except (ImportError, OSError, RuntimeError) as e:
                print(f"Warning: hybrid PNG not written ({e}).", file=sys.stderr)

        abs_pkl = _abspath(pkl_path)
        if path_style == "absolute":
            path_str = abs_pkl
        else:
            try:
                path_str = os.path.relpath(abs_pkl, project_root)
            except ValueError:
                path_str = abs_pkl
        out.loc[i, "hybrid"] = os.path.normpath(path_str)
        if debug:
            print(f"[debug] hybrid[{i}] = {out.loc[i, 'hybrid']}")
        _flush_checkpoint()

    return out


def main() -> None:
    default_input = os.path.join(PROJECT_ROOT, "data", "test-set.csv")
    default_out_dir = os.path.join(PROJECT_ROOT, "data-temp")

    parser = argparse.ArgumentParser(
        description="ENRICH.md: AMR backbone + CoBaLD deepslot/semclass via anchor overlap"
    )
    parser.add_argument("-i", "--input", type=str, default=default_input, help="CSV with graph + conllu")
    parser.add_argument(
        "-o",
        "--output-csv",
        type=str,
        default=None,
        help="Output CSV with hybrid column (default: overwrite input)",
    )
    parser.add_argument("--data-temp", type=str, default=default_out_dir, help="Output dir for hybrid .pkl/.png")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Re-enrich every row; default: skip rows with non-empty hybrid column",
    )
    parser.add_argument(
        "--path-style",
        choices=("relative", "absolute"),
        default="relative",
        help="Path style for hybrid column",
    )
    args = parser.parse_args()

    inp = _abspath(args.input)
    if not os.path.isfile(inp):
        print(f"File not found: {inp}", file=sys.stderr)
        sys.exit(1)

    out_csv = _abspath(args.output_csv) if args.output_csv else inp

    df = utils.read_documents_csv(inp)
    for col in ("graph", "conllu"):
        if col not in df.columns:
            print(f"CSV must contain column '{col}'. Found: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)

    df_out = process_dataframe(
        df,
        project_root=PROJECT_ROOT,
        data_temp=args.data_temp,
        path_style=args.path_style,
        render_png=not args.no_png,
        debug=args.debug,
        rewrite=args.rewrite,
        checkpoint_csv=out_csv,
    )

    utils.write_documents_csv(df_out, out_csv)
    print(f"Wrote table: {out_csv}")


if __name__ == "__main__":
    main()
