"""
Load hybrid AMR+CoBaLD pickles (CSV column ``hybrid``), merge document graph per MERGE.md,
save merged pickle + optional PNG.

Run from project root:
  python merge/merge.py -i data/test-set.csv
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from typing import Any

import networkx as nx
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
_AMR_DIR = os.path.join(PROJECT_ROOT, "amr")
if _AMR_DIR not in sys.path:
    sys.path.insert(0, _AMR_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import amr_save  # noqa: E402
import merge_graph  # noqa: E402
import name_cleanup  # noqa: E402
import utils  # noqa: E402


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


def merged_artifact_paths(output_dir: str, row_index: int) -> tuple[str, str]:
    output_dir = _abspath(output_dir)
    base = f"merged_{row_index}"
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
    spacy_model: str,
    device: str,
    rewrite: bool = False,
    checkpoint_csv: str | None = None,
) -> pd.DataFrame:
    """
    For each row: coref-merge hybrid graph, apply ``name_cleanup`` (duplicate ``:name`` /
    literal collapse + strip ``/name`` nodes), save merged pickle, fill ``merged`` column.
    If ``rewrite`` is False, a row is skipped (no merge, pickle not overwritten) when
    ``{data_temp}/merged_{row_index}.pkl`` already exists, after hybrid/text checks pass.
    If ``checkpoint_csv`` is set, the table is written after each row that updates ``merged``.
    """
    out = df.copy()
    if "merged" not in out.columns:
        out["merged"] = ""

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

        hy = _cell_str(row.get("hybrid", ""))
        if not hy:
            out.loc[i, "merged"] = ""
            if debug:
                print(f"[debug] Skip row {i}: missing hybrid path")
            _flush_checkpoint()
            continue

        p_hybrid = resolve_artifact_path(hy, project_root)
        if not os.path.isfile(p_hybrid):
            if debug:
                print(f"[debug] Skip row {i}: hybrid file not found {p_hybrid}")
            out.loc[i, "merged"] = ""
            _flush_checkpoint()
            continue

        graph, bundle = load_graph_bundle(p_hybrid)
        text = _cell_str(row.get("content", "")) or str(bundle.get("content", "")).strip()
        if not text:
            out.loc[i, "merged"] = ""
            if debug:
                print(f"[debug] Skip row {i}: empty content for coreference")
            _flush_checkpoint()
            continue

        pkl_path, png_base = merged_artifact_paths(data_temp, int(i))

        if not rewrite and os.path.isfile(pkl_path):
            print(f"Skip row {i}: merged pickle already exists")
            abs_pkl = _abspath(pkl_path)
            if path_style == "absolute":
                path_str = abs_pkl
            else:
                try:
                    path_str = os.path.relpath(abs_pkl, project_root)
                except ValueError:
                    path_str = abs_pkl
            out.loc[i, "merged"] = os.path.normpath(path_str)
            if debug:
                print(f"[debug] merged[{i}] = {out.loc[i, 'merged']}")
            _flush_checkpoint()
            continue

        t_merge = time.perf_counter()
        merged_g = merge_graph.merge_document_graph(
            graph,
            text,
            spacy_model=spacy_model,
            device=device,
            debug=debug,
        )
        merge_sec = time.perf_counter() - t_merge
        print(f"[timing] row {i} merge: {merge_sec:.1f}s", flush=True)

        t_clean = time.perf_counter()
        name_cleanup.cleanup_duplicate_name_structures(merged_g, debug=debug)
        clean_sec = time.perf_counter() - t_clean
        print(f"[timing] row {i} name cleanup: {clean_sec:.2f}s", flush=True)

        out_bundle = dict(bundle)
        out_bundle["graph"] = merged_g
        out_bundle["source_hybrid_pkl"] = p_hybrid
        amr_save.save_graph_pickle(out_bundle, pkl_path)

        if render_png:
            try:
                amr_save.render_amr_graph(
                    merged_g,
                    png_base,
                    title=f"Merged document graph merged_{i}",
                    show_cobald_attrs=True,
                )
            except (ImportError, OSError, RuntimeError) as e:
                print(f"Warning: merged PNG not written ({e}).", file=sys.stderr)

        abs_pkl = _abspath(pkl_path)
        if path_style == "absolute":
            path_str = abs_pkl
        else:
            try:
                path_str = os.path.relpath(abs_pkl, project_root)
            except ValueError:
                path_str = abs_pkl
        out.loc[i, "merged"] = os.path.normpath(path_str)
        if debug:
            print(f"[debug] merged[{i}] = {out.loc[i, 'merged']}")
        _flush_checkpoint()

    return out


def main() -> None:
    default_input = os.path.join(PROJECT_ROOT, "data", "test-set.csv")
    default_out_dir = os.path.join(PROJECT_ROOT, "data-temp")

    parser = argparse.ArgumentParser(
        description="MERGE.md: coref-based merge of hybrid AMR document graph"
    )
    parser.add_argument("-i", "--input", type=str, default=default_input, help="CSV with hybrid column")
    parser.add_argument(
        "-o",
        "--output-csv",
        type=str,
        default=None,
        help="Output CSV with merged column (default: overwrite input)",
    )
    parser.add_argument("--data-temp", type=str, default=default_out_dir, help="Output dir for merged .pkl/.png")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Re-merge and overwrite pickles; default: skip if data-temp/merged_{i}.pkl exists",
    )
    parser.add_argument(
        "--path-style",
        choices=("relative", "absolute"),
        default="relative",
        help="Path style for merged column",
    )
    parser.add_argument("--spacy-model", type=str, default="en_core_web_sm", help="spaCy model for alignment")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for FastCoref (e.g. cpu, cuda:0)",
    )
    args = parser.parse_args()

    inp = _abspath(args.input)
    if not os.path.isfile(inp):
        print(f"File not found: {inp}", file=sys.stderr)
        sys.exit(1)

    out_csv = _abspath(args.output_csv) if args.output_csv else inp

    df = utils.read_documents_csv(inp)
    if "hybrid" not in df.columns:
        print(f"CSV must contain column 'hybrid'. Found: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    df_out = process_dataframe(
        df,
        project_root=PROJECT_ROOT,
        data_temp=args.data_temp,
        path_style=args.path_style,
        render_png=not args.no_png,
        debug=args.debug,
        spacy_model=args.spacy_model,
        device=args.device,
        rewrite=args.rewrite,
        checkpoint_csv=out_csv,
    )

    utils.write_documents_csv(df_out, out_csv)
    print(f"Wrote table: {out_csv}")


if __name__ == "__main__":
    main()
