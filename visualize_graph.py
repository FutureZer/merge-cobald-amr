"""
Point visualization for a single graph pickle (AMR, CoBaLD, hybrid, merged).

Run from project root:
  python visualize_graph.py --type amr -p data-temp/amr_0.pkl
  python visualize_graph.py --type hybrid -p data-temp/hybrid_0.pkl -o out.png
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import Any

import networkx as nx

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = _THIS_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
_AMR_DIR = os.path.join(PROJECT_ROOT, "amr")
_COBALD_DIR = os.path.join(PROJECT_ROOT, "cobald")
if _AMR_DIR not in sys.path:
    sys.path.insert(0, _AMR_DIR)
if _COBALD_DIR not in sys.path:
    sys.path.insert(0, _COBALD_DIR)

import amr_save  # noqa: E402
import cobald_save  # noqa: E402


def _abspath(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def load_graph_from_pickle(pkl_path: str) -> nx.DiGraph:
    pkl_path = _abspath(pkl_path)
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"Not a file: {pkl_path}")
    with open(pkl_path, "rb") as f:
        bundle: dict[str, Any] = pickle.load(f)
    g = bundle.get("graph")
    if not isinstance(g, nx.DiGraph):
        raise TypeError(f"Pickle must contain a NetworkX DiGraph under key 'graph', got {type(g)}")
    return g


def default_output_path(pkl_path: str, graph_type: str) -> str:
    base, _ = os.path.splitext(_abspath(pkl_path))
    return f"{base}_{graph_type}_viz.png"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render one document graph pickle to PNG (AMR / CoBaLD / hybrid / merged)."
    )
    parser.add_argument(
        "--type",
        "-t",
        choices=("amr", "cobald", "hybrid", "merged"),
        required=True,
        help="Graph kind: amr=clean AMR viz; cobald=CoBaLD viz; hybrid|merged=AMR viz + CoBaLD attrs on nodes",
    )
    parser.add_argument(
        "--pickle",
        "-p",
        type=str,
        required=True,
        help="Path to .pkl (dict with 'graph' key)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output PNG path (default: next to pickle, *_{type}_viz.png)",
    )
    parser.add_argument("--title", type=str, default=None, help="Figure / graph title")
    args = parser.parse_args()

    pkl_path = _abspath(args.pickle)
    graph = load_graph_from_pickle(pkl_path)

    out = _abspath(args.output) if args.output else default_output_path(pkl_path, args.type)
    if not out.lower().endswith(".png"):
        out = out + ".png"

    stem = os.path.splitext(os.path.basename(pkl_path))[0]
    title = args.title or f"{args.type.upper()} {stem}"

    if args.type == "cobald":
        path_written = cobald_save.render_cobald_graph(graph, out, title=title)
    else:
        out_base, _ext = os.path.splitext(out)
        path_written = amr_save.render_amr_graph(
            graph,
            out_base,
            title=title,
            show_cobald_attrs=args.type in ("hybrid", "merged"),
        )

    print(f"Wrote {path_written}")


if __name__ == "__main__":
    main()
