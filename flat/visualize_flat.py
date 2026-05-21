"""
Render one graph from a Gaston multi-graph .txt (FLAT.md) to PNG.

Run from project root:
  python flat/visualize_flat.py flat/amr/business.txt -n 2
  python flat/visualize_flat.py flat/enr-node/sport.txt -n 0 -o out.png
  python flat/visualize_flat.py flat/train/cobald/business.txt -n 0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

import networkx as nx

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
_AMR_DIR = os.path.join(PROJECT_ROOT, "amr")
if _AMR_DIR not in sys.path:
    sys.path.insert(0, _AMR_DIR)

import amr_save  # noqa: E402


def _abspath(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _find_flat_export_root(start_dir: str) -> str | None:
    """Directory that contains ``vocab_nodes_global.json`` (``flat/`` from export_gaston)."""
    d = start_dir
    while True:
        if os.path.isfile(os.path.join(d, "vocab_nodes_global.json")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def default_vocab_paths(flat_txt: str) -> tuple[str, str]:
    """
    Resolve vocab JSON next to export_gaston layout:
    ``flat/{train|test}/amr/*.txt`` / ``.../cobald/*.txt`` -> ``flat/vocab_*_global.json``;
    ``.../enr-node/*.txt`` -> ``flat/enr-node/vocab_*_enr.json`` (vocabs live at export root).
    """
    flat_txt = _abspath(flat_txt)
    cur_dir = os.path.dirname(flat_txt)

    strategy_dir: str | None = None
    probe = cur_dir
    while True:
        folder = os.path.basename(probe).lower()
        if folder in ("amr", "enr-node", "cobald", "add-node"):
            strategy_dir = probe
            break
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent

    if strategy_dir is None:
        raise ValueError(
            f"Cannot guess vocab paths for {flat_txt!r} (path must contain a folder "
            f"amr, enr-node, or cobald). Use --nodes-vocab and --edges-vocab."
        )

    export_root = _find_flat_export_root(strategy_dir)
    if export_root is None:
        raise ValueError(
            f"Cannot find flat export root (vocab_nodes_global.json) above {strategy_dir!r}. "
            "Use --nodes-vocab and --edges-vocab."
        )

    strat = os.path.basename(strategy_dir).lower()
    if strat == "enr-node":
        enr = os.path.join(export_root, "enr-node")
        return (
            os.path.join(enr, "vocab_nodes_enr.json"),
            os.path.join(enr, "vocab_edges_enr.json"),
        )
    if strat in ("amr", "cobald", "add-node"):
        return (
            os.path.join(export_root, "vocab_nodes_global.json"),
            os.path.join(export_root, "vocab_edges_global.json"),
        )
    raise ValueError(
        f"Cannot guess vocab paths for {flat_txt!r} (unexpected strategy folder {strat!r}). "
        "Use --nodes-vocab and --edges-vocab."
    )


def load_id_label_map(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)
    return {str(k): str(v) for k, v in raw.items()}


def parse_flat_file(path: str) -> dict[int, dict[str, Any]]:
    """
    Map transaction index ``t # N`` to section payload.

    Supports both:
    - dataset files with ``t/v/e`` blocks
    - Gaston ``.OUT`` pattern files with leading ``# support`` and trailing ``x ...``.
    """
    path = _abspath(path)
    by_index: dict[int, dict[str, Any]] = {}
    current: int | None = None
    pending_support: int | None = None
    header_re = re.compile(r"^t\s*#\s*(\d+)\s*$")
    support_re = re.compile(r"^#\s*(\d+)\s*$")

    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            sm = support_re.match(line)
            if sm:
                pending_support = int(sm.group(1))
                continue
            m = header_re.match(line)
            if m:
                current = int(m.group(1))
                by_index[current] = {
                    "lines": [],
                    "support": pending_support,
                    "occurrences": [],
                }
                pending_support = None
                continue
            if current is None:
                raise ValueError(f"{path}: line before first transaction header: {line!r}")
            parts = line.split()
            kind = parts[0] if parts else ""
            if kind in ("v", "e"):
                by_index[current]["lines"].append(line)
            elif kind == "x":
                occ = []
                for p in parts[1:]:
                    try:
                        occ.append(int(p))
                    except ValueError:
                        continue
                by_index[current]["occurrences"] = occ
            else:
                # Ignore unknown lines to stay robust with minor format variations.
                continue
    return by_index


def section_to_digraph(
    lines: list[str],
    node_id_to_label: dict[str, str],
    edge_id_to_label: dict[str, str],
) -> nx.DiGraph:
    G = nx.DiGraph()
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        kind = parts[0]
        if kind == "v" and len(parts) >= 3:
            vid = int(parts[1])
            lid = int(parts[2])
            lab = node_id_to_label.get(str(lid))
            if lab is None:
                raise KeyError(f"Unknown node label id {lid} (vocab size {len(node_id_to_label)})")
            # ``name`` only -> amr_save shows the label without ``id / concept`` prefix.
            G.add_node(vid, name=lab, is_constant=False)
        elif kind == "e" and len(parts) >= 4:
            u, v, eid = int(parts[1]), int(parts[2]), int(parts[3])
            elab = edge_id_to_label.get(str(eid))
            if elab is None:
                raise KeyError(f"Unknown edge label id {eid}")
            G.add_edge(u, v, relation=elab)
        else:
            raise ValueError(f"Bad line (expected v or e): {line!r}")
    return G


def default_output_base(flat_txt: str, graph_index: int) -> str:
    base, _ = os.path.splitext(_abspath(flat_txt))
    return f"{base}_g{graph_index}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize one transaction from a Gaston flat .txt (see FLAT.md)."
    )
    parser.add_argument(
        "flat_txt",
        type=str,
        help="Path to multi-graph file, e.g. flat/amr/business.txt",
    )
    parser.add_argument(
        "-n",
        "--graph-index",
        type=int,
        default=0,
        metavar="N",
        help="Graph index from the ``t # N`` header (default: 0)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output PNG path (default: <flat_txt>_g<N>.png next to the .txt)",
    )
    parser.add_argument("--title", type=str, default=None, help="Figure title")
    parser.add_argument(
        "--nodes-vocab",
        type=str,
        default=None,
        help="Override path to vocab_nodes_*.json",
    )
    parser.add_argument(
        "--edges-vocab",
        type=str,
        default=None,
        help="Override path to vocab_edges_*.json",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_only",
        help="Print transaction indices present in the file and exit",
    )
    args = parser.parse_args()

    flat_path = _abspath(args.flat_txt)
    if not os.path.isfile(flat_path):
        print(f"File not found: {flat_path}", file=sys.stderr)
        sys.exit(1)

    sections = parse_flat_file(flat_path)
    indices = sorted(sections.keys())

    if args.list_only:
        print(f"{flat_path}: {len(indices)} graph(s), indices: {indices}")
        sys.exit(0)

    if not indices:
        print(f"No graphs found in {flat_path}", file=sys.stderr)
        sys.exit(1)

    n = args.graph_index
    if n not in sections:
        print(
            f"No transaction ``t # {n}`` in file. Available indices: {indices}",
            file=sys.stderr,
        )
        sys.exit(1)

    nv_path = _abspath(args.nodes_vocab) if args.nodes_vocab else None
    ev_path = _abspath(args.edges_vocab) if args.edges_vocab else None
    if not nv_path or not ev_path:
        auto_nv, auto_ev = default_vocab_paths(flat_path)
        nv_path = nv_path or auto_nv
        ev_path = ev_path or auto_ev

    for p, label in ((nv_path, "nodes vocab"), (ev_path, "edges vocab")):
        if not os.path.isfile(p):
            print(f"Missing {label}: {p}", file=sys.stderr)
            sys.exit(1)

    node_map = load_id_label_map(nv_path)
    edge_map = load_id_label_map(ev_path)

    payload = sections[n]
    G = section_to_digraph(payload["lines"], node_map, edge_map)

    out = args.output
    if out:
        out = _abspath(out)
        if not out.lower().endswith(".png"):
            out = out + ".png"
        out_base = os.path.splitext(out)[0]
    else:
        out_base = default_output_base(flat_path, n)
        out = out_base + ".png"

    stem = os.path.basename(flat_path)
    support = payload.get("support")
    occ = payload.get("occurrences") or []
    if args.title:
        title = args.title
    else:
        meta = []
        if isinstance(support, int):
            meta.append(f"sup={support}")
        if occ:
            meta.append(f"x={len(occ)}")
        meta_s = ("  " + " ".join(meta)) if meta else ""
        title = f"flat {stem}  t # {n}  ({G.number_of_nodes()}n/{G.number_of_edges()}e){meta_s}"

    path_written = amr_save.render_amr_graph(
        G,
        out_base,
        title=title,
        show_cobald_attrs=False,
    )
    print(f"Wrote {path_written}")


if __name__ == "__main__":
    main()
