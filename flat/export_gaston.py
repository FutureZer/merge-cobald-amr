"""
Build Gaston / gSofia flat graph files from merged hybrid pickles (FLAT.md).

Run from project root:
  python flat/export_gaston.py -i data/bbc-news-data.csv --train-ratio 0.8

Output layout:
  flat/
    train/{amr,enr-node,cobald}/*.txt
    test/{amr,enr-node,cobald}/*.txt
    vocab_nodes_global.json
    vocab_edges_global.json
    enr-node/vocab_nodes_enr.json
    enr-node/vocab_edges_enr.json
"""

from __future__ import annotations

import argparse
import json
import random
import os
import pickle
import re
import sys
from typing import Any

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import networkx as nx
import pandas as pd

import utils  # noqa: E402

STRATEGIES = ("amr", "enr-node", "cobald")


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


def load_merged_graph(pkl_path: str) -> nx.DiGraph:
    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)
    g = bundle.get("graph")
    if not isinstance(g, nx.DiGraph):
        raise TypeError(f"Pickle {pkl_path} has no DiGraph under key 'graph'.")
    return g


def safe_category_filename(category: str) -> str:
    s = str(category).strip() or "unknown"
    s = re.sub(r"[^\w\-]+", "_", s, flags=re.UNICODE)
    s = s.strip("_") or "unknown"
    return s


def amr_baseline_node_label(data: dict[str, Any]) -> str:
    """Strategy 1: AMR concept (constants: name or concept)."""
    if bool(data.get("is_constant")):
        name = data.get("name")
        if name is not None and str(name).strip():
            return str(name).strip()
        c = data.get("concept")
        if c is not None and str(c).strip():
            return str(c).strip()
        return "_"
    c = data.get("concept")
    if c is not None and str(c).strip():
        return str(c).strip()
    return "_"


def enriched_tuple_node_label(data: dict[str, Any]) -> str:
    """
    Strategy 2: AMR ``concept`` + ``semclass`` only (deepslot ignored).
    Missing SC -> ``(concept, _)``.
    """
    concept = str(data.get("concept", "") or "").strip() or "_"
    sc = str(data.get("semclass", "") or "").strip() or "_"
    return f"({concept}, {sc})"


def edge_export_relation(edata: dict[str, Any]) -> str:
    r = str(edata.get("export_rel", edata.get("relation", ""))).strip()
    return r if r else "_"


def graph_strategy_amr(G: nx.DiGraph) -> tuple[dict[Any, str], list[tuple[Any, Any, str]]]:
    nodes: dict[Any, str] = {}
    edges: list[tuple[Any, Any, str]] = []
    for n, d in G.nodes(data=True):
        nodes[n] = amr_baseline_node_label(dict(d))
    for u, v, ed in G.edges(data=True):
        er = str(ed.get("relation", "")).strip() or "_"
        edges.append((u, v, er))
    return nodes, edges


def graph_strategy_enr_node(G: nx.DiGraph) -> tuple[dict[Any, str], list[tuple[Any, Any, str]]]:
    nodes: dict[Any, str] = {}
    edges: list[tuple[Any, Any, str]] = []
    for n, d in G.nodes(data=True):
        nodes[n] = enriched_tuple_node_label(dict(d))
    for u, v, ed in G.edges(data=True):
        er = str(ed.get("relation", "")).strip() or "_"
        edges.append((u, v, er))
    return nodes, edges


def cobald_node_label(data: dict[str, Any]) -> str:
    """
    Strategy 3 (FLAT.md ``cobald``): primary label is CoBaLD semclass when available.
    Constants keep AMR-style labels; missing SC falls back to AMR concept.
    """
    if bool(data.get("is_constant")):
        return amr_baseline_node_label(dict(data))
    sc = str(data.get("semclass", "") or "").strip()
    if sc and sc != "_":
        return sc
    return amr_baseline_node_label(dict(data))


def graph_strategy_cobald(G: nx.DiGraph) -> tuple[dict[Any, str], list[tuple[Any, Any, str]]]:
    """Same AMR edges as ``amr``; node labels follow ``cobald_node_label``."""
    nodes: dict[Any, str] = {}
    edges: list[tuple[Any, Any, str]] = []
    for n, d in G.nodes(data=True):
        nodes[n] = cobald_node_label(dict(d))
    for u, v, ed in G.edges(data=True):
        er = str(ed.get("relation", "")).strip() or "_"
        edges.append((u, v, er))
    return nodes, edges


def invert_vocab(label_to_id: dict[str, int]) -> dict[str, str]:
    """JSON storage: string keys -> string values (FLAT.md)."""
    return {str(v): k for k, v in label_to_id.items()}


def build_label_to_id(all_labels: set[str]) -> dict[str, int]:
    ordered = sorted(all_labels)
    return {lab: i for i, lab in enumerate(ordered)}


def collect_label_sets_from_graphs(
    graphs: list[nx.DiGraph],
) -> tuple[set[str], set[str], set[str], set[str], set[str], set[str]]:
    """Returns (n1, e1, n2, e2, n3, e3) label sets for strategies 1,2,3."""
    n1, e1 = set(), set()
    n2, e2 = set(), set()
    n3, e3 = set(), set()
    for G in graphs:
        n_map, e_list = graph_strategy_amr(G)
        n1.update(str(v) for v in n_map.values())
        e1.update(str(rel) for _u, _v, rel in e_list)

        n_map, e_list = graph_strategy_enr_node(G)
        n2.update(str(v) for v in n_map.values())
        e2.update(str(rel) for _u, _v, rel in e_list)

        n_map, e_list = graph_strategy_cobald(G)
        n3.update(str(v) for v in n_map.values())
        e3.update(str(rel) for _u, _v, rel in e_list)
    return n1, e1, n2, e2, n3, e3


def is_quoted_constant_label(label: str) -> bool:
    return '"' in str(label)


def quoted_constant_occurrence_counts(graphs: list[nx.DiGraph]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for G in graphs:
        for _n, d in G.nodes(data=True):
            if not bool(d.get("is_constant")):
                continue
            lab = amr_baseline_node_label(dict(d))
            if is_quoted_constant_label(lab):
                counts[lab] = counts.get(lab, 0) + 1
    return counts


def remove_singleton_quoted_nodes(
    nodes: dict[Any, str],
    edges: list[tuple[Any, Any, str]],
    quoted_counts: dict[str, int],
) -> tuple[dict[Any, str], list[tuple[Any, Any, str]], int]:
    to_remove: set[Any] = set()
    for nid, lab in nodes.items():
        if is_quoted_constant_label(lab) and quoted_counts.get(lab, 0) <= 1:
            to_remove.add(nid)
    if not to_remove:
        return nodes, edges, 0
    new_nodes = {nid: lab for nid, lab in nodes.items() if nid not in to_remove}
    new_edges = [(u, v, r) for (u, v, r) in edges if u not in to_remove and v not in to_remove]
    return new_nodes, new_edges, len(to_remove)


def normalize_edges_for_gsofia(
    nodes: dict[Any, str],
    edges: list[tuple[Any, Any, str]],
) -> tuple[list[tuple[Any, Any, str]], dict[str, int]]:
    """Self-loop drop + dedupe + :mod removal on multi-label undirected pairs."""
    removed_self_loops = 0
    grouped: dict[tuple[str, str], list[tuple[Any, Any, str]]] = {}
    for u, v, rel in edges:
        if u not in nodes or v not in nodes:
            continue
        if u == v:
            removed_self_loops += 1
            continue
        ku, kv = str(u), str(v)
        pair = (ku, kv) if ku <= kv else (kv, ku)
        grouped.setdefault(pair, []).append((u, v, rel))

    deduped: list[tuple[Any, Any, str]] = []
    removed_duplicate_same_label = 0
    removed_mod_on_conflict = 0

    for pair_edges in grouped.values():
        unique_directed: list[tuple[Any, Any, str]] = []
        # Deduplicate same-label edges on an undirected pair {u,v}.
        # Keep only the first encountered orientation.
        seen_undirected_with_label: set[tuple[str, str, str]] = set()
        for u, v, rel in pair_edges:
            ku, kv = str(u), str(v)
            a, b = (ku, kv) if ku <= kv else (kv, ku)
            key = (a, b, rel)
            if key in seen_undirected_with_label:
                removed_duplicate_same_label += 1
                continue
            seen_undirected_with_label.add(key)
            unique_directed.append((u, v, rel))

        label_set = {rel for _u, _v, rel in unique_directed}
        if len(label_set) > 1 and ":mod" in label_set:
            filtered = []
            for u, v, rel in unique_directed:
                if rel == ":mod":
                    removed_mod_on_conflict += 1
                    continue
                filtered.append((u, v, rel))
            unique_directed = filtered
        deduped.extend(unique_directed)

    stats = {
        "removed_self_loops": removed_self_loops,
        "removed_duplicate_same_label": removed_duplicate_same_label,
        "removed_mod_on_conflict": removed_mod_on_conflict,
    }
    return deduped, stats


def prune_isolated_nodes(
    nodes: dict[Any, str],
    edges: list[tuple[Any, Any, str]],
) -> tuple[dict[Any, str], int]:
    used: set[Any] = set()
    for u, v, _rel in edges:
        if u in nodes:
            used.add(u)
        if v in nodes:
            used.add(v)
    if len(used) == len(nodes):
        return nodes, 0
    pruned = {nid: lab for nid, lab in nodes.items() if nid in used}
    return pruned, len(nodes) - len(pruned)


def emit_transaction_lines(
    nodes: dict[Any, str],
    edges: list[tuple[Any, Any, str]],
    *,
    node_label_to_id: dict[str, int],
    edge_label_to_id: dict[str, int],
    transaction_index: int,
) -> list[str]:
    node_ids = sorted(nodes.keys(), key=str)
    local_id: dict[Any, int] = {nid: i for i, nid in enumerate(node_ids)}
    lines: list[str] = [f"t # {transaction_index}"]
    for nid in node_ids:
        lab = str(nodes.get(nid, "_"))
        if lab not in node_label_to_id:
            raise KeyError(f"Missing node label in vocab: {lab!r}")
        lines.append(f"v {local_id[nid]} {node_label_to_id[lab]}")
    for u, v, rel in edges:
        if u not in local_id or v not in local_id:
            continue
        if rel not in edge_label_to_id:
            raise KeyError(f"Missing edge label in vocab: {rel!r}")
        lines.append(f"e {local_id[u]} {local_id[v]} {edge_label_to_id[rel]}")
    return lines


def load_all_graphs(
    df: pd.DataFrame,
    project_root: str,
) -> tuple[list[tuple[Any, str, nx.DiGraph]], list[str]]:
    """
    Returns (list of (row_index, category, graph), list of error messages for skipped rows).
    """
    out: list[tuple[Any, str, nx.DiGraph]] = []
    errors: list[str] = []
    for i, row in df.sort_index().iterrows():
        cat = _cell_str(row.get("category", "")) or "unknown"
        merged = _cell_str(row.get("merged", ""))
        if not merged:
            errors.append(f"row {i}: empty merged")
            continue
        path = resolve_artifact_path(merged, project_root)
        if not os.path.isfile(path):
            errors.append(f"row {i}: merged file not found {path}")
            continue
        try:
            g = load_merged_graph(path)
        except (OSError, pickle.UnpicklingError, TypeError, KeyError) as e:
            errors.append(f"row {i}: cannot load graph ({e})")
            continue
        out.append((i, cat, g))
    return out, errors


def split_train_test_by_category(
    rows_graphs: list[tuple[Any, str, nx.DiGraph]],
    train_ratio: float,
    seed: int,
) -> tuple[dict[str, list[tuple[Any, str, nx.DiGraph]]], dict[str, list[tuple[Any, str, nx.DiGraph]]]]:
    by_cat: dict[str, list[tuple[Any, str, nx.DiGraph]]] = {}
    for item in rows_graphs:
        by_cat.setdefault(item[1], []).append(item)

    rng = random.Random(seed)
    train: dict[str, list[tuple[Any, str, nx.DiGraph]]] = {}
    test: dict[str, list[tuple[Any, str, nx.DiGraph]]] = {}
    for cat, items in by_cat.items():
        local = list(items)
        rng.shuffle(local)
        n = len(local)
        n_train = int(round(n * train_ratio))
        if n > 1:
            n_train = max(1, min(n - 1, n_train))
        else:
            n_train = 1
        train[cat] = local[:n_train]
        test[cat] = local[n_train:]
    return train, test


def main() -> None:
    default_input = os.path.join(PROJECT_ROOT, "data", "bbc-news-data.csv")
    default_out = os.path.join(PROJECT_ROOT, "flat")

    parser = argparse.ArgumentParser(description="FLAT.md: Gaston flat graphs from merged pickles")
    parser.add_argument("-i", "--input", type=str, default=default_input, help="CSV with category + merged columns")
    parser.add_argument("-o", "--output-dir", type=str, default=default_out, help="Output root (default: flat/)")
    parser.add_argument("--project-root", type=str, default=PROJECT_ROOT, help="Project root for relative paths")
    parser.add_argument("--debug", action="store_true", help="Print skip reasons")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Class-stratified train share in [0,1], e.g. 0.8",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split")
    parser.add_argument(
        "--gsofia-filter-rare-quoted",
        action="store_true",
        help="Drop quoted constant nodes that appear once globally (and incident edges) in emitted .txt only",
    )
    args = parser.parse_args()

    project_root = _abspath(args.project_root)
    out_root = _abspath(args.output_dir)
    inp = _abspath(args.input)
    if not (0.0 <= float(args.train_ratio) <= 1.0):
        print("--train-ratio must be in [0,1]", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(inp):
        print(f"File not found: {inp}", file=sys.stderr)
        sys.exit(1)

    df = utils.read_documents_csv(inp)
    if "merged" not in df.columns:
        print(f"CSV must contain column 'merged'. Found: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    rows_graphs, skip_msgs = load_all_graphs(df, project_root)
    if args.debug and skip_msgs:
        for m in skip_msgs[:50]:
            print(f"[skip] {m}", file=sys.stderr)
        if len(skip_msgs) > 50:
            print(f"[skip] ... and {len(skip_msgs) - 50} more", file=sys.stderr)

    graphs = [g for _, _, g in rows_graphs]
    if not graphs:
        print("No graphs loaded; check merged paths and pickles.", file=sys.stderr)
        sys.exit(1)

    quoted_counts = quoted_constant_occurrence_counts(graphs)
    n_unique_quoted_tokens = len(quoted_counts)

    n1, e1, n2, e2, n3, e3 = collect_label_sets_from_graphs(graphs)
    global_nodes = n1 | n3
    global_edges = e1 | e3
    enr_nodes = n2
    enr_edges = e2

    node_vocab_global = build_label_to_id(global_nodes)
    edge_vocab_global = build_label_to_id(global_edges)
    node_vocab_enr = build_label_to_id(enr_nodes)
    edge_vocab_enr = build_label_to_id(enr_edges)

    os.makedirs(out_root, exist_ok=True)
    for split in ("train", "test"):
        for sub in STRATEGIES:
            os.makedirs(os.path.join(out_root, split, sub), exist_ok=True)
    # Keep dedicated vocab location for enr-node compatibility.
    os.makedirs(os.path.join(out_root, "enr-node"), exist_ok=True)

    with open(os.path.join(out_root, "vocab_nodes_global.json"), "w", encoding="utf-8") as f:
        json.dump(invert_vocab(node_vocab_global), f, ensure_ascii=False, indent=0)
    with open(os.path.join(out_root, "vocab_edges_global.json"), "w", encoding="utf-8") as f:
        json.dump(invert_vocab(edge_vocab_global), f, ensure_ascii=False, indent=0)
    with open(os.path.join(out_root, "enr-node", "vocab_nodes_enr.json"), "w", encoding="utf-8") as f:
        json.dump(invert_vocab(node_vocab_enr), f, ensure_ascii=False, indent=0)
    with open(os.path.join(out_root, "enr-node", "vocab_edges_enr.json"), "w", encoding="utf-8") as f:
        json.dump(invert_vocab(edge_vocab_enr), f, ensure_ascii=False, indent=0)

    train_by_cat, test_by_cat = split_train_test_by_category(
        rows_graphs, train_ratio=float(args.train_ratio), seed=int(args.seed)
    )

    strat_builders: dict[str, tuple[Any, dict[str, int], dict[str, int]]] = {
        "amr": (graph_strategy_amr, node_vocab_global, edge_vocab_global),
        "enr-node": (graph_strategy_enr_node, node_vocab_enr, edge_vocab_enr),
        "cobald": (graph_strategy_cobald, node_vocab_global, edge_vocab_global),
    }

    stat_total_edges_before = 0
    stat_total_edges_after = 0
    stat_removed_self_loops = 0
    stat_removed_duplicate_same_label = 0
    stat_removed_mod_on_conflict = 0
    stat_removed_rare_quoted_nodes = 0
    stat_removed_isolated_nodes = 0
    stat_unique_tokens_by_strategy = {
        "amr": len(n1),
        "enr-node": len(n2),
        "cobald": len(n3),
    }

    split_maps = {"train": train_by_cat, "test": test_by_cat}
    for split_name, by_cat in split_maps.items():
        for strat, (builder, nv, ev) in strat_builders.items():
            strat_dir = os.path.join(out_root, split_name, strat)
            for cat, items in sorted(by_cat.items(), key=lambda x: x[0]):
                fname = safe_category_filename(cat) + ".txt"
                path = os.path.join(strat_dir, fname)
                lines_out: list[str] = []
                t_i = 0
                for _idx, _cat, G in items:
                    nodes, edges = builder(G)
                    stat_total_edges_before += len(edges)

                    if args.gsofia_filter_rare_quoted:
                        nodes, edges, dropped = remove_singleton_quoted_nodes(nodes, edges, quoted_counts)
                        stat_removed_rare_quoted_nodes += dropped

                    edges, norm_stats = normalize_edges_for_gsofia(nodes, edges)
                    stat_removed_self_loops += norm_stats["removed_self_loops"]
                    stat_removed_duplicate_same_label += norm_stats["removed_duplicate_same_label"]
                    stat_removed_mod_on_conflict += norm_stats["removed_mod_on_conflict"]
                    nodes, removed_isolated = prune_isolated_nodes(nodes, edges)
                    stat_removed_isolated_nodes += removed_isolated

                    stat_total_edges_after += len(edges)

                    if not nodes:
                        continue
                    tx_lines = emit_transaction_lines(
                        nodes,
                        edges,
                        node_label_to_id=nv,
                        edge_label_to_id=ev,
                        transaction_index=t_i,
                    )
                    lines_out.extend(tx_lines)
                    t_i += 1
                with open(path, "w", encoding="utf-8", newline="\n") as f:
                    f.write("\n".join(lines_out))
                    if lines_out:
                        f.write("\n")

    n_loaded = len(rows_graphs)
    print(f"Loaded {n_loaded} graph(s), skipped {len(skip_msgs)} row(s).")
    print(
        f"Split: train_ratio={args.train_ratio} seed={args.seed}; "
        f"train={sum(len(v) for v in train_by_cat.values())}, test={sum(len(v) for v in test_by_cat.values())}"
    )
    print(
        "Unique node labels: "
        f"amr={stat_unique_tokens_by_strategy['amr']}, "
        f"enr-node={stat_unique_tokens_by_strategy['enr-node']}, "
        f"cobald={stat_unique_tokens_by_strategy['cobald']}, "
        f"quoted_constant_tokens={n_unique_quoted_tokens}"
    )
    print(
        "Edges: "
        f"before={stat_total_edges_before}, after={stat_total_edges_after}, "
        f"removed_self_loops={stat_removed_self_loops}, "
        f"removed_duplicate_same_label={stat_removed_duplicate_same_label}, "
        f"removed_mod_on_conflict={stat_removed_mod_on_conflict}, "
        f"removed_isolated_nodes={stat_removed_isolated_nodes}"
    )
    if args.gsofia_filter_rare_quoted:
        print(f"gSofia rare-quoted filter: removed_nodes={stat_removed_rare_quoted_nodes}")
    print(f"Wrote Gaston files under: {out_root}/train and {out_root}/test")


if __name__ == "__main__":
    main()
