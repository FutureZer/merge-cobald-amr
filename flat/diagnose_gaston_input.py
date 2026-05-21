"""
Diagnose Gaston/gSofia input files (flat/*.txt) for structural issues.

Examples:
  python flat/diagnose_gaston_input.py flat/amr/business.txt
  python flat/diagnose_gaston_input.py flat/amr
  python flat/diagnose_gaston_input.py flat --glob "*.txt" --json report.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any


def _abspath(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


@dataclass
class GraphDiag:
    tx_id: int
    line_no: int
    nodes: set[int] = field(default_factory=set)
    edges: list[tuple[int, int, int]] = field(default_factory=list)
    vertex_lines: int = 0
    edge_lines: int = 0
    bad_lines: list[str] = field(default_factory=list)
    duplicate_vertices: int = 0
    out_of_range_vertex_ids: int = 0
    edge_unknown_endpoints: int = 0
    self_loops: int = 0
    duplicate_edges_exact: int = 0
    duplicate_edges_undirected_same_label: int = 0
    duplicate_edges_undirected_any_label: int = 0
    isolated_vertices: int = 0


@dataclass
class FileDiag:
    path: str
    total_lines: int = 0
    graph_count: int = 0
    parse_errors: list[str] = field(default_factory=list)
    header_sequence_errors: list[str] = field(default_factory=list)
    duplicate_tids: list[int] = field(default_factory=list)
    tids_seen_in_order: list[int] = field(default_factory=list)
    graphs: list[GraphDiag] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        s = summarize(self)
        return {
            "path": self.path,
            "total_lines": self.total_lines,
            "graph_count": self.graph_count,
            "parse_errors": self.parse_errors,
            "header_sequence_errors": self.header_sequence_errors,
            "duplicate_tids": self.duplicate_tids,
            "summary": s,
        }


def parse_header(line: str) -> int | None:
    s = line.strip()
    if not s.startswith("t"):
        return None
    parts = s.split()
    if len(parts) == 3 and parts[0] == "t" and parts[1] == "#" and parts[2].isdigit():
        return int(parts[2])
    return None


def parse_int(s: str) -> int | None:
    try:
        return int(s)
    except ValueError:
        return None


def finalize_graph(g: GraphDiag) -> None:
    if g.nodes:
        max_id = max(g.nodes)
        expected = set(range(max_id + 1))
        if g.nodes != expected:
            g.out_of_range_vertex_ids += 1

    edge_set: set[tuple[int, int, int]] = set()
    undir_same_label: set[tuple[int, int, int]] = set()
    undir_any_label: set[tuple[int, int]] = set()
    touched_nodes: set[int] = set()
    for u, v, el in g.edges:
        touched_nodes.add(u)
        touched_nodes.add(v)
        e = (u, v, el)
        if e in edge_set:
            g.duplicate_edges_exact += 1
        else:
            edge_set.add(e)

        und = (u, v) if u <= v else (v, u)
        und_l = (und[0], und[1], el)
        if und_l in undir_same_label:
            g.duplicate_edges_undirected_same_label += 1
        else:
            undir_same_label.add(und_l)

        if und in undir_any_label:
            g.duplicate_edges_undirected_any_label += 1
        else:
            undir_any_label.add(und)
    g.isolated_vertices = len(g.nodes - touched_nodes)


def diagnose_file(path: str) -> FileDiag:
    path = _abspath(path)
    diag = FileDiag(path=path)
    current: GraphDiag | None = None

    seen_tids: set[int] = set()
    prev_tid: int | None = None

    with open(path, encoding="utf-8") as f:
        for ln, raw in enumerate(f, start=1):
            diag.total_lines += 1
            line = raw.strip()
            if not line:
                continue

            tid = parse_header(line)
            if tid is not None:
                if current is not None:
                    finalize_graph(current)
                    diag.graphs.append(current)
                current = GraphDiag(tx_id=tid, line_no=ln)
                diag.tids_seen_in_order.append(tid)
                if tid in seen_tids:
                    diag.duplicate_tids.append(tid)
                seen_tids.add(tid)
                if prev_tid is not None and tid <= prev_tid:
                    diag.header_sequence_errors.append(
                        f"line {ln}: tid {tid} is not strictly increasing (prev {prev_tid})"
                    )
                prev_tid = tid
                continue

            if current is None:
                diag.parse_errors.append(f"line {ln}: content before first header: {line[:120]}")
                continue

            parts = line.split()
            kind = parts[0]
            if kind == "v":
                if len(parts) < 3:
                    current.bad_lines.append(f"line {ln}: bad vertex line {line!r}")
                    continue
                vid = parse_int(parts[1])
                if vid is None:
                    current.bad_lines.append(f"line {ln}: non-int vertex id {parts[1]!r}")
                    continue
                current.vertex_lines += 1
                if vid in current.nodes:
                    current.duplicate_vertices += 1
                current.nodes.add(vid)
            elif kind == "e":
                if len(parts) < 4:
                    current.bad_lines.append(f"line {ln}: bad edge line {line!r}")
                    continue
                u = parse_int(parts[1])
                v = parse_int(parts[2])
                el = parse_int(parts[3])
                if u is None or v is None or el is None:
                    current.bad_lines.append(f"line {ln}: non-int edge fields in {line!r}")
                    continue
                current.edge_lines += 1
                if u == v:
                    current.self_loops += 1
                current.edges.append((u, v, el))
                if u not in current.nodes or v not in current.nodes:
                    current.edge_unknown_endpoints += 1
            else:
                current.bad_lines.append(f"line {ln}: unknown line kind {kind!r}")

    if current is not None:
        finalize_graph(current)
        diag.graphs.append(current)

    diag.graph_count = len(diag.graphs)
    return diag


def collect_targets(root_or_file: str, glob_pattern: str) -> list[str]:
    p = _abspath(root_or_file)
    if os.path.isfile(p):
        return [p]
    if not os.path.isdir(p):
        return []
    out: list[str] = []
    for cur_root, _dirs, files in os.walk(p):
        for name in files:
            if glob_pattern == "*.txt" and not name.lower().endswith(".txt"):
                continue
            out.append(os.path.join(cur_root, name))
    out.sort()
    return out


def summarize(diag: FileDiag) -> dict[str, int]:
    s = {
        "parse_errors": len(diag.parse_errors),
        "header_sequence_errors": len(diag.header_sequence_errors),
        "duplicate_tids": len(diag.duplicate_tids),
        "graphs_with_bad_lines": 0,
        "duplicate_vertices": 0,
        "edge_unknown_endpoints": 0,
        "self_loops": 0,
        "duplicate_edges_exact": 0,
        "duplicate_edges_undirected_same_label": 0,
        "duplicate_edges_undirected_any_label": 0,
        "out_of_range_vertex_ids": 0,
        "graphs_with_self_loops": 0,
        "graphs_with_multi_edges_same_label_uv": 0,
        "graphs_with_multi_edges_different_label_uv": 0,
        "isolated_vertices": 0,
        "graphs_with_isolated_vertices": 0,
    }
    for g in diag.graphs:
        if g.bad_lines:
            s["graphs_with_bad_lines"] += 1
        if g.self_loops > 0:
            s["graphs_with_self_loops"] += 1
        if g.duplicate_edges_undirected_same_label > 0:
            s["graphs_with_multi_edges_same_label_uv"] += 1
        # Different labels on same {u,v}: any undirected duplicate minus same-label duplicates.
        if g.duplicate_edges_undirected_any_label > g.duplicate_edges_undirected_same_label:
            s["graphs_with_multi_edges_different_label_uv"] += 1
        if g.isolated_vertices > 0:
            s["graphs_with_isolated_vertices"] += 1
        s["duplicate_vertices"] += g.duplicate_vertices
        s["edge_unknown_endpoints"] += g.edge_unknown_endpoints
        s["self_loops"] += g.self_loops
        s["duplicate_edges_exact"] += g.duplicate_edges_exact
        s["duplicate_edges_undirected_same_label"] += g.duplicate_edges_undirected_same_label
        s["duplicate_edges_undirected_any_label"] += g.duplicate_edges_undirected_any_label
        s["out_of_range_vertex_ids"] += g.out_of_range_vertex_ids
        s["isolated_vertices"] += g.isolated_vertices
    s["has_self_loops"] = 1 if s["self_loops"] > 0 else 0
    s["has_multi_edges_same_label_uv"] = 1 if s["duplicate_edges_undirected_same_label"] > 0 else 0
    s["has_multi_edges_different_label_uv"] = (
        1 if s["duplicate_edges_undirected_any_label"] > s["duplicate_edges_undirected_same_label"] else 0
    )
    s["has_isolated_vertices"] = 1 if s["isolated_vertices"] > 0 else 0
    return s


def print_report(diag: FileDiag, top_k: int = 5) -> None:
    s = summarize(diag)
    print(f"\n== {diag.path} ==")
    print(f"graphs={diag.graph_count} lines={diag.total_lines}")
    print(
        "issues:"
        f" parse={s['parse_errors']}"
        f", header={s['header_sequence_errors']}"
        f", duplicate_tids={s['duplicate_tids']}"
        f", bad_lines={s['graphs_with_bad_lines']}"
        f", dup_vertices={s['duplicate_vertices']}"
        f", edge_unknown_endpoints={s['edge_unknown_endpoints']}"
        f", self_loops={s['self_loops']}"
        f", has_self_loops={s['has_self_loops']}"
        f", graphs_with_self_loops={s['graphs_with_self_loops']}"
        f", dup_edges_exact={s['duplicate_edges_exact']}"
        f", dup_edges_undir_same_label={s['duplicate_edges_undirected_same_label']}"
        f", has_multi_edges_same_label_uv={s['has_multi_edges_same_label_uv']}"
        f", graphs_with_multi_edges_same_label_uv={s['graphs_with_multi_edges_same_label_uv']}"
        f", dup_edges_undir_any_label={s['duplicate_edges_undirected_any_label']}"
        f", has_multi_edges_different_label_uv={s['has_multi_edges_different_label_uv']}"
        f", graphs_with_multi_edges_different_label_uv={s['graphs_with_multi_edges_different_label_uv']}"
        f", isolated_vertices={s['isolated_vertices']}"
        f", has_isolated_vertices={s['has_isolated_vertices']}"
        f", graphs_with_isolated_vertices={s['graphs_with_isolated_vertices']}"
        f", non_contiguous_vertex_ids={s['out_of_range_vertex_ids']}"
    )
    if diag.duplicate_tids:
        print(f"duplicate tids examples: {diag.duplicate_tids[:top_k]}")
    if diag.header_sequence_errors:
        print("header order examples:")
        for msg in diag.header_sequence_errors[:top_k]:
            print(f"  - {msg}")
    if diag.parse_errors:
        print("parse examples:")
        for msg in diag.parse_errors[:top_k]:
            print(f"  - {msg}")

    noisy = [g for g in diag.graphs if g.bad_lines]
    if noisy:
        print("graphs with unknown/non-gaston lines examples:")
        for g in noisy[:top_k]:
            print(f"  - t # {g.tx_id} (line {g.line_no}): {g.bad_lines[:2]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose flat Gaston input files for gSofia.")
    parser.add_argument("path", type=str, help="Path to .txt file or root directory")
    parser.add_argument("--glob", type=str, default="*.txt", help="File filter when path is directory")
    parser.add_argument("--json", type=str, default=None, help="Optional JSON report path")
    args = parser.parse_args()

    targets = collect_targets(args.path, args.glob)
    if not targets:
        print("No files found.")
        raise SystemExit(1)

    all_diags: list[FileDiag] = []
    for p in targets:
        d = diagnose_file(p)
        all_diags.append(d)
        print_report(d)

    if args.json:
        out = _abspath(args.json)
        parent = os.path.dirname(out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        payload = [d.to_dict() for d in all_diags]
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nWrote JSON report: {out}")


if __name__ == "__main__":
    main()

