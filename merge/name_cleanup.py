"""
Post-merge cleanup for mining: merge duplicate ``:name`` / literal structures, then
strip intermediate ``/name`` instance nodes (lift ``:op*`` edges to the entity parent).

Called from ``merge/merge.py`` after ``merge_graph.merge_document_graph``. Operates on
an in-memory ``networkx.DiGraph`` (mutates the graph in place).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterator, Mapping

import networkx as nx

from merge_graph import _concept_base_key, merge_nodes_atomic, node_concept


def edge_relation(edata: dict[str, Any] | nx.classes.coreviews.AtlasView) -> str:
    return str(edata.get("relation", ""))


def normalize_literal_concept(concept: Any) -> str:
    """Normalize quoted AMR surface strings for equality (case-insensitive)."""
    c = str(concept).strip()
    if len(c) >= 2 and c[0] == '"' and c[-1] == '"':
        c = c[1:-1]
    return c.strip().casefold()


def exact_literal_label(concept: Any) -> str:
    """
    Surface string for constants (Graphviz **box** nodes: ``is_constant``), preserving case.
    Outer quotes removed; used for exact token / label matching (no casefold).
    """
    c = str(concept).strip()
    if len(c) >= 2 and c[0] == '"' and c[-1] == '"':
        c = c[1:-1]
    return c.strip()


def _anchor_tuple(data: Mapping[str, Any]) -> tuple[int, int] | None:
    anch = data.get("anchor")
    if not anch or not isinstance(anch, (list, tuple)) or len(anch) < 2:
        return None
    try:
        return (int(anch[0]), int(anch[1]))
    except (TypeError, ValueError):
        return None


def _is_name_instance_node(G: nx.DiGraph, n: str) -> bool:
    if n not in G.nodes:
        return False
    return str(G.nodes[n].get("concept", "")).strip().lower() == "name" and not G.nodes[n].get(
        "is_constant"
    )


def name_subgraph_signature(G: nx.DiGraph, n: str) -> tuple[tuple[str, str], ...]:
    """
    Ordered tuple of (``:op*`` relation, normalized literal) for constant children
    of a ``/name`` instance node. Empty if no usable ops.
    """
    if n not in G.nodes:
        return ()
    parts: list[tuple[str, str]] = []
    for _, succ, ed in G.out_edges(n, data=True):
        rel = edge_relation(ed)
        if not rel.startswith(":op"):
            continue
        if succ not in G.nodes:
            continue
        if not G.nodes[succ].get("is_constant"):
            continue
        norm = normalize_literal_concept(G.nodes[succ].get("concept", ""))
        if not norm:
            continue
        parts.append((rel, norm))
    parts.sort(key=lambda x: x[0])
    return tuple(parts)


def iter_name_structures(G: nx.DiGraph) -> Iterator[tuple[str, str, tuple[tuple[str, str], ...]]]:
    """Yield ``(parent, name_node, signature)`` for each ``parent -:name-> /name`` arc."""
    for p, n, ed in G.edges(data=True):
        if edge_relation(ed) != ":name":
            continue
        if not _is_name_instance_node(G, n):
            continue
        sig = name_subgraph_signature(G, n)
        if not sig:
            continue
        yield p, n, sig


def strip_name_intermediates(G: nx.DiGraph) -> int:
    """
    For each ``/name`` instance node ``N``: remove ``N`` and attach its ``:op*`` children
    directly to each parent of ``N`` that used ``:name`` (preserve child edge data).
    Returns count of removed name nodes.
    """
    removed = 0
    for n in list(G.nodes()):
        if n not in G.nodes:
            continue
        if not _is_name_instance_node(G, n):
            continue
        parents = [
            p
            for p in G.predecessors(n)
            if G.has_edge(p, n) and edge_relation(G[p][n]) == ":name"
        ]
        child_edges: list[tuple[str, dict[str, Any]]] = []
        for succ in G.successors(n):
            if not G.has_edge(n, succ):
                continue
            child_edges.append((succ, dict(G[n][succ])))

        for p in parents:
            for succ, edata in child_edges:
                if not G.has_edge(p, succ):
                    G.add_edge(p, succ, **edata)
        G.remove_node(n)
        removed += 1
    return removed


def cleanup_duplicate_name_structures(G: nx.DiGraph, *, debug: bool = False) -> None:
    """
    Merge duplicate named-entity subgraphs, strip ``/name`` nodes, then dedupe literals.

    1. Same surface signature **and** same parent concept (e.g. two ``company`` nodes
       with ``:name`` -> ``"Rosneft"``): merge parent instance nodes (and thus their
       subgraphs) via ``merge_nodes_atomic``.
    2. Under one parent, merge sibling ``/name`` nodes that share the same signature.
    3. Same signature but **different** parent concepts (e.g. ``person`` vs ``company``):
       merge duplicate literal constants, then merge all ``/name`` nodes for that
       signature into one (several parents may keep ``:name`` -> one name node).
    4. Strip ``/name`` nodes: parents connect directly to string nodes with ``:op*``.
    5. **Constant (box) nodes:** merge duplicates with the same character ``anchor``
       and the **exact** same surface label (same mention in text).
    6. **Constant nodes:** merge duplicates that share the same graph parent and the
       **exact** same label (Graphviz boxes = ``is_constant``).
    """
    # --- Round 1: merge parents that share (signature, parent concept base) ---
    while True:
        buckets: dict[tuple[Any, str], list[tuple[str, str]]] = defaultdict(list)
        for p, n, sig in iter_name_structures(G):
            if p == "document":
                continue
            b = _concept_base_key(node_concept(G, p))
            buckets[(sig, b)].append((p, n))

        changed = False
        for _key, pairs in list(buckets.items()):
            parents_ordered: list[str] = []
            seen: set[str] = set()
            for p, _n in pairs:
                if p not in G.nodes:
                    continue
                if p in seen:
                    continue
                seen.add(p)
                parents_ordered.append(p)
            if len(parents_ordered) < 2:
                continue
            master = parents_ordered[0]
            for other in parents_ordered[1:]:
                if other in G.nodes and master in G.nodes and other != master:
                    if debug:
                        print(
                            f"[name_cleanup] merge parent {other!r} "
                            f"({node_concept(G, other)!r}) -> {master!r}"
                        )
                    merge_nodes_atomic(G, master, other)
                    changed = True
        if not changed:
            break

    # --- Round 2: sibling /name nodes under same parent, same signature ---
    while True:
        changed = False
        for p in list(G.nodes()):
            if p not in G.nodes or p == "document":
                continue
            name_children: list[tuple[str, tuple[tuple[str, str], ...]]] = []
            for n in list(G.successors(p)):
                if not G.has_edge(p, n):
                    continue
                if edge_relation(G[p][n]) != ":name":
                    continue
                if not _is_name_instance_node(G, n):
                    continue
                sig = name_subgraph_signature(G, n)
                if not sig:
                    continue
                name_children.append((n, sig))
            by_sig: dict[tuple[tuple[str, str], ...], list[str]] = defaultdict(list)
            for n, sig in name_children:
                by_sig[sig].append(n)
            for _sig, nodes in by_sig.items():
                uniq = [x for x in nodes if x in G.nodes]
                if len(uniq) < 2:
                    continue
                master_n = uniq[0]
                for o in uniq[1:]:
                    if o in G.nodes and master_n in G.nodes and o != master_n:
                        if debug:
                            print(f"[name_cleanup] merge sibling /name {o!r} -> {master_n!r}")
                        merge_nodes_atomic(G, master_n, o)
                        changed = True
        if not changed:
            break

    # --- Round 3: same signature, different parent concepts — literals + /name nodes ---
    by_sig_global: dict[tuple[tuple[str, str], ...], list[str]] = defaultdict(list)
    for p, n, sig in iter_name_structures(G):
        if p not in G.nodes or n not in G.nodes:
            continue
        by_sig_global[sig].append(n)

    for sig, name_nodes in by_sig_global.items():
        uniq_n = []
        for nn in name_nodes:
            if nn in G.nodes and nn not in uniq_n:
                uniq_n.append(nn)
        if len(uniq_n) < 2:
            continue

        literals_by_norm: dict[str, list[str]] = defaultdict(list)
        for nn in uniq_n:
            if nn not in G.nodes:
                continue
            for _nn, succ, ed in G.out_edges(nn, data=True):
                if not edge_relation(ed).startswith(":op"):
                    continue
                if succ not in G.nodes:
                    continue
                if not G.nodes[succ].get("is_constant"):
                    continue
                k = normalize_literal_concept(G.nodes[succ].get("concept", ""))
                if not k:
                    continue
                literals_by_norm[k].append(succ)

        for _k, lits in literals_by_norm.items():
            uniq_lit = []
            for lit in lits:
                if lit in G.nodes and lit not in uniq_lit:
                    uniq_lit.append(lit)
            if len(uniq_lit) < 2:
                continue
            lm = uniq_lit[0]
            for o in uniq_lit[1:]:
                if o in G.nodes and lm in G.nodes and o != lm:
                    if debug:
                        print(f"[name_cleanup] merge literal {o!r} -> {lm!r} ({_k!r})")
                    merge_nodes_atomic(G, lm, o)

        uniq_n = [nn for nn in uniq_n if nn in G.nodes]
        if len(uniq_n) < 2:
            continue
        master_n = uniq_n[0]
        for o in uniq_n[1:]:
            if o in G.nodes and master_n in G.nodes and o != master_n:
                if debug:
                    print(f"[name_cleanup] merge /name (cross-type) {o!r} -> {master_n!r} sig={sig!r}")
                merge_nodes_atomic(G, master_n, o)

    n_strip = strip_name_intermediates(G)
    if debug and n_strip:
        print(f"[name_cleanup] stripped {n_strip} /name intermediate node(s)")

    # --- Constant (box) cleanup: same text span + exact label, then same parent + label ---
    while True:
        ch_span = _merge_duplicate_constants_same_span_exact_label(G, debug=debug)
        ch_parent = _merge_duplicate_constants_same_parent_exact_label(G, debug=debug)
        if not ch_span and not ch_parent:
            break


def _merge_duplicate_constants_same_span_exact_label(G: nx.DiGraph, *, debug: bool) -> bool:
    """
    Merge constant nodes that denote the **same character span** in the document and
    the **same** surface string (exact label, case-sensitive after quote strip).
    """
    buckets: dict[tuple[int, int, str], list[str]] = defaultdict(list)
    for n in list(G.nodes()):
        if n not in G.nodes:
            continue
        d = G.nodes[n]
        if not d.get("is_constant"):
            continue
        lab = exact_literal_label(d.get("concept", ""))
        if not lab:
            continue
        at = _anchor_tuple(d)
        if at is None:
            continue
        buckets[(*at, lab)].append(n)

    changed = False
    for key, nodes in buckets.items():
        uniq = list(dict.fromkeys(x for x in nodes if x in G.nodes))
        if len(uniq) < 2:
            continue
        master = uniq[0]
        for other in uniq[1:]:
            if other in G.nodes and master in G.nodes and other != master:
                if debug:
                    print(
                        f"[name_cleanup] merge const (same span+label) {other!r} -> {master!r} "
                        f"key={key!r}"
                    )
                merge_nodes_atomic(G, master, other)
                changed = True
    return changed


def _merge_duplicate_constants_same_parent_exact_label(G: nx.DiGraph, *, debug: bool) -> bool:
    """
    Merge **box** (constant) children of the same parent when their surface labels match
    exactly (same rule as ``exact_literal_label``).
    """
    changed = False
    for p in list(G.nodes()):
        if p not in G.nodes or p == "document":
            continue
        by_label: dict[str, list[str]] = defaultdict(list)
        for succ in list(G.successors(p)):
            if not G.has_edge(p, succ) or succ not in G.nodes:
                continue
            if not G.nodes[succ].get("is_constant"):
                continue
            lab = exact_literal_label(G.nodes[succ].get("concept", ""))
            if not lab:
                continue
            by_label[lab].append(succ)

        for lab, nodes in by_label.items():
            uniq = list(dict.fromkeys(n for n in nodes if n in G.nodes))
            if len(uniq) < 2:
                continue
            master = uniq[0]
            for other in uniq[1:]:
                if other in G.nodes and master in G.nodes and other != master:
                    if debug:
                        print(
                            f"[name_cleanup] merge const (same parent+label) {other!r} -> {master!r} "
                            f"parent={p!r} label={lab!r}"
                        )
                    merge_nodes_atomic(G, master, other)
                    changed = True
    return changed
