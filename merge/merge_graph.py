"""
Document graph merging (MERGE.md): FastCoref clusters, anchor-based AMR node mapping,
entity / predicate consolidation, edge deduplication.
"""

from __future__ import annotations

import copy
import re
from collections import defaultdict, deque
from typing import Any, Mapping

import networkx as nx

PREDICATE_SENSE_RE = re.compile(r".+-\d+$")


def edge_relation(edata: Mapping[str, Any]) -> str:
    return str(edata.get("relation", ""))


def node_concept(G: nx.DiGraph, n: str) -> str:
    return str(G.nodes[n].get("concept", ""))


def anchor_overlaps_span(anchor: Any, start: int, end: int) -> bool:
    if not anchor or not isinstance(anchor, (list, tuple)) or len(anchor) < 2:
        return False
    a0, a1 = int(anchor[0]), int(anchor[1])
    return max(0, min(a1, end) - max(a0, start)) > 0


def anchor_contains_token(anchor: Any, token) -> bool:
    if not anchor or not isinstance(anchor, (list, tuple)) or len(anchor) < 2:
        return False
    a0, a1 = int(anchor[0]), int(anchor[1])
    t0, t1 = int(token.idx), int(token.idx) + len(token.text)
    return a0 <= t0 and a1 >= t1


def resolve_name_entity_parent(G: nx.DiGraph, n: str) -> str:
    """
    MERGE.md head heuristic P1: if under :op of :name, return the category node (e.g. person).
    """
    for p, _, ed in G.in_edges(n, data=True):
        rel = edge_relation(ed)
        if rel.startswith(":op"):
            for gp, _, ed2 in G.in_edges(p, data=True):
                if edge_relation(ed2) == ":name":
                    return gp
        if rel == ":name":
            return p
    return n


def is_child_of_name_subgraph(G: nx.DiGraph, n: str) -> bool:
    """True if ``n`` is under a ``:name`` branch (e.g. Bill under person :name)."""
    for p, _, ed in G.in_edges(n, data=True):
        rel = edge_relation(ed)
        if rel == ":name":
            return True
        if rel.startswith(":op"):
            for gp, _, ed2 in G.in_edges(p, data=True):
                if edge_relation(ed2) == ":name":
                    return True
    return False


def candidates_for_span(G: nx.DiGraph, start: int, end: int) -> list[str]:
    out: list[str] = []
    for n, d in G.nodes(data=True):
        if n == "document":
            continue
        if anchor_overlaps_span(d.get("anchor"), start, end):
            out.append(n)
    return out


def best_amr_node_raw(
    G: nx.DiGraph,
    doc,
    start_char: int,
    end_char: int,
) -> str | None:
    """Map coref span to one AMR node by anchors + spaCy head (lift comes later)."""
    span = doc.char_span(start_char, end_char, alignment_mode="expand")
    if span is None or len(span) == 0:
        return None

    cands = candidates_for_span(G, start_char, end_char)
    if not cands:
        return None

    head = span.root
    for n in cands:
        if anchor_contains_token(G.nodes[n].get("anchor"), head):
            return n
    return cands[0]


def is_pronoun_node(G: nx.DiGraph, doc, n: str) -> bool:
    d = G.nodes[n]
    concept = str(d.get("concept", "")).lower().strip('"')
    if concept == "person":
        has_name_child = any(
            edge_relation(ed) == ":name" for _u, _v, ed in G.out_edges(n, data=True)
        )
        if not has_name_child:
            return True

    anch = d.get("anchor")
    if not anch or len(anch) < 2:
        return False
    for t in doc:
        if int(t.idx) >= int(anch[0]) and int(t.idx) + len(t.text) <= int(anch[1]):
            if t.pos_ == "PRON":
                return True
            if t.lemma_.lower() in ("my", "your", "his", "her", "its", "our", "their"):
                return True
    return False


def is_predicate_node(G: nx.DiGraph, n: str) -> bool:
    c = node_concept(G, n)
    if not c or G.nodes[n].get("is_constant"):
        return False
    return bool(PREDICATE_SENSE_RE.match(c))


def is_wordnet_hypernym(ancestor_concept: str, desc_concept: str) -> bool:
    """True if ``ancestor`` (more general) appears on a noun hypernym path of ``desc``."""
    if not ancestor_concept or not desc_concept:
        return False
    try:
        from nltk.corpus import wordnet as wn
    except ImportError:
        return False

    a = ancestor_concept.split("-")[0].lower().replace(" ", "_")
    d = desc_concept.split("-")[0].lower().replace(" ", "_")
    if a == d:
        return False

    anc_names = {lm.name().lower() for syn in wn.synsets(a, pos=wn.NOUN) for lm in syn.lemmas()}
    if not anc_names:
        return False

    for dsyn in wn.synsets(d, pos=wn.NOUN):
        for path in dsyn.hypernym_paths():
            for parent in path:
                for lm in parent.lemmas():
                    if lm.name().lower() in anc_names:
                        return True
    return False


def find_hypernym_parent_among_ancestors(
    G: nx.DiGraph,
    n: str,
    *,
    max_depth: int = 4,
) -> str | None:
    """
    WordNet hypernym among (transitive) predecessors of ``n``, not among sibling mentions.
    Predicate nodes may sit on the path and are skipped as hypernym candidates but traversed.
    """
    if n not in G.nodes:
        return None
    mention_concept = node_concept(G, n)
    if not mention_concept:
        return None

    frontier = {n}
    seen: set[str] = set()
    for _ in range(max_depth):
        nxt: set[str] = set()
        for cur in frontier:
            for p in G.predecessors(cur):
                if p == "document" or p in seen:
                    continue
                seen.add(p)
                if is_predicate_node(G, p):
                    nxt.add(p)
                    continue
                pc = node_concept(G, p)
                if pc and is_wordnet_hypernym(pc, mention_concept):
                    return p
                nxt.add(p)
        frontier = nxt
        if not frontier:
            break
    return None


def ensure_mod_edge(G: nx.DiGraph, u: str, v: str) -> bool:
    """Add ``u -:mod-> v`` if there is no edge ``u -> v`` yet (``DiGraph``: one edge only)."""
    if u not in G.nodes or v not in G.nodes or u == v:
        return False
    if G.has_edge(u, v):
        return False
    G.add_edge(u, v, relation=":mod")
    return True


def redirect_incident_edges_to_hub(G: nx.DiGraph, n: str, hub: str) -> None:
    """Move all incident edges of ``n`` onto ``hub``, except the structural ``hub -> n`` link."""
    if n not in G.nodes or hub not in G.nodes or n == hub:
        return

    for u, _, data in list(G.in_edges(n, data=True)):
        if u == hub:
            continue
        d = dict(data)
        if not G.has_edge(u, hub):
            G.add_edge(u, hub, **d)
        G.remove_edge(u, n)

    for _, v, data in list(G.out_edges(n, data=True)):
        if v == hub:
            continue
        d = dict(data)
        if not G.has_edge(hub, v):
            G.add_edge(hub, v, **d)
        G.remove_edge(n, v)


def lift_mention_to_hub(
    G: nx.DiGraph,
    n: str,
    *,
    debug: bool = False,
) -> str:
    """
    Named: parent of ``:name`` (category ``person``). Unnamed: WordNet hypernym among
    ancestors; add ``hub -:mod-> mention`` and move mention's other edges to ``hub``.
    """
    if n not in G.nodes:
        return n

    if is_child_of_name_subgraph(G, n):
        h = resolve_name_entity_parent(G, n)
        if debug and h != n:
            print(
                f"    [lift] named mention {n} ({node_concept(G, n)!r}) "
                f"-> hub {h} ({node_concept(G, h)!r})"
            )
        return h

    p = find_hypernym_parent_among_ancestors(G, n)
    if p is not None:
        added = ensure_mod_edge(G, p, n)
        redirect_incident_edges_to_hub(G, n, p)
        if debug:
            print(
                f"    [lift] hypernym parent {p} ({node_concept(G, p)!r}) "
                f"for mention {n} ({node_concept(G, n)!r})"
                + ("; added :mod" if added else "")
            )
        return p

    if debug:
        print(f"    [lift] no change for {n} ({node_concept(G, n)!r})")
    return n


def _fresh_subtype_node_id(G: nx.DiGraph) -> str:
    i = 0
    while True:
        nid = f"_subtype_mod_{i}"
        if nid not in G.nodes:
            return nid
        i += 1


def _concept_base_key(concept: str) -> str:
    return concept.split("-")[0].lower() if concept else ""


def merge_entity_hub_into_master(
    G: nx.DiGraph,
    master: str,
    other: str,
) -> list[str]:
    """
    Merge ``other`` into ``master`` after optional ``:mod``; if ``other`` vanishes,
    reattach its concept as a constant ``:mod`` child when it refines the master.
    Returns concepts recorded as newly attached ``:mod`` children (for debug).
    """
    added_labels: list[str] = []
    if other not in G or master not in G or master == other:
        return added_labels

    cm, co = node_concept(G, master), node_concept(G, other)
    if (
        cm
        and co
        and _concept_base_key(cm) != _concept_base_key(co)
        and is_wordnet_hypernym(cm, co)
    ):
        ensure_mod_edge(G, master, other)
    subtype_concept = node_concept(G, other)
    was_constant = bool(G.nodes[other].get("is_constant"))

    merge_nodes_atomic(G, master, other)

    master_key = _concept_base_key(node_concept(G, master))
    need_leaf = (
        subtype_concept
        and not was_constant
        and not bool(PREDICATE_SENSE_RE.match(subtype_concept))
        and _concept_base_key(subtype_concept) != master_key
    )
    if need_leaf:
        new_id = _fresh_subtype_node_id(G)
        base = subtype_concept.split("-")[0]
        G.add_node(
            new_id,
            concept=subtype_concept,
            is_constant=True,
            name=f"c/{base}",
        )
        if not G.has_edge(master, new_id):
            G.add_edge(master, new_id, relation=":mod")
        added_labels.append(subtype_concept)

    return added_labels


def pick_entity_master(nodes: list[str], G: nx.DiGraph) -> str:
    """Most abstract entity: prefer a node that is WordNet hypernym of others."""
    nodes = [n for n in nodes if n in G.nodes and not is_predicate_node(G, n)]
    if not nodes:
        return ""
    for cand in nodes:
        if all(
            cand == o or is_wordnet_hypernym(node_concept(G, cand), node_concept(G, o))
            for o in nodes
        ):
            return cand
    return min(nodes, key=lambda x: len(node_concept(G, x)))


def merge_nodes_atomic(G: nx.DiGraph, master: str, other: str) -> None:
    """Redirect all edges from ``other`` to ``master`` and remove ``other``."""
    if other not in G or master not in G or master == other:
        return

    for u, _, data in list(G.in_edges(other, data=True)):
        if u == other:
            continue
        if u == master:
            continue
        d = dict(data)
        if G.has_edge(u, master):
            continue
        G.add_edge(u, master, **d)

    for _, v, data in list(G.out_edges(other, data=True)):
        if v == other:
            continue
        if v == master:
            continue
        d = dict(data)
        if G.has_edge(master, v):
            continue
        G.add_edge(master, v, **d)

    G.remove_node(other)


def dedupe_edges_same_relation(G: nx.DiGraph) -> int:
    """MERGE §4: duplicate (u,v,relation) collapse; no-op for ``DiGraph`` (at most one u→v)."""
    return 0


def structural_cleanup_merged_graph(G: nx.DiGraph, *, debug: bool = False) -> None:
    """
    Drop the synthetic ``document`` root and any nodes with no incident edges
    (repeat until stable — removing ``document`` or isolates can expose new isolates).
    """
    removed_doc = False
    if "document" in G.nodes:
        G.remove_node("document")
        removed_doc = True

    total_isolates = 0
    while True:
        isolates = [n for n in G.nodes() if G.degree(n) == 0]
        if not isolates:
            break
        total_isolates += len(isolates)
        G.remove_nodes_from(isolates)

    if debug and (removed_doc or total_isolates):
        print(
            f"[debug] structural cleanup: removed document={removed_doc}, "
            f"isolated_nodes={total_isolates}"
        )


def run_coref_clusters(text: str, device: str = "cpu"):
    try:
        from fastcoref import FCoref
    except ImportError as e:
        raise ImportError(
            "Install fastcoref for merge: pip install fastcoref"
        ) from e

    model = FCoref(device=device)
    pred = model.predict(texts=[text.strip()])[0]
    return pred.get_clusters(as_strings=False)


def merge_document_graph(
    G: nx.DiGraph,
    text: str,
    *,
    spacy_model: str = "en_core_web_sm",
    device: str = "cpu",
    debug: bool = False,
) -> nx.DiGraph:
    """
    Full MERGE.md pipeline on a hybrid AMR graph (copy).
    """
    try:
        import nltk

        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    except Exception:
        pass

    import spacy

    nlp = spacy.load(spacy_model)
    doc = nlp(text.strip())
    txt = doc.text
    H = copy.deepcopy(G)

    clusters_char = run_coref_clusters(text, device=device)
    if debug:
        print(f"[debug] FastCoref: {len(clusters_char)} cluster(s), span text:")
        for ci, spans in enumerate(clusters_char):
            bits: list[str] = []
            for a, b in spans:
                if 0 <= a <= b <= len(txt):
                    bits.append(f"{txt[a:b]!r} [{a},{b})")
                else:
                    bits.append(f"<?> [{a},{b})")
            print(f"  cluster {ci}: " + " | ".join(bits))

    for cluster_idx, spans in enumerate(clusters_char):
        raw_nodes: list[str] = []
        raw_trace: list[tuple[str, str]] = []
        for start_c, end_c in spans:
            raw = best_amr_node_raw(H, doc, start_c, end_c)
            span_t = txt[start_c:end_c] if 0 <= start_c <= end_c <= len(txt) else ""
            if raw and raw in H.nodes:
                raw_trace.append((span_t, raw))
                if raw not in raw_nodes:
                    raw_nodes.append(raw)

        if len(raw_nodes) < 2:
            continue

        if debug:
            print(f"[debug] cluster {cluster_idx}: AMR node per coref span (raw, before lift)")
            for span_t, nid in raw_trace:
                print(
                    f"    span {span_t!r} -> {nid} "
                    f"(concept={node_concept(H, nid)!r})"
                )

        hubs: list[str] = []
        if debug:
            print(f"  [lift] cluster {cluster_idx}:")
        for n in raw_nodes:
            hub = lift_mention_to_hub(H, n, debug=debug)
            if hub in H.nodes and hub not in hubs:
                hubs.append(hub)

        if len(hubs) < 2:
            if debug:
                print(
                    f"[debug] cluster {cluster_idx}: skip (fewer than 2 distinct hubs "
                    f"after lift: {hubs})"
                )
            continue

        if debug:
            print(
                f"[debug] cluster {cluster_idx}: entity/predicate hubs marked for merge (*): "
                + ", ".join(f"{h}* ({node_concept(H, h)!r})" for h in hubs)
            )

        pronoun_hubs = [h for h in hubs if is_pronoun_node(H, doc, h)]
        informative_hubs = [h for h in hubs if h not in pronoun_hubs]

        if not informative_hubs and not pronoun_hubs:
            continue

        hub_set = set(hubs)
        all_in_edges: list[tuple[str, dict[str, Any]]] = []
        for n in hubs:
            if n not in H.nodes:
                continue
            for source, _, edata in list(H.in_edges(n, data=True)):
                if source not in hub_set:
                    all_in_edges.append((source, dict(edata)))

        for inf in informative_hubs:
            if inf not in H.nodes:
                continue
            for source, edata in all_in_edges:
                if not H.has_edge(source, inf):
                    H.add_edge(source, inf, **edata)

        for p_node in pronoun_hubs:
            if p_node in H.nodes:
                H.remove_node(p_node)

        informative_hubs = [h for h in informative_hubs if h in H.nodes]
        preds = [h for h in informative_hubs if is_predicate_node(H, h)]
        ents = [h for h in informative_hubs if not is_predicate_node(H, h)]

        if debug:
            print(
                f"[debug] cluster {cluster_idx}: split for merge — "
                f"entity hubs {ents}, predicate hubs {preds}"
            )

        by_concept: dict[str, list[str]] = defaultdict(list)
        for n in preds:
            by_concept[node_concept(H, n)].append(n)
        for _c, nodes in by_concept.items():
            if len(nodes) < 2:
                continue
            master_p = nodes[0]
            if debug:
                print(
                    f"[debug] cluster {cluster_idx}: predicate merge (same sense {_c!r}): "
                    f"master {master_p}, absorb {nodes[1:]}"
                )
            for other in nodes[1:]:
                if other in H.nodes:
                    merge_nodes_atomic(H, master_p, other)

        if len(ents) >= 2:
            master_e = pick_entity_master(ents, H)
            if master_e:
                cluster_mod_added: list[str] = []
                for other in ents:
                    if other != master_e and other in H.nodes:
                        added = merge_entity_hub_into_master(H, master_e, other)
                        cluster_mod_added.extend(added)
                if debug:
                    mod_children = [
                        f"{v} ({node_concept(H, v)!r})"
                        for v in H.successors(master_e)
                        if H.has_edge(master_e, v)
                        and edge_relation(H[master_e][v]) == ":mod"
                    ]
                    print(
                        f"[debug] cluster {cluster_idx}: entity merge master = "
                        f"{master_e} ({node_concept(H, master_e)!r})"
                    )
                    print(
                        f"  :mod attributes on master: "
                        f"{', '.join(mod_children) if mod_children else '(none)'}"
                    )
                    if cluster_mod_added:
                        print(
                            f"  (subtype constants attached this step: {cluster_mod_added})"
                        )

    nodes_to_check = deque(n for n in H.nodes() if n != "document")
    processed_for_triples: set[str] = set()

    def concept_base(nid: str) -> str:
        c = node_concept(H, nid).split("-")[0].lower()
        return c

    while nodes_to_check:
        current = nodes_to_check.popleft()
        if current not in H.nodes:
            continue

        parents_by_rel: dict[tuple[str, str], list[str]] = defaultdict(list)
        for parent, _, edata in H.in_edges(current, data=True):
            if parent == "document":
                continue
            rel = edge_relation(edata)
            parents_by_rel[(concept_base(parent), rel)].append(parent)

        for (_pb, rel), p_nodes in parents_by_rel.items():
            if len(p_nodes) > 1:
                master_p = p_nodes[0]
                for other_p in p_nodes[1:]:
                    if other_p in H.nodes:
                        merge_nodes_atomic(H, master_p, other_p)
                if master_p not in processed_for_triples:
                    nodes_to_check.append(master_p)
        processed_for_triples.add(current)

    dedupe_edges_same_relation(H)

    structural_cleanup_merged_graph(H, debug=debug)

    if debug:
        print(
            f"[debug] merged graph: nodes={H.number_of_nodes()} edges={H.number_of_edges()}"
        )
    return H
