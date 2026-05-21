"""Pickle persistence for NetworkX graphs and Graphviz rendering."""

from __future__ import annotations

import os
import pickle
from typing import Any, Mapping, Optional

import networkx as nx


def _abspath(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _alignment_suffix(sent_idx: Optional[int], idx: Optional[int]) -> str:
    if sent_idx is None or idx is None:
        return ""
    if sent_idx < 0 or idx < 0:
        return ""
    return f"\n[s{sent_idx}.t{idx}]"


def _node_label(
    node_id: str,
    data: Mapping[str, Any],
    *,
    show_cobald_attrs: bool = False,
) -> str:
    """Prefer AMR ``concept`` (PropBank sense, e.g. move-01) on instance nodes for labels."""
    name = data.get("name")
    concept = data.get("concept")
    is_constant = bool(data.get("is_constant"))
    if is_constant:
        base = str(name) if name else (str(concept) if concept else str(node_id))
    elif concept:
        base = f"{node_id} / {concept}"
    elif name:
        base = str(name)
    else:
        base = str(node_id)
    out = base + _alignment_suffix(data.get("sent_idx"), data.get("idx"))
    if show_cobald_attrs:
        ds = data.get("deepslot")
        sc = data.get("semclass")
        if ds is not None or sc is not None:
            ds_s = "_" if ds is None or ds == "" else str(ds)
            sc_s = "_" if sc is None or sc == "" else str(sc)
            out += f"\nDS: {ds_s}\nSC: {sc_s}"
    return out


def _edge_label(data: Mapping[str, Any]) -> str:
    return str(data.get("relation", ""))


def render_amr_graph(
    graph: nx.DiGraph,
    output_base: str,
    *,
    title: str = "AMR",
    graphviz_engine: str = "dot",
    show_cobald_attrs: bool = False,
    dpi: int = 600,
    size_inches: str = "20,20",
) -> str:
    """
    Render PNG via the ``graphviz`` package (system Graphviz must be on PATH).
    ``output_base`` is the path without extension (same as ``Digraph.render``).
    If ``show_cobald_attrs`` is True, node labels include CoBaLD ``deepslot`` and ``semclass``
    (after AMR+CoBaLD hybrid enrichment).

    ``dpi`` and ``size_inches`` control bitmap resolution (larger graphs: raise ``dpi`` and/or
    ``size_inches`` for sharper labels when zooming).
    """
    try:
        from graphviz import Digraph
    except ImportError as e:
        raise ImportError(
            "Install: pip install graphviz, and install Graphviz binaries (on PATH)."
        ) from e

    output_base = _abspath(output_base)
    out_dir = os.path.dirname(output_base)
    base_name = os.path.basename(output_base)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    dot = Digraph(comment=title)
    dot.attr(rankdir="TB", dpi=str(dpi), size=size_inches)
    dot.attr("node", fontname="Arial", fontsize="11")
    dot.attr("edge", fontname="Arial", fontsize="10", color="blue")

    for n, data in graph.nodes(data=True):
        is_constant = bool(data.get("is_constant"))
        dot.node(
            str(n),
            _node_label(str(n), data, show_cobald_attrs=show_cobald_attrs),
            shape="box" if is_constant else "ellipse",
        )

    for u, v, data in graph.edges(data=True):
        dot.edge(str(u), str(v), label=_edge_label(data))

    out = dot.render(
        filename=base_name,
        directory=out_dir or ".",
        format="png",
        engine=graphviz_engine,
        cleanup=True,
    )
    return os.path.abspath(out)


def save_graph_pickle(bundle: dict[str, Any], pkl_path: str) -> str:
    """Save a dict with a ``graph`` key (and metadata) to ``.pkl``."""
    pkl_path = _abspath(pkl_path)
    parent = os.path.dirname(pkl_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    return pkl_path


def artifact_paths(output_dir: str, stem: str) -> tuple[str, str]:
    """Paths for ``{stem}_amr.pkl`` and PNG base path (no ``.png`` suffix)."""
    output_dir = _abspath(output_dir)
    pkl = os.path.join(output_dir, f"{stem}_amr.pkl")
    png_base = os.path.join(output_dir, f"{stem}_amr")
    return pkl, png_base
