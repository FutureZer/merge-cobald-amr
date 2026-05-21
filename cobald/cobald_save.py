"""Pickle persistence for CoBaLD NetworkX graphs and Graphviz (pydot) rendering."""

from __future__ import annotations

import os
import pickle
from typing import Any, Mapping

import networkx as nx


def _abspath(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


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
    """Paths for ``{stem}_cobald.pkl`` and PNG path (with ``.png`` suffix)."""
    output_dir = _abspath(output_dir)
    pkl = os.path.join(output_dir, f"{stem}_cobald.pkl")
    png = os.path.join(output_dir, f"{stem}_cobald.png")
    return pkl, png


def _node_viz_label(node_id: str, attrs: Mapping[str, Any]) -> str:
    if str(node_id) == "document" or attrs.get("token") == "DOCUMENT":
        return "DOCUMENT"
    token = attrs.get("token", "")
    lem = attrs.get("lemma", "")
    sc = attrs.get("semclass", attrs.get("sc", "_"))
    ds = attrs.get("deepslot", attrs.get("ds", "_"))
    idx = attrs.get("idx", "")
    return f"{token} ({node_id})\nSC: {sc}\nDS: {ds}\nidx: {idx}"


def _edge_viz_label(attrs: Mapping[str, Any]) -> str:
    rel = attrs.get("relation", attrs.get("eud_rel", ""))
    ds = attrs.get("ds_label", attrs.get("deepslot", ""))
    return f"{rel} / {ds}" if ds else str(rel)


def render_cobald_graph(
    graph: nx.DiGraph,
    output_png: str,
    *,
    title: str = "CoBaLD",
    dpi: int = 600,
    size_inches: str = "20,20",
) -> str:
    """
    Render PNG via NetworkX -> pydot -> Graphviz (system Graphviz on PATH).
    Requires: pip install pydot

    ``dpi`` / ``size_inches`` are passed to Graphviz for sharper PNG output on large graphs.
    """
    output_png = _abspath(output_png)
    parent = os.path.dirname(output_png)
    if parent:
        os.makedirs(parent, exist_ok=True)

    try:
        import networkx.drawing.nx_pydot as nx_pydot
    except ImportError as e:
        raise ImportError("Install pydot: pip install pydot (and system Graphviz).") from e

    G = graph.copy()
    for node, attrs in G.nodes(data=True):
        if str(node) == "document" or attrs.get("token") == "DOCUMENT":
            attrs["label"] = "DOCUMENT"
            attrs["style"] = "filled"
            attrs["fillcolor"] = "#e8e8ff"
            attrs["shape"] = "ellipse"
        elif attrs.get("lemma") == "<ROOT>" or str(attrs.get("token", "")) == "ROOT":
            attrs["label"] = "ROOT"
            attrs["style"] = "filled"
            attrs["fillcolor"] = "#f0f0f0"
            attrs["shape"] = "box"
        else:
            attrs["label"] = _node_viz_label(str(node), attrs)
            attrs["shape"] = "box"
            attrs["style"] = "filled"
            attrs["fillcolor"] = "#e6f2ff"
        attrs["fontname"] = "Helvetica"
        attrs["fontsize"] = "11"

    for _u, _v, attrs in G.edges(data=True):
        attrs["label"] = _edge_viz_label(attrs)
        attrs["fontname"] = "Helvetica"
        attrs["fontsize"] = "10"
        attrs["fontcolor"] = "darkred"
        attrs["color"] = "grey"
        attrs["dir"] = "forward"

    pdot = nx_pydot.to_pydot(G)
    pdot.set_graph_defaults(
        rankdir="TB",
        overlap="false",
        splines="true",
        label=title,
        labelloc="t",
        fontname="Helvetica",
        fontsize="16",
        dpi=str(dpi),
        size=size_inches,
    )
    pdot.write_png(output_png)
    return output_png
