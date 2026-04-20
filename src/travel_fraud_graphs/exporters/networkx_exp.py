"""
NetworkX exporter.

Returns a single nx.MultiDiGraph where every node has a 'type' attribute
and all feature keys are stored as node attributes.  Node IDs are
globally unique strings of the form "<type>_<local_idx>".
"""

from __future__ import annotations
from typing import Optional

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False

from ..graph.builder import GraphData
from ..schema import ALL_NODE_TYPES


def export_networkx(data: GraphData) -> "nx.MultiDiGraph":
    """
    Convert GraphData to a NetworkX MultiDiGraph.

    Node attributes
    ---------------
    type, is_fraud, ring_id, ring_type, + all feature keys.

    Edge attributes
    ---------------
    relation : str  (e.g. "made", "uses_device", …)
    """
    if not _NX_AVAILABLE:
        raise ImportError(
            "networkx is not installed.  Run: pip install networkx"
        )

    G = nx.MultiDiGraph()

    # Add nodes
    for ntype in ALL_NODE_TYPES:
        features   = data.node_features.get(ntype, [])
        labels     = data.node_labels.get(ntype, [])
        ring_ids   = data.node_ring_ids.get(ntype, [])
        ring_types = data.node_ring_types.get(ntype, [])

        for i, (feat, lbl, rid, rtype) in enumerate(
            zip(features, labels, ring_ids, ring_types)
        ):
            node_id = f"{ntype}_{i}"
            attrs = {
                "type":      ntype,
                "is_fraud":  lbl,
                "ring_id":   rid,
                "ring_type": rtype,
            }
            attrs.update(feat)
            G.add_node(node_id, **attrs)

    # Add edges
    for (src_type, rel, dst_type), edge_list in data.edges.items():
        for src_idx, dst_idx in edge_list:
            G.add_edge(
                f"{src_type}_{src_idx}",
                f"{dst_type}_{dst_idx}",
                relation=rel,
            )

    return G
