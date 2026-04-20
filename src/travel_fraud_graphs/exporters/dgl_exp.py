"""
DGL (Deep Graph Library) exporter.

Returns a dgl.heterograph suitable for use with DGL-based GNN models.
Node and edge features are stored as DGL ndata / edata tensors.
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, Tuple, List

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import dgl
    _DGL_AVAILABLE = True
except ImportError:
    _DGL_AVAILABLE = False

from ..graph.builder import GraphData
from ..schema import ALL_NODE_TYPES


def export_dgl(data: GraphData) -> "dgl.DGLGraph":
    """
    Convert GraphData to a DGL heterogeneous graph.

    Raises
    ------
    ImportError if torch or dgl are not installed.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is not installed.  Run: pip install torch")
    if not _DGL_AVAILABLE:
        raise ImportError(
            "dgl is not installed.  See https://www.dgl.ai/pages/start.html"
        )

    # Build edge data dict for dgl.heterograph
    # Format: { (src_type, rel, dst_type): (src_tensor, dst_tensor) }
    graph_data: Dict[Tuple, Tuple] = {}
    for (src_type, rel, dst_type), edge_list in data.edges.items():
        if not edge_list:
            continue
        src_t = torch.tensor([e[0] for e in edge_list], dtype=torch.long)
        dst_t = torch.tensor([e[1] for e in edge_list], dtype=torch.long)
        graph_data[(src_type, rel, dst_type)] = (src_t, dst_t)

    # Node counts per type
    num_nodes_dict: Dict[str, int] = {}
    for ntype in ALL_NODE_TYPES:
        n = len(data.node_labels.get(ntype, []))
        if n > 0:
            num_nodes_dict[ntype] = n

    g = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)

    # Attach node features
    for ntype in ALL_NODE_TYPES:
        features   = data.node_features.get(ntype, [])
        labels     = data.node_labels.get(ntype, [])
        ring_ids   = data.node_ring_ids.get(ntype, [])
        ring_types = data.node_ring_types.get(ntype, [])

        if not features or ntype not in g.ntypes:
            continue

        keys = list(features[0].keys())
        x = torch.tensor(
            [[float(f.get(k, 0.0)) for k in keys] for f in features],
            dtype=torch.float32,
        )
        g.nodes[ntype].data["feat"]      = x
        g.nodes[ntype].data["label"]     = torch.tensor(labels,     dtype=torch.long)
        g.nodes[ntype].data["ring_id"]   = torch.tensor(ring_ids,   dtype=torch.long)
        g.nodes[ntype].data["ring_type"] = torch.tensor(ring_types, dtype=torch.long)

    return g
