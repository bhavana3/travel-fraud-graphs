"""
PyTorch Geometric (PyG) exporter.

Returns a torch_geometric.data.HeteroData object suitable for direct
use in GNN training pipelines.

Each node type has:
  x         : float32 feature tensor  [N, F]
  y         : int64 label tensor      [N]     (0=legit, 1=fraud)
  ring_id   : int64 tensor            [N]
  ring_type : int64 tensor            [N]

Each edge type has:
  edge_index : int64 tensor  [2, E]
"""

from __future__ import annotations
from typing import List, Dict, Tuple

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from torch_geometric.data import HeteroData
    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False

from ..graph.builder import GraphData
from ..schema import ALL_NODE_TYPES


def _features_to_tensor(feature_list: List[dict]) -> "torch.Tensor":
    """Convert a list of feature dicts to a float32 tensor."""
    if not feature_list:
        return torch.zeros((0, 1), dtype=torch.float32)
    keys = list(feature_list[0].keys())
    rows = [[float(feat.get(k, 0.0)) for k in keys] for feat in feature_list]
    return torch.tensor(rows, dtype=torch.float32)


def export_pyg(data: GraphData) -> "HeteroData":
    """
    Convert GraphData to a PyTorch Geometric HeteroData object.

    Raises
    ------
    ImportError if torch or torch_geometric are not installed.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is not installed.  Run: pip install torch")
    if not _PYG_AVAILABLE:
        raise ImportError(
            "torch_geometric is not installed.  "
            "See https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"
        )

    hetero = HeteroData()

    # Node tensors
    for ntype in ALL_NODE_TYPES:
        features   = data.node_features.get(ntype, [])
        labels     = data.node_labels.get(ntype, [])
        ring_ids   = data.node_ring_ids.get(ntype, [])
        ring_types = data.node_ring_types.get(ntype, [])

        if not features:
            continue

        hetero[ntype].x         = _features_to_tensor(features)
        hetero[ntype].y         = torch.tensor(labels,     dtype=torch.long)
        hetero[ntype].ring_id   = torch.tensor(ring_ids,   dtype=torch.long)
        hetero[ntype].ring_type = torch.tensor(ring_types, dtype=torch.long)

    # Edge tensors
    for (src_type, rel, dst_type), edge_list in data.edges.items():
        if not edge_list:
            continue
        src_idx = [e[0] for e in edge_list]
        dst_idx = [e[1] for e in edge_list]
        edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
        hetero[src_type, rel, dst_type].edge_index = edge_index

    return hetero
