"""
Homophily analysis for the Travel Fraud Graph.

Homophily (the tendency of similar nodes to be connected) is a key graph
property that GNN reviewers check:
  - High fraud-fraud homophily validates that the ring topology produces
    realistic, learnable subgraphs.
  - Low cross-type homophily isolates which edge types carry fraud signal.

We compute node homophily (Zhu et al., 2020) per edge type:
  h = (# edges where both endpoints share label) / total edges of that type

A value near 1.0 means same-label endpoints dominate (homophilic).
A value near 0.0 means cross-label edges dominate (heterophilic).
GNN fraud detection literature consistently shows fraud rings produce
homophilic subgraphs — this module produces the table that proves it.
"""

from __future__ import annotations
from typing import Dict, Tuple, List

from ..graph.builder import GraphData


def compute_edge_homophily(data: GraphData) -> Dict[str, float]:
    """
    Compute node homophily for each edge type.

    Returns
    -------
    dict  edge_key -> homophily_score (0.0 - 1.0)
    """
    results = {}
    for (src_type, rel, dst_type), edge_list in data.edges.items():
        if not edge_list:
            continue
        src_labels = data.node_labels.get(src_type, [])
        dst_labels = data.node_labels.get(dst_type, [])
        if not src_labels or not dst_labels:
            continue

        same = sum(
            1 for s, d in edge_list
            if s < len(src_labels) and d < len(dst_labels)
            and src_labels[s] == dst_labels[d]
        )
        total = len(edge_list)
        key = f"{src_type}__{rel}__{dst_type}"
        results[key] = round(same / total, 4) if total > 0 else 0.0

    return results


def compute_fraud_subgraph_density(data: GraphData) -> Dict[str, float]:
    """
    For each edge type, compute what fraction of edges connect two fraud nodes.
    High density = the fraud ring topology forms dense cliques (good for GNN motif learning).
    """
    results = {}
    for (src_type, rel, dst_type), edge_list in data.edges.items():
        if not edge_list:
            continue
        src_labels = data.node_labels.get(src_type, [])
        dst_labels = data.node_labels.get(dst_type, [])
        if not src_labels or not dst_labels:
            continue

        fraud_fraud = sum(
            1 for s, d in edge_list
            if s < len(src_labels) and d < len(dst_labels)
            and src_labels[s] == 1 and dst_labels[d] == 1
        )
        key = f"{src_type}__{rel}__{dst_type}"
        results[key] = round(fraud_fraud / len(edge_list), 4)

    return results


def format_homophily_table(
    homophily: Dict[str, float],
    fraud_density: Dict[str, float],
) -> str:
    """Format paper-ready Table 3: Homophily and Fraud Density per edge type."""
    header = f"{'Edge Type':<55} {'Homophily':>10} {'Fraud-Fraud Density':>20}"
    sep = "-" * 87
    lines = [sep, header, sep]
    for key in sorted(homophily.keys()):
        h = homophily.get(key, 0.0)
        fd = fraud_density.get(key, 0.0)
        lines.append(f"{key:<55} {h:>10.4f} {fd:>20.4f}")
    lines.append(sep)
    return "\n".join(lines)
