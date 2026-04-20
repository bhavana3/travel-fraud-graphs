"""
Dataset statistics reporter.

Computes and formats a comprehensive summary of the generated graph,
including class balance, ring size distributions, feature statistics,
and graph structural metrics — all tables you'd include in the
dataset paper (Section 3: Dataset Statistics).
"""

from __future__ import annotations
import json
from collections import Counter
from typing import Dict, List

from ..graph.builder import GraphData
from ..schema import ALL_NODE_TYPES, FRAUD_RING_TYPES, NODE_USER, NODE_BOOKING, NODE_HOTEL


def compute_stats(data: GraphData) -> dict:
    """
    Compute a statistics dictionary from a GraphData object.

    Returns
    -------
    dict with keys: node_counts, edge_counts, fraud_ratios,
    ring_stats, structural, feature_stats
    """
    stats: dict = {}

    # --- Node counts ---
    node_counts = {}
    for ntype in ALL_NODE_TYPES:
        labels = data.node_labels.get(ntype, [])
        node_counts[ntype] = {
            "total":      len(labels),
            "legitimate": labels.count(0),
            "fraud":      labels.count(1),
            "fraud_pct":  round(labels.count(1) / max(len(labels), 1) * 100, 2),
        }
    stats["node_counts"] = node_counts

    # --- Edge counts ---
    edge_counts = {}
    for rel, edges in data.edges.items():
        key = f"{rel[0]}__{rel[1]}__{rel[2]}"
        edge_counts[key] = len(edges)
    stats["edge_counts"] = edge_counts
    stats["total_edges"] = sum(edge_counts.values())

    # --- Fraud ratios ---
    fraud_ratios = {
        ntype: round(
            data.node_labels.get(ntype, []).count(1) /
            max(len(data.node_labels.get(ntype, [])), 1), 4
        )
        for ntype in ALL_NODE_TYPES
    }
    stats["fraud_ratios"] = fraud_ratios

    # --- Ring stats ---
    ring_type_counts: Counter = Counter()
    ring_sizes: Dict[int, List[int]] = {}  # ring_id -> user count

    user_ring_ids   = data.node_ring_ids.get(NODE_USER, [])
    user_ring_types = data.node_ring_types.get(NODE_USER, [])
    user_labels     = data.node_labels.get(NODE_USER, [])

    for lbl, rid, rtype in zip(user_labels, user_ring_ids, user_ring_types):
        if lbl == 1:
            ring_type_counts[rtype] += 1
            ring_sizes.setdefault(rid, []).append(1)

    ring_type_summary = {}
    for rtype_id, count in ring_type_counts.items():
        rname = FRAUD_RING_TYPES.get(rtype_id, f"type_{rtype_id}")
        ring_type_summary[rname] = {"fraud_users": count}

    ring_size_list = [len(v) for v in ring_sizes.values()]
    stats["ring_stats"] = {
        "total_rings":       len(ring_sizes),
        "ring_type_summary": ring_type_summary,
        "ring_size_min":     min(ring_size_list) if ring_size_list else 0,
        "ring_size_max":     max(ring_size_list) if ring_size_list else 0,
        "ring_size_mean":    round(sum(ring_size_list) / max(len(ring_size_list), 1), 2),
    }

    # --- Structural metrics ---
    # Degree stats for user nodes
    user_out_degree = Counter()
    for (src_type, rel, dst_type), edges in data.edges.items():
        if src_type == NODE_USER:
            for src, _ in edges:
                user_out_degree[src] += 1

    degrees = list(user_out_degree.values())
    stats["structural"] = {
        "user_avg_out_degree":  round(sum(degrees) / max(len(degrees), 1), 2),
        "user_max_out_degree":  max(degrees) if degrees else 0,
        "user_min_out_degree":  min(degrees) if degrees else 0,
        "total_node_types":     len([t for t in ALL_NODE_TYPES
                                     if data.node_labels.get(t)]),
        "total_edge_types":     len(data.edges),
    }

    # --- Metadata passthrough ---
    stats["metadata"] = data.metadata

    return stats


def format_report(stats: dict) -> str:
    """
    Format statistics as a human-readable text report.
    Suitable for pasting into a paper or README.
    """
    lines = []
    lines.append("=" * 64)
    lines.append("  TRAVEL FRAUD GRAPH  —  Dataset Statistics Report")
    lines.append("=" * 64)

    lines.append("\n[ Node Counts & Fraud Ratios ]\n")
    lines.append(f"{'Node Type':<20} {'Total':>8} {'Fraud':>8} {'Fraud %':>10}")
    lines.append("-" * 50)
    for ntype, c in stats["node_counts"].items():
        if c["total"] > 0:
            lines.append(
                f"{ntype:<20} {c['total']:>8,} {c['fraud']:>8,} {c['fraud_pct']:>9.2f}%"
            )
    lines.append(f"\n{'Total edges':<30} {stats['total_edges']:>10,}")

    lines.append("\n\n[ Edge Type Counts ]\n")
    for rel, cnt in sorted(stats["edge_counts"].items(), key=lambda x: -x[1]):
        if cnt > 0:
            lines.append(f"  {rel:<55} {cnt:>8,}")

    r = stats["ring_stats"]
    lines.append(f"\n\n[ Ring Statistics ]\n")
    lines.append(f"  Total rings injected : {r['total_rings']}")
    lines.append(f"  Ring size  min / mean / max : "
                 f"{r['ring_size_min']} / {r['ring_size_mean']} / {r['ring_size_max']}")
    lines.append("\n  Ring types:")
    for rname, info in r["ring_type_summary"].items():
        lines.append(f"    {rname:<35}  fraud users = {info['fraud_users']}")

    s = stats["structural"]
    lines.append(f"\n\n[ Structural Metrics ]\n")
    lines.append(f"  Node types        : {s['total_node_types']}")
    lines.append(f"  Edge types        : {s['total_edge_types']}")
    lines.append(f"  User out-degree   mean = {s['user_avg_out_degree']}  "
                 f"max = {s['user_max_out_degree']}")

    m = stats.get("metadata", {})
    lines.append(f"\n\n[ Generation Config ]\n")
    lines.append(f"  Seed              : {m.get('seed', 'n/a')}")
    lines.append(f"  Ticketing rings   : {m.get('n_ticketing_rings', 0)}")
    lines.append(f"  Ghost hotel rings : {m.get('n_ghost_hotel_rings', 0)}")
    lines.append(f"  ATO rings         : {m.get('n_ato_rings', 0)}")
    lines.append(f"  Fraud user ratio  : {m.get('fraud_user_ratio', 0):.2%}")

    lines.append("\n" + "=" * 64)
    return "\n".join(lines)


def save_report(stats: dict, filepath: str):
    """Save both the JSON stats and the human-readable report."""
    import pathlib
    p = pathlib.Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(str(p) + ".json", "w") as f:
        json.dump(stats, f, indent=2)

    # Text
    with open(str(p) + ".txt", "w") as f:
        f.write(format_report(stats))
