"""
Motif / structural pattern analysis.

Fraud rings produce distinctive subgraph motifs:
  - Star motifs:   one device/IP connected to many users  (ATO, ticketing rings)
  - Bipartite cliques: reviewer cluster × hotel cluster  (ghost hotel rings)
  - Transfer chains: loyalty_a → loyalty_b → loyalty_c  (ATO point draining)

This module counts and characterises these motifs, producing:
  1. Ring-type motif fingerprints (per ring type statistics)
  2. Shared-resource concentration (how many users per device/IP)
  3. Transfer chain length distribution
  4. Bipartite clique sizes for review networks

These are the numbers that go in Section 3.3 of the paper and distinguish
TFG from flat tabular fraud datasets.
"""

from __future__ import annotations
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

from ..graph.builder import GraphData
from ..schema import (
    NODE_USER, NODE_DEVICE, NODE_IP, NODE_BOOKING,
    NODE_HOTEL, NODE_REVIEW, NODE_LOYALTY,
    FRAUD_RING_TYPES,
)


def shared_resource_concentration(data: GraphData) -> dict:
    """
    Compute how many users share each device / IP.
    Returns distribution statistics split by fraud vs. legitimate.
    """
    results = {}
    for resource_type in (NODE_DEVICE, NODE_IP):
        rel_name = "uses_device" if resource_type == NODE_DEVICE else "uses_ip"
        edge_key = None
        for (s, r, d) in data.edges.keys():
            if s == NODE_USER and r == rel_name and d == resource_type:
                edge_key = (s, r, d)
                break
        if edge_key is None:
            continue

        resource_user_count: Counter = Counter()
        for src, dst in data.edges[edge_key]:
            resource_user_count[dst] += 1

        # Split by fraud / legit label on the resource node
        res_labels = data.node_labels.get(resource_type, [])
        fraud_counts = [v for k, v in resource_user_count.items()
                        if k < len(res_labels) and res_labels[k] == 1]
        legit_counts = [v for k, v in resource_user_count.items()
                        if k < len(res_labels) and res_labels[k] == 0]

        def _stats(lst):
            if not lst:
                return {"count": 0, "mean": 0, "max": 0, "p95": 0}
            lst = sorted(lst)
            p95_idx = int(len(lst) * 0.95)
            return {
                "count": len(lst),
                "mean":  round(sum(lst) / len(lst), 2),
                "max":   max(lst),
                "p95":   lst[p95_idx],
            }

        results[resource_type] = {
            "fraud":      _stats(fraud_counts),
            "legitimate": _stats(legit_counts),
        }

    return results


def loyalty_transfer_chain_lengths(data: GraphData) -> dict:
    """
    Find all transfer chains in the loyalty graph and return length distribution.
    A chain: loyalty_a → loyalty_b → loyalty_c
    """
    transfer_key = None
    for k in data.edges.keys():
        if k == (NODE_LOYALTY, "transferred_to", NODE_LOYALTY):
            transfer_key = k
            break
    if transfer_key is None:
        return {"chains": 0, "lengths": {}}

    edges = data.edges[transfer_key]
    # Build adjacency
    outgoing: Dict[int, int] = {}
    incoming: set = set()
    for s, d in edges:
        outgoing[s] = d
        incoming.add(d)

    # Find chain roots (nodes with no incoming transfer)
    roots = [n for n in outgoing if n not in incoming]

    chain_lengths: Counter = Counter()
    for root in roots:
        length = 1
        cur = root
        visited = {cur}
        while cur in outgoing:
            nxt = outgoing[cur]
            if nxt in visited:
                break
            cur = nxt
            length += 1
            visited.add(cur)
        chain_lengths[length] += 1

    return {
        "total_transfer_edges": len(edges),
        "unique_chains":        len(roots),
        "length_distribution":  dict(sorted(chain_lengths.items())),
        "max_chain_length":     max(chain_lengths.keys()) if chain_lengths else 0,
    }


def review_bipartite_clique_stats(data: GraphData) -> dict:
    """
    For ghost hotel rings: measure the density of reviewer×hotel bipartite cliques.
    A perfect clique = every reviewer reviewed every hotel in the ring.
    """
    review_hotel_key = (NODE_REVIEW, "about", NODE_HOTEL)
    user_review_key  = (NODE_USER,   "wrote", NODE_REVIEW)

    if review_hotel_key not in data.edges:
        return {}

    # Group reviews by hotel
    hotel_reviews: Dict[int, List[int]] = defaultdict(list)
    for rev_id, hotel_id in data.edges[review_hotel_key]:
        hotel_reviews[hotel_id].append(rev_id)

    # Group reviews by reviewer
    reviewer_reviews: Dict[int, List[int]] = defaultdict(list)
    for user_id, rev_id in data.edges.get(user_review_key, []):
        reviewer_reviews[user_id].append(rev_id)

    # Identify fraud hotels
    hotel_labels = data.node_labels.get(NODE_HOTEL, [])
    fraud_hotels  = [hid for hid, revs in hotel_reviews.items()
                     if hid < len(hotel_labels) and hotel_labels[hid] == 1]

    if not fraud_hotels:
        return {"fraud_hotels_with_reviews": 0}

    fraud_review_counts = [len(hotel_reviews[hid]) for hid in fraud_hotels]
    return {
        "fraud_hotels_with_reviews": len(fraud_hotels),
        "avg_reviews_per_fraud_hotel": round(
            sum(fraud_review_counts) / len(fraud_review_counts), 2),
        "max_reviews_per_fraud_hotel": max(fraud_review_counts),
        "min_reviews_per_fraud_hotel": min(fraud_review_counts),
    }


def ring_type_motif_fingerprints(data: GraphData) -> dict:
    """
    Per ring-type: count nodes, edges, and compute avg user degree.
    Produces the motif fingerprint table in the paper.
    """
    fingerprints: Dict[str, dict] = {}

    for rtype_id, rtype_name in FRAUD_RING_TYPES.items():
        if rtype_id == 0:
            continue

        # Collect all fraud node indices per type in this ring type
        user_ids = [
            i for i, rt in enumerate(data.node_ring_types.get(NODE_USER, []))
            if rt == rtype_id
        ]
        booking_ids = [
            i for i, rt in enumerate(data.node_ring_types.get(NODE_BOOKING, []))
            if rt == rtype_id
        ]
        device_ids = [
            i for i, rt in enumerate(data.node_ring_types.get(NODE_DEVICE, []))
            if rt == rtype_id
        ]
        ip_ids = [
            i for i, rt in enumerate(data.node_ring_types.get(NODE_IP, []))
            if rt == rtype_id
        ]

        user_set    = set(user_ids)
        booking_set = set(booking_ids)

        # Count edges involving these users
        booking_edges = sum(
            1 for s, d in data.edges.get((NODE_USER, "made", NODE_BOOKING), [])
            if s in user_set
        )
        device_edges = sum(
            1 for s, d in data.edges.get((NODE_USER, "uses_device", NODE_DEVICE), [])
            if s in user_set
        )

        fingerprints[rtype_name] = {
            "fraud_users":    len(user_ids),
            "fraud_bookings": len(booking_ids),
            "fraud_devices":  len(device_ids),
            "fraud_ips":      len(ip_ids),
            "booking_edges":  booking_edges,
            "device_edges":   device_edges,
            "avg_bookings_per_user": round(
                booking_edges / max(len(user_ids), 1), 2),
            "avg_devices_per_user": round(
                device_edges / max(len(user_ids), 1), 2),
        }

    return fingerprints


def format_motif_report(data: GraphData) -> str:
    """Run all motif analyses and format as a paper-ready text report."""
    lines = ["=" * 70, "  MOTIF ANALYSIS  —  TravelFraudGraph", "=" * 70]

    # Ring fingerprints
    lines.append("\n[ Table 3: Ring-Type Motif Fingerprints ]\n")
    fp = ring_type_motif_fingerprints(data)
    header = f"{'Ring Type':<30} {'Users':>7} {'Bookings':>9} {'Devices':>8} {'IPs':>6} {'Bk/User':>8} {'Dev/User':>9}"
    lines.append(header)
    lines.append("-" * 80)
    for rname, v in fp.items():
        lines.append(
            f"{rname:<30} {v['fraud_users']:>7} {v['fraud_bookings']:>9} "
            f"{v['fraud_devices']:>8} {v['fraud_ips']:>6} "
            f"{v['avg_bookings_per_user']:>8.1f} {v['avg_devices_per_user']:>9.1f}"
        )

    # Shared resource concentration
    lines.append("\n[ Table 4: Shared Resource Concentration ]\n")
    src = shared_resource_concentration(data)
    for rtype_name, split in src.items():
        lines.append(f"  {rtype_name}")
        for group, stats in split.items():
            lines.append(
                f"    {group:<12} count={stats['count']}  "
                f"mean users/resource={stats['mean']}  "
                f"max={stats['max']}  p95={stats['p95']}"
            )

    # Loyalty chains
    lines.append("\n[ Table 5: Loyalty Transfer Chain Analysis ]\n")
    lc = loyalty_transfer_chain_lengths(data)
    lines.append(f"  Total transfer edges : {lc.get('total_transfer_edges', 0)}")
    lines.append(f"  Unique chains        : {lc.get('unique_chains', 0)}")
    lines.append(f"  Max chain length     : {lc.get('max_chain_length', 0)}")
    for length, cnt in lc.get("length_distribution", {}).items():
        lines.append(f"    length {length}: {cnt} chains")

    # Review cliques
    lines.append("\n[ Table 6: Ghost Hotel Review Clique Stats ]\n")
    rc = review_bipartite_clique_stats(data)
    for k, v in rc.items():
        lines.append(f"  {k}: {v}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)
