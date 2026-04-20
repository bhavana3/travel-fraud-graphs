"""
CSV / edge-list exporter.

Produces a directory of CSV files compatible with common graph ML
frameworks and easy to inspect in pandas / R.

Output layout
-------------
<outdir>/
  nodes/
    user.csv
    device.csv
    ip_address.csv
    booking.csv
    flight.csv
    hotel.csv
    review.csv
    payment_card.csv
    loyalty_account.csv
  edges/
    user__made__booking.csv
    user__uses_device__device.csv
    ... (one file per relation)
  metadata.json
"""

from __future__ import annotations
import csv
import json
import os
from pathlib import Path
from typing import Union

from ..graph.builder import GraphData
from ..schema import ALL_NODE_TYPES


def export_csv(data: GraphData, outdir: Union[str, Path]) -> Path:
    """
    Write the graph to a directory of CSV files.

    Parameters
    ----------
    data : GraphData
    outdir : str or Path

    Returns
    -------
    Path to the output directory.
    """
    outdir = Path(outdir)
    node_dir = outdir / "nodes"
    edge_dir = outdir / "edges"
    node_dir.mkdir(parents=True, exist_ok=True)
    edge_dir.mkdir(parents=True, exist_ok=True)

    # --- Node files ---
    for ntype in ALL_NODE_TYPES:
        features   = data.node_features.get(ntype, [])
        labels     = data.node_labels.get(ntype, [])
        ring_ids   = data.node_ring_ids.get(ntype, [])
        ring_types = data.node_ring_types.get(ntype, [])

        if not features:
            continue

        filepath = node_dir / f"{ntype}.csv"
        fieldnames = (
            ["node_id", "is_fraud", "ring_id", "ring_type"]
            + list(features[0].keys())
        )
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for i, (feat, lbl, rid, rtype) in enumerate(
                zip(features, labels, ring_ids, ring_types)
            ):
                row = {"node_id": i, "is_fraud": lbl, "ring_id": rid, "ring_type": rtype}
                row.update(feat)
                writer.writerow(row)

    # --- Edge files ---
    for (src_type, rel, dst_type), edge_list in data.edges.items():
        safe_rel = f"{src_type}__{rel}__{dst_type}".replace(" ", "_")
        filepath = edge_dir / f"{safe_rel}.csv"
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["src_id", "dst_id"])
            writer.writeheader()
            for src, dst in edge_list:
                writer.writerow({"src_id": src, "dst_id": dst})

    # --- Metadata ---
    with open(outdir / "metadata.json", "w") as f:
        json.dump(data.metadata, f, indent=2)

    return outdir
