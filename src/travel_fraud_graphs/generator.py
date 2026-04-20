"""
High-level generator API — the main public entry point.

Quick start
-----------
from travel_fraud_graphs import generate

# Generate a medium-scale dataset
data = generate(scale="medium", seed=42)

# Export to CSV
from travel_fraud_graphs.exporters import export_csv
export_csv(data, "./tfg_output")

# Export to PyG HeteroData
from travel_fraud_graphs.exporters import export_pyg
hetero = export_pyg(data)

# Get dataset statistics
from travel_fraud_graphs.stats import compute_stats, format_report
stats = compute_stats(data)
print(format_report(stats))
"""

from __future__ import annotations
from typing import Optional

from .graph.builder import TravelFraudGraphBuilder, GraphData


# Predefined scales matching paper benchmark configurations
SCALE_PRESETS = {
    "toy": dict(
        n_users=500, n_hotels=50, n_flights=80,
        n_ticketing_rings=3, n_ghost_hotel_rings=2, n_ato_rings=2,
    ),
    "small": dict(
        n_users=2_000, n_hotels=200, n_flights=300,
        n_ticketing_rings=8, n_ghost_hotel_rings=6, n_ato_rings=6,
    ),
    "medium": dict(
        n_users=10_000, n_hotels=1_000, n_flights=1_500,
        n_ticketing_rings=30, n_ghost_hotel_rings=25, n_ato_rings=25,
    ),
    "large": dict(
        n_users=50_000, n_hotels=5_000, n_flights=8_000,
        n_ticketing_rings=100, n_ghost_hotel_rings=80, n_ato_rings=80,
    ),
    "xlarge": dict(
        n_users=200_000, n_hotels=20_000, n_flights=30_000,
        n_ticketing_rings=300, n_ghost_hotel_rings=250, n_ato_rings=250,
    ),
}


def generate(
    scale: str = "medium",
    seed: int = 42,
    **override_kwargs,
) -> GraphData:
    """
    Generate a Travel Fraud Graph dataset.

    Parameters
    ----------
    scale : str
        One of "toy", "small", "medium", "large", "xlarge".
        Controls number of users, hotels, flights, and fraud rings.
    seed : int
        Random seed for reproducibility.
    **override_kwargs
        Any parameter accepted by TravelFraudGraphBuilder
        (e.g., n_users=5000, n_ticketing_rings=20).

    Returns
    -------
    GraphData
        Heterogeneous graph with node features, labels, ring metadata,
        and edge lists for all 12 relation types.

    Examples
    --------
    >>> from travel_fraud_graphs import generate
    >>> data = generate(scale="small", seed=0)
    >>> print(data.metadata["fraud_user_ratio"])
    """
    if scale not in SCALE_PRESETS:
        raise ValueError(
            f"Unknown scale '{scale}'.  Choose from: {list(SCALE_PRESETS.keys())}"
        )
    config = {**SCALE_PRESETS[scale], **override_kwargs, "seed": seed}
    builder = TravelFraudGraphBuilder(**config)
    return builder.build()
