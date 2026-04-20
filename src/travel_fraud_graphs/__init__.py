"""
travel_fraud_graphs
===================
Graph-Based Synthetic Fraud Ring Generator for Travel Networks.

Generates labeled heterogeneous property graphs containing three travel-domain
fraud ring types: ticketing fraud rings, ghost hotel schemes, and account
takeover rings.

Quick start
-----------
>>> from travel_fraud_graphs import generate
>>> data = generate(scale="medium", seed=42)
>>> data.metadata
"""

from .generator import generate, SCALE_PRESETS
from .graph.builder import GraphData, TravelFraudGraphBuilder

__version__ = "0.1.0"
__all__ = [
    "generate",
    "SCALE_PRESETS",
    "GraphData",
    "TravelFraudGraphBuilder",
]
