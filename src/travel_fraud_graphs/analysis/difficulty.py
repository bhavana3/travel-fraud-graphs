"""
Controlled Difficulty Study utilities.

A benchmark's defining property — vs. a plain dataset — is controllable
evaluation difficulty. This module provides the experiment scaffold for
the paper's Figure 3 and supports Evaluative Claim E2:

  "A model's AUC on TFG declines predictably as ring size decreases,
   establishing a difficulty axis that no flat fraud dataset provides."

Usage (Databricks Notebook 05):
    from travel_fraud_graphs.analysis.difficulty import DifficultyStudy
    study = DifficultyStudy(model_fn=train_rgcn, seed=42)
    results = study.run()
    print(study.format_table(results))
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..generator import generate


@dataclass
class DifficultyConfig:
    """
    One point on the difficulty curve.

    ring_size_target : int   — mean users per ring
    fraud_rate       : float — fraction of users that are fraudulent
    n_rings_per_type : int   — rings of each type to inject
    """
    ring_size_target:  int   = 10
    fraud_rate:        float = 0.15
    n_rings_per_type:  int   = 10
    seed:              int   = 42
    scale:             str   = "small"

    def to_generate_kwargs(self) -> dict:
        """Convert to kwargs for generate()."""
        kwargs = {
            "scale": self.scale,
            "seed":  self.seed,
            "n_ticketing_rings":   self.n_rings_per_type,
            "n_ghost_hotel_rings": self.n_rings_per_type,
            "n_ato_rings":         self.n_rings_per_type,
        }
        if self.ring_size_target > 0:
            kwargs["ring_size_target"] = self.ring_size_target
        return kwargs


# Pre-defined difficulty axis: vary ring size from very small (hard) to large (easy)
RING_SIZE_AXIS = [3, 5, 8, 12, 20, 30]

# Pre-defined fraud rate axis
FRAUD_RATE_AXIS = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30]


def build_difficulty_configs(
    axis: str = "ring_size",
    seed: int = 42,
    scale: str = "small",
) -> List[DifficultyConfig]:
    """
    Build a list of DifficultyConfig objects along one difficulty axis.

    Parameters
    ----------
    axis  : "ring_size" | "fraud_rate"
    seed  : random seed
    scale : dataset scale preset

    Returns
    -------
    List[DifficultyConfig] ordered from hardest to easiest.
    """
    # Scale-aware budget: total fraud users ≈ 15% of n_users_in_scale
    _scale_user_budget = {
        "toy": 75, "small": 300, "medium": 1500,
        "large": 7500, "xlarge": 30000,
    }
    fraud_budget = _scale_user_budget.get(scale, 300)

    configs = []
    if axis == "ring_size":
        for rs in RING_SIZE_AXIS:
            # Keep total fraud nodes ≈ fraud_budget across all three ring types
            n_rings = max(2, fraud_budget // (rs * 3))
            configs.append(DifficultyConfig(
                ring_size_target=rs,
                n_rings_per_type=n_rings,
                seed=seed,
                scale=scale,
            ))
    elif axis == "fraud_rate":
        for fr in FRAUD_RATE_AXIS:
            configs.append(DifficultyConfig(
                fraud_rate=fr,
                n_rings_per_type=8,
                seed=seed,
                scale=scale,
            ))
    else:
        raise ValueError(f"Unknown axis: {axis}")
    return configs


@dataclass
class DifficultyResult:
    """Results for one point on the difficulty curve."""
    config:        DifficultyConfig
    auc_overall:   float = 0.0
    ap_overall:    float = 0.0
    f1_overall:    float = 0.0
    auc_ticketing: float = 0.0   # AUC restricted to ticketing ring users
    auc_ghost:     float = 0.0   # AUC restricted to ghost hotel ring users
    auc_ato:       float = 0.0   # AUC restricted to ATO ring users
    n_fraud_users: int   = 0
    fraud_ratio:   float = 0.0
    notes:         str   = ""


class DifficultyStudy:
    """
    Orchestrates a controlled difficulty experiment for the paper.

    Parameters
    ----------
    model_fn : Callable that receives a GraphData object and returns a dict
               {"auc": float, "ap": float, "f1": float,
                "auc_ticketing": float, "auc_ghost": float, "auc_ato": float}
               This function trains a model from scratch on the graph and
               returns held-out test metrics.
    axis     : "ring_size" or "fraud_rate"
    seed     : random seed
    scale    : dataset scale preset
    """

    def __init__(
        self,
        model_fn: Optional[Callable] = None,
        axis:     str = "ring_size",
        seed:     int = 42,
        scale:    str = "small",
    ):
        self.model_fn = model_fn
        self.axis     = axis
        self.seed     = seed
        self.scale    = scale
        self.configs  = build_difficulty_configs(axis, seed, scale)

    def run(self) -> List[DifficultyResult]:
        """
        Run the full difficulty study.

        If model_fn is None, returns stub results with dataset statistics only.
        """
        results = []
        for cfg in self.configs:
            print(f"  Generating: ring_size~{cfg.ring_size_target}  "
                  f"n_rings={cfg.n_rings_per_type} per type ...")
            data = generate(**cfg.to_generate_kwargs())

            n_fraud = sum(data.node_labels.get("user", []))
            n_total = len(data.node_labels.get("user", []))
            fraud_ratio = round(n_fraud / max(n_total, 1), 4)

            if self.model_fn is not None:
                metrics = self.model_fn(data)
                result = DifficultyResult(
                    config=cfg,
                    auc_overall=metrics.get("auc", 0.0),
                    ap_overall=metrics.get("ap", 0.0),
                    f1_overall=metrics.get("f1", 0.0),
                    auc_ticketing=metrics.get("auc_ticketing", 0.0),
                    auc_ghost=metrics.get("auc_ghost", 0.0),
                    auc_ato=metrics.get("auc_ato", 0.0),
                    n_fraud_users=n_fraud,
                    fraud_ratio=fraud_ratio,
                )
            else:
                # Stub: returns dataset stats only (useful for quick sanity check)
                result = DifficultyResult(
                    config=cfg,
                    n_fraud_users=n_fraud,
                    fraud_ratio=fraud_ratio,
                    notes="model_fn not provided — dataset stats only",
                )
            results.append(result)
            print(f"    fraud_users={n_fraud}  fraud_ratio={fraud_ratio:.1%}  "
                  f"auc={result.auc_overall:.4f}")

        return results

    @staticmethod
    def format_table(results: List[DifficultyResult], axis: str = "ring_size") -> str:
        """Format results as a paper-ready table (Figure 3 source data)."""
        lines = [
            "=" * 80,
            "  Figure 3: TFG Controlled Difficulty Study",
            f"  Axis: {axis}",
            "=" * 80,
        ]
        header = (
            f"{'Ring Size':>10} {'Fraud%':>8} "
            f"{'AUC-All':>9} {'AUC-Tick':>10} {'AUC-Ghost':>11} {'AUC-ATO':>9}"
        )
        lines.append(header)
        lines.append("-" * 80)
        for r in results:
            lines.append(
                f"{r.config.ring_size_target:>10} "
                f"{r.fraud_ratio:>8.1%} "
                f"{r.auc_overall:>9.4f} "
                f"{r.auc_ticketing:>10.4f} "
                f"{r.auc_ghost:>11.4f} "
                f"{r.auc_ato:>9.4f}"
            )
        lines.append("=" * 80)
        lines.append(
            "Interpretation: AUC should decrease as ring_size decreases,\n"
            "confirming that smaller rings provide weaker graph-structural signal.\n"
            "Differential AUC by ring type supports Evaluative Claim E3.\n"
        )
        return "\n".join(lines)

    @staticmethod
    def format_fraud_rate_table(results: List[DifficultyResult]) -> str:
        """Format fraud-rate axis results."""
        lines = ["=" * 60, "  Fraud Rate Sensitivity Study", "=" * 60]
        header = f"{'Target Fraud%':>14} {'Actual Fraud%':>14} {'AUC':>8} {'AP':>8}"
        lines.append(header)
        lines.append("-" * 60)
        for r in results:
            lines.append(
                f"{r.config.fraud_rate:>14.1%} "
                f"{r.fraud_ratio:>14.1%} "
                f"{r.auc_overall:>8.4f} "
                f"{r.ap_overall:>8.4f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)
