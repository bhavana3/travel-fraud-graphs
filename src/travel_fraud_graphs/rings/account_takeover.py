"""
Account Takeover (ATO) Ring Simulator.

Models credential-stuffing rings that compromise legitimate travel accounts
and drain loyalty points / make fraudulent bookings.

Two ATO sub-patterns:

1. Credential-stuffing cluster:
   Multiple compromised accounts accessed from the same device/IP pool
   in a short time window (burst login pattern).

2. Account mutation chain:
   A single compromised account undergoes rapid mutation:
   login → change contact info → transfer loyalty points → make booking
   All within hours; produces a temporal chain in the graph.

Graph topology:
  - Star: attacker device cluster ← accessed by → compromised accounts
  - Chain: user → loyalty_account → transferred_to → mule_loyalty_account
  - Dense bipartite: attacker IPs × compromised users
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class AccountTakeoverRing:
    """One instantiated ATO fraud ring."""

    ring_id: int
    rng: np.random.Generator
    ring_size_override: int = -1                 # -1 = random; >0 = target compromised accounts

    # participants
    compromised_user_ids: List[int] = field(default_factory=list)
    attacker_device_ids: List[int] = field(default_factory=list)
    attacker_ip_ids: List[int] = field(default_factory=list)
    mule_loyalty_ids: List[int] = field(default_factory=list)   # loyalty accounts receiving transfers
    fraudulent_booking_ids: List[int] = field(default_factory=list)
    loyalty_transfer_pairs: List[tuple] = field(default_factory=list)  # (src_lid, dst_lid)

    def compromised_account_count(self) -> int:
        if self.ring_size_override > 0:
            lo = max(3, self.ring_size_override - 2)
            hi = min(60, self.ring_size_override + 2)
            return int(self.rng.integers(lo, hi + 1))
        return int(self.rng.integers(5, 30))

    def attacker_device_count(self) -> int:
        return int(self.rng.integers(2, 8))

    def attacker_ip_count(self) -> int:
        return int(self.rng.integers(2, 10))

    def mule_loyalty_count(self) -> int:
        return int(self.rng.integers(2, 8))

    # ------------------------------------------------------------------
    # Feature injection
    # ------------------------------------------------------------------

    def compromised_user_features(self) -> dict:
        """Compromised accounts look like established legitimate users historically.

        Design principle (Fix 6 revised):
        ATO victims are REAL established users — their historical profile is
        indistinguishable from any other frequent traveller. The fraud signal is:
          1. GRAPH: attacker connects many accounts to the same device/IP pool
             (distinct_device_count / distinct_ip_count elevated)
          2. BOOKING: high chargeback_flag on post-takeover bookings (GNN-accessible)
          3. LOYALTY: suspicious point transfer patterns (loyalty node features)
        Tabular features are calibrated to match a plausible heavy-traveller profile.
        """
        r = self.rng
        # Fix 9: allow same age mixture as legit so velocity_score distribution matches
        # ATO victims are mostly established (90%) but some (10%) are mid-age accounts
        if r.random() < 0.10:
            acct_age = int(r.integers(30, 200))    # younger established
        else:
            acct_age = int(r.integers(200, 2000))  # established account (realistic)
        # Booking count: same Poisson as legit but slightly elevated (attacker makes
        # additional bookings post-takeover, but history looks like a normal traveller)
        n_bookings = max(1, int(r.poisson(3.0)))
        # Velocity naturally low because account is old: 3 bookings / 1100 days ≈ 1.0
        velocity = round(min(n_bookings / max(acct_age, 1) * 365, 50), 2)
        # Device and IP counts: match legit distribution.
        # Rationale: the ATO detection signal is GRAPH-STRUCTURAL — many compromised
        # accounts all connect to the SAME small pool of attacker devices/IPs
        # (attacker_device_ids / attacker_ip_ids edges in the builder).
        # From each individual account's perspective, it merely has 1-3 extra devices,
        # indistinguishable from a legit user adding a new work laptop.
        # Only a GNN traversing user→device→user paths can see all 20+ compromised
        # accounts converging on the same 2-5 attacker devices.
        n_devices = int(r.choice([1, 2, 3], p=[0.60, 0.30, 0.10]))   # legit-like
        # Fix 9: match legit IP mean 2.6 (was p=[0.35,0.30,0.20,0.10,0.04,0.01] → mean 2.21)
        n_ips     = int(r.choice([1, 2, 3, 4, 5, 6], p=[0.25, 0.25, 0.25, 0.15, 0.07, 0.03]))
        return {
            "account_age_days":      acct_age,
            "booking_count_30d":     n_bookings,
            "cancellation_rate":     round(r.uniform(0.0, 0.35), 3),
            "distinct_device_count": n_devices,
            "distinct_ip_count":     n_ips,
            # Fix 9: same 15-country distribution as legit (was 5-country set)
            "country_code": int(r.choice(
                [840, 156, 276, 826, 250, 392, 124, 36, 356, 76, 484, 380, 724, 528, 410],
                p=[0.20, 0.15, 0.10, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03, 0.03, 0.04]
            )),
            # Fix 9: 75% loyalty (above legit 60% since victims are established, but not 100%)
            "is_loyalty_member":     int(r.random() < 0.75),
            # Fix 9: log-normal same as legit (was uniform(300, 2000) → high mean, high separation)
            "avg_booking_value_usd": round(float(np.exp(r.normal(loc=6.1, scale=0.7))), 2),
            "referral_count":        0,
            "velocity_score":        velocity,
        }

    def attacker_device_features(self) -> dict:
        """Attacker-controlled device features."""
        r = self.rng
        n = len(self.compromised_user_ids)
        return {
            "device_type":       int(r.choice([0, 1, 2])),
            "os_type":           int(r.choice([0, 1, 2, 3])),
            "shared_user_count": n,
            "first_seen_days":   int(r.integers(1, 30)),
            "is_emulator":       int(r.random() < 0.50),
        }

    def attacker_ip_features(self) -> dict:
        """Attacker-controlled IP features."""
        r = self.rng
        n = len(self.compromised_user_ids)
        return {
            "is_vpn":            int(r.random() < 0.80),
            "is_datacenter":     int(r.random() < 0.65),
            "country_code":      int(r.choice([156, 356, 642, 804, 788])),
            "shared_user_count": n,
            "abuse_score":       int(r.integers(60, 100)),
        }

    def mule_loyalty_features(self) -> dict:
        """Loyalty accounts used to receive stolen points."""
        r = self.rng
        return {
            "point_balance":        int(r.integers(50_000, 500_000)),
            "lifetime_points":      int(r.integers(50_000, 600_000)),
            "redemption_count_30d": int(r.integers(5, 20)),    # high redemptions
            "transfer_count_30d":   int(r.integers(3, 15)),    # high transfers in
            "account_age_days":     int(r.integers(1, 30)),    # new mule account
            "status_tier":          0,
            "suspicious_velocity":  1,
        }

    def compromised_loyalty_features(self) -> dict:
        """Loyalty account belonging to a compromised user (victim)."""
        r = self.rng
        return {
            "point_balance":        int(r.integers(0, 5000)),      # drained
            "lifetime_points":      int(r.integers(50_000, 400_000)),
            "redemption_count_30d": int(r.integers(3, 15)),
            "transfer_count_30d":   int(r.integers(2, 10)),        # transfers out
            "account_age_days":     int(r.integers(200, 2000)),
            "status_tier":          int(r.integers(1, 4)),
            "suspicious_velocity":  1,
        }

    def fraudulent_booking_features(self) -> dict:
        """Bookings made post-takeover."""
        r = self.rng
        return {
            "booking_value_usd":  round(r.uniform(500, 3000), 2),  # high-value
            "lead_time_days":     int(r.integers(0, 3)),             # same-day / next-day
            "duration_nights":    int(r.integers(1, 7)),
            "is_cancelled":       int(r.random() < 0.60),
            "is_refunded":        0,
            "chargeback_flag":    int(r.random() < 0.70),
            "booking_channel":    int(r.choice([0, 2])),             # web or API
            "passengers":         int(r.integers(1, 4)),
            "timestamp_unix":     int(r.integers(1_600_000_000, 1_750_000_000)),
        }
