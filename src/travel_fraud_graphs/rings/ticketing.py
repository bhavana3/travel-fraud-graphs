"""
Ticketing Fraud Ring Simulator.

Models coordinated bulk-purchase / resale rings in airline ticketing.
A central orchestrator account controls multiple satellite accounts; all
satellites share a small pool of devices and IP addresses and make
temporally clustered bookings for the same high-demand flight(s).

Ring topology: star (orchestrator + satellites) with shared device/IP
sub-clusters.  Chargebacks are filed after resale, producing a distinctive
chargeback-burst pattern on the same flight node.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict


@dataclass
class TicketingFraudRing:
    """One instantiated ticketing fraud ring."""

    ring_id: int
    rng: np.random.Generator
    ring_size_override: int = -1                 # -1 = random; >0 = target for difficulty study

    # ring configuration
    ring_size: int = field(init=False)           # total accounts (incl. orchestrator)
    orchestrator_uid: int = -1                   # set by builder
    satellite_uids: List[int] = field(default_factory=list)
    shared_device_ids: List[int] = field(default_factory=list)
    shared_ip_ids: List[int] = field(default_factory=list)
    target_flight_ids: List[int] = field(default_factory=list)

    # bookings produced
    booking_ids: List[int] = field(default_factory=list)
    chargeback_booking_ids: List[int] = field(default_factory=list)

    def __post_init__(self):
        if self.ring_size_override > 0:
            # Difficulty study: narrow window ±2 around target, clamped to valid range
            lo = max(3, self.ring_size_override - 2)
            hi = min(40, self.ring_size_override + 2)
            self.ring_size = int(self.rng.integers(lo, hi + 1))
        else:
            # Ring size: 3–20 accounts; most rings are small
            self.ring_size = int(self.rng.integers(3, 21))

    # ------------------------------------------------------------------
    # Feature injection helpers (called by GraphBuilder)
    # ------------------------------------------------------------------

    def fraudster_user_features(self, uid: int) -> dict:
        """Features for a user node that is part of this ring.

        Design principle (Fix 6 revised):
        Fraud ring members should look INDIVIDUALLY indistinguishable from legitimate
        users in tabular space. Their fraud signal comes from GRAPH STRUCTURE:
          - sharing devices/IPs with many other accounts
          - high chargeback_flag on connected booking nodes (GNN-accessible only)
        Features are generated using the SAME actuarial logic as TravelerAgent so
        that each individual fraudster's feature vector is plausible for a real user.

        What tabular models CAN'T see: that these users all share the same 1-4 devices.
        What GNNs CAN see: the dense user-device-user subgraph forming the ring.
        """
        r = self.rng
        # Account age: same mixture as legitimate users (15% new, 85% established)
        if r.random() < 0.15:
            acct_age = int(r.integers(1, 30))
        else:
            acct_age = int(r.integers(30, 2000))
        # Booking count: Poisson(2.5), same distribution as legitimate travellers
        n_bookings = max(1, int(r.poisson(2.5)))
        # Velocity: derived from booking count and account age — same formula as legit
        velocity = round(min(n_bookings / max(acct_age, 1) * 365, 50), 2)
        # Cancellation rate: slightly elevated vs legit (legit mean ≈ 0.18) but overlapping
        cancel_rate = round(r.uniform(0.0, 0.35), 3)
        return {
            "account_age_days":      acct_age,
            "booking_count_30d":     n_bookings,
            "cancellation_rate":     cancel_rate,
            "distinct_device_count": len(self.shared_device_ids),   # ring-shared (1-4)
            # IP count: match legit mean 2.6 (Fix 9: was p=[0.35,0.30,0.20,0.10,0.04,0.01] → mean 2.21)
            "distinct_ip_count": int(r.choice([1,2,3,4,5,6], p=[0.25,0.25,0.25,0.15,0.07,0.03])),
            # Country: same 15-country distribution as TravelerAgent (Fix 9: was 5-country limited set)
            "country_code": int(r.choice(
                [840, 156, 276, 826, 250, 392, 124, 36, 356, 76, 484, 380, 724, 528, 410],
                p=[0.20, 0.15, 0.10, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03, 0.03, 0.04]
            )),
            "is_loyalty_member":     int(r.random() < 0.60),        # Fix 9: match legit 60% (was 40%)
            # Booking value: same log-normal as legit (~$450 median, Fix 9: was uniform(200,1500))
            "avg_booking_value_usd": round(float(np.exp(r.normal(loc=6.1, scale=0.7))), 2),
            "referral_count":        0,
            "velocity_score":        velocity,
        }

    def booking_features(self, is_chargeback: bool = False) -> dict:
        """Features for a booking node created by this ring."""
        r = self.rng
        return {
            "booking_value_usd":  round(r.uniform(280, 1100), 2),
            "lead_time_days":     int(r.integers(1, 8)),    # last-minute
            "duration_nights":    0,                         # flight-only
            "is_cancelled":       int(is_chargeback),
            "is_refunded":        int(is_chargeback),
            "chargeback_flag":    int(is_chargeback),
            "booking_channel":    int(r.choice([0, 2], p=[0.4, 0.6])),  # web/api
            "passengers":         int(r.integers(1, 5)),
            "timestamp_unix":     int(r.integers(1_600_000_000, 1_750_000_000)),
        }

    def device_features(self) -> dict:
        """Features for shared device nodes in this ring."""
        r = self.rng
        return {
            "device_type":       int(r.choice([0, 1])),
            "os_type":           int(r.choice([0, 1, 2, 3])),
            "shared_user_count": self.ring_size,
            "first_seen_days":   int(r.integers(1, 60)),
            "is_emulator":       int(r.random() < 0.35),
        }

    def ip_features(self) -> dict:
        """Features for shared IP nodes in this ring."""
        r = self.rng
        return {
            "is_vpn":            int(r.random() < 0.70),
            "is_datacenter":     int(r.random() < 0.50),
            "country_code":      int(r.choice([840, 156, 356])),
            "shared_user_count": self.ring_size,
            "abuse_score":       int(r.integers(50, 100)),
        }

    def shared_device_count(self) -> int:
        """How many devices the ring shares (typically 1-4)."""
        return int(self.rng.integers(1, min(4, self.ring_size) + 1))

    def shared_ip_count(self) -> int:
        """How many IPs the ring shares (typically 1-3)."""
        return int(self.rng.integers(1, min(3, self.ring_size) + 1))

    def bookings_per_satellite(self) -> int:
        """Number of ticket-buying bookings per satellite account."""
        return int(self.rng.integers(2, 10))

    def chargeback_rate(self) -> float:
        """Fraction of bookings that generate a chargeback."""
        return float(self.rng.uniform(0.55, 0.95))
