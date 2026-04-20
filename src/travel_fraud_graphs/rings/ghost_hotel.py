"""
Ghost Hotel Scheme Simulator.

Models fake accommodation listing networks with coordinated fake review
injection.  A ring operator controls:
  - 1-3 ghost hotel listings (fraudulent properties)
  - A cluster of reviewer accounts (fake reviews)
  - A cluster of booker accounts (fake bookings that generate revenue
    before cancellation)

Graph topology:
  ghost hotels ← reviewed by → reviewer cluster
  ghost hotels ← booked by  → booker cluster
  All reviewer/booker accounts share device/IP pool.
  Bipartite subgraph: reviewers × hotels forms a dense clique.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class GhostHotelRing:
    """One instantiated ghost hotel fraud ring."""

    ring_id: int
    rng: np.random.Generator
    ring_size_override: int = -1                 # -1 = random; >0 = target total fraud users

    # participants
    ghost_hotel_ids: List[int] = field(default_factory=list)
    reviewer_uids: List[int] = field(default_factory=list)
    booker_uids: List[int] = field(default_factory=list)
    shared_device_ids: List[int] = field(default_factory=list)
    shared_ip_ids: List[int] = field(default_factory=list)

    # outputs
    review_ids: List[int] = field(default_factory=list)
    booking_ids: List[int] = field(default_factory=list)

    def ghost_hotel_count(self) -> int:
        return int(self.rng.integers(1, 4))

    def reviewer_count(self) -> int:
        if self.ring_size_override > 0:
            # ~60% of ring users are reviewers; clamp to realistic range
            target = max(3, int(round(self.ring_size_override * 0.6)))
            lo = max(3, target - 2)
            hi = min(60, target + 2)
            return int(self.rng.integers(lo, hi + 1))
        return int(self.rng.integers(4, 25))

    def booker_count(self) -> int:
        if self.ring_size_override > 0:
            # ~40% of ring users are bookers
            target = max(2, int(round(self.ring_size_override * 0.4)))
            lo = max(2, target - 1)
            hi = min(30, target + 2)
            return int(self.rng.integers(lo, hi + 1))
        return int(self.rng.integers(3, 12))

    def shared_device_count(self) -> int:
        return int(self.rng.integers(1, 4))

    def shared_ip_count(self) -> int:
        return int(self.rng.integers(1, 3))

    # ------------------------------------------------------------------
    # Feature injection
    # ------------------------------------------------------------------

    def ghost_hotel_features(self) -> dict:
        """Features for a ghost hotel node."""
        r = self.rng
        return {
            "hotel_class":         int(r.integers(3, 6)),     # always 3-5 stars
            "review_count":        int(r.integers(10, 80)),   # suspiciously high for new listing
            "avg_rating":          round(r.uniform(4.6, 5.0), 1),  # unnaturally high
            "listing_age_days":    int(r.integers(1, 60)),    # very new
            "cancellation_policy": 0,                          # always flexible (attract bookings)
            "country_code":        int(r.choice([840, 826, 724, 380, 300, 792])),
            "is_ghost":            1,
            "photo_count":         int(r.integers(3, 10)),    # low photo count
            "price_percentile":    int(r.integers(10, 40)),   # suspiciously cheap
        }

    def legitimate_hotel_features(self) -> dict:
        """Features for a normal hotel node."""
        r = self.rng
        return {
            "hotel_class":         int(r.integers(1, 6)),
            "review_count":        int(r.integers(20, 2000)),
            "avg_rating":          round(r.uniform(2.8, 4.9), 1),
            "listing_age_days":    int(r.integers(100, 5000)),
            "cancellation_policy": int(r.integers(0, 3)),
            "country_code":        int(r.choice([840, 156, 276, 826, 250, 392, 36])),
            "is_ghost":            0,
            "photo_count":         int(r.integers(10, 200)),
            "price_percentile":    int(r.integers(10, 91)),
        }

    def fake_review_features(self) -> dict:
        """Features for a fake review node."""
        r = self.rng
        return {
            "rating":            5,                              # always 5-star
            "verified_booking":  int(r.random() < 0.30),        # often not verified
            "review_length_chars": int(r.integers(40, 200)),    # short generic reviews
            "sentiment_score":   round(r.uniform(0.7, 1.0), 3),
            "days_after_checkin": int(r.integers(0, 3)),         # posted immediately
            "is_incentivised":   1,
        }

    def fake_booker_features(self) -> dict:
        """Features for a fake booker user node.

        Design principle (Fix 6 revised): Booker accounts look like plausible
        occasional travellers; their fraud signal is graph-structural (shared
        device/IP with other ring accounts) and booking-content (high chargeback_flag
        on fake_booking_features, accessible to GNNs via user→made→booking edges).
        """
        r = self.rng
        # Same age mixture as legitimate users
        if r.random() < 0.15:
            acct_age = int(r.integers(1, 30))
        else:
            acct_age = int(r.integers(30, 2000))
        # Fix 9: match legit Poisson (was 2.0, legit mean is 2.8)
        n_bookings = max(1, int(r.poisson(2.5)))
        velocity = round(min(n_bookings / max(acct_age, 1) * 365, 50), 2)
        return {
            "account_age_days":      acct_age,
            "booking_count_30d":     n_bookings,
            "cancellation_rate":     round(r.uniform(0.0, 0.35), 3),
            "distinct_device_count": int(r.integers(1, 3)),
            # Fix 9: match legit IP mean 2.6 (was p=[0.35,0.30,0.20,0.10,0.04,0.01] → mean 2.21)
            "distinct_ip_count":     int(r.choice([1,2,3,4,5,6], p=[0.25,0.25,0.25,0.15,0.07,0.03])),
            # Fix 9: same 15-country distribution as legit (was [840, 156, 276] — 3 countries)
            "country_code": int(r.choice(
                [840, 156, 276, 826, 250, 392, 124, 36, 356, 76, 484, 380, 724, 528, 410],
                p=[0.20, 0.15, 0.10, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03, 0.03, 0.04]
            )),
            "is_loyalty_member":     int(r.random() < 0.60),   # Fix 9: match legit 60% (was 35%)
            # Fix 9: log-normal same as legit (was uniform(50, 500))
            "avg_booking_value_usd": round(float(np.exp(r.normal(loc=6.1, scale=0.7))), 2),
            "referral_count":        0,
            "velocity_score":        velocity,
        }

    def fake_reviewer_features(self) -> dict:
        """Features for a fake reviewer user node.

        Reviewers are designed to look like legitimate low-activity users.
        Their only fraud signal is graph topology: shared device/IP with bookers
        and review edges pointing to ghost hotel nodes with suspicious properties.
        """
        r = self.rng
        if r.random() < 0.15:
            acct_age = int(r.integers(1, 30))
        else:
            acct_age = int(r.integers(30, 2000))
        # Fix 9: match legit booking count (was Poisson(1.0) → mean 1.0, caused low velocity)
        n_bookings = max(1, int(r.poisson(2.5)))
        velocity = round(min(n_bookings / max(acct_age, 1) * 365, 50), 2)
        return {
            "account_age_days":      acct_age,
            "booking_count_30d":     n_bookings,
            # Fix 9: was hardcoded 0.0 — single biggest separator, pulled fraud mean to 0.128 vs benign 0.186
            "cancellation_rate":     round(r.uniform(0.0, 0.35), 3),
            "distinct_device_count": int(r.choice([1, 2], p=[0.70, 0.30])),
            # Fix 9: match legit IP mean 2.6 (was [1,2,3,4] p=[0.45,0.30,0.15,0.10] → mean 1.90)
            "distinct_ip_count":     int(r.choice([1,2,3,4,5,6], p=[0.25,0.25,0.25,0.15,0.07,0.03])),
            # Fix 9: same 15-country distribution as legit (was [840, 156, 356] — 3 countries)
            "country_code": int(r.choice(
                [840, 156, 276, 826, 250, 392, 124, 36, 356, 76, 484, 380, 724, 528, 410],
                p=[0.20, 0.15, 0.10, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03, 0.03, 0.04]
            )),
            "is_loyalty_member":     int(r.random() < 0.60),   # Fix 9: match legit 60% (was 25%)
            # Fix 9: use log-normal same as legit (was uniform(0, 200) → very low, high separation)
            "avg_booking_value_usd": round(float(np.exp(r.normal(loc=6.1, scale=0.7))), 2),
            "referral_count":        0,
            "velocity_score":        velocity,
        }

    def fake_booking_features(self) -> dict:
        """Features for a fraudulent booking on a ghost hotel."""
        r = self.rng
        return {
            "booking_value_usd":  round(r.uniform(30, 250), 2),
            "lead_time_days":     int(r.integers(1, 5)),
            "duration_nights":    int(r.integers(1, 7)),
            "is_cancelled":       1,
            "is_refunded":        int(r.random() < 0.70),
            "chargeback_flag":    int(r.random() < 0.20),
            "booking_channel":    int(r.choice([0, 1])),
            "passengers":         int(r.integers(1, 3)),
            "timestamp_unix":     int(r.integers(1_600_000_000, 1_750_000_000)),
        }

    def device_features(self) -> dict:
        r = self.rng
        n = len(self.reviewer_uids) + len(self.booker_uids)
        return {
            "device_type":       int(r.choice([0, 1])),
            "os_type":           int(r.choice([0, 1, 2, 3])),
            "shared_user_count": n,
            "first_seen_days":   int(r.integers(1, 30)),
            "is_emulator":       int(r.random() < 0.45),
        }

    def ip_features(self) -> dict:
        r = self.rng
        n = len(self.reviewer_uids) + len(self.booker_uids)
        return {
            "is_vpn":            int(r.random() < 0.60),
            "is_datacenter":     int(r.random() < 0.40),
            "country_code":      int(r.choice([840, 156])),
            "shared_user_count": n,
            "abuse_score":       int(r.integers(40, 95)),
        }
