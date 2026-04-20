"""
Legitimate traveler agent.

Each traveler generates realistic booking behaviour drawn from travel-industry
empirical distributions (lead times, cancellation rates, device diversity,
loyalty programme participation, etc.).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TravelerAgent:
    """Represents one legitimate user account."""

    user_id: int
    rng: np.random.Generator

    # --- profile (set in __post_init__) ---
    account_age_days: int = field(init=False)
    country_code: int = field(init=False)
    is_loyalty_member: bool = field(init=False)
    status_tier: int = field(init=False)

    # --- behavioural stats accumulate during simulation ---
    booking_ids: List[int] = field(default_factory=list)
    device_ids: List[int] = field(default_factory=list)
    ip_ids: List[int] = field(default_factory=list)
    payment_ids: List[int] = field(default_factory=list)
    review_ids: List[int] = field(default_factory=list)
    loyalty_account_id: Optional[int] = None
    chargeback_count: int = 0
    cancellation_count: int = 0
    referral_count: int = 0

    is_fraud: bool = False
    ring_id: int = -1

    def __post_init__(self):
        r = self.rng
        # Account age: mixture of new (< 30 d) and established accounts
        if r.random() < 0.15:
            self.account_age_days = int(r.integers(1, 30))
        else:
            self.account_age_days = int(r.integers(30, 2000))

        # Country distribution (top travel-originating markets, numeric codes)
        country_weights = [0.20, 0.15, 0.10, 0.08, 0.07, 0.06, 0.05, 0.05,
                           0.04, 0.04, 0.03, 0.03, 0.03, 0.03, 0.04]
        country_codes   = [840, 156, 276, 826, 250, 392, 124, 36,
                           356, 76, 484, 380, 724, 528, 410]
        self.country_code = int(r.choice(country_codes, p=country_weights))

        # Loyalty membership (~60 % of travellers)
        self.is_loyalty_member = r.random() < 0.60
        if self.is_loyalty_member:
            self.status_tier = int(r.choice([0, 1, 2, 3], p=[0.55, 0.25, 0.15, 0.05]))
        else:
            self.status_tier = 0

    # ------------------------------------------------------------------
    # Behavioural sampling helpers
    # ------------------------------------------------------------------

    def sample_booking_count(self) -> int:
        """Number of bookings this agent makes in the simulation window."""
        # Heavy-tailed: most users book 1-3 times; frequent travellers up to 20
        base = self.rng.poisson(lam=2.2)
        if self.status_tier >= 2:          # gold / platinum travellers book more
            base += self.rng.integers(2, 8)
        return max(1, int(base))

    def sample_lead_time_days(self) -> int:
        """Days between booking and departure."""
        # Gamma distribution fitted to OTA data (shape≈2, scale≈30 days)
        return max(1, int(self.rng.gamma(shape=2.0, scale=30.0)))

    def sample_duration_nights(self) -> int:
        """Hotel stay duration."""
        return max(1, int(self.rng.gamma(shape=1.8, scale=3.0)))

    def sample_booking_value(self) -> float:
        """Booking value in USD (log-normal centred around $450)."""
        return round(float(np.exp(self.rng.normal(loc=6.1, scale=0.7))), 2)

    def sample_cancels(self) -> bool:
        """Whether a booking gets cancelled (~18 % base rate)."""
        return self.rng.random() < 0.18

    def sample_review(self) -> bool:
        """Whether the user leaves a review after a completed stay (~35 %)."""
        return self.rng.random() < 0.35

    def sample_review_rating(self) -> int:
        """Review rating distribution: bimodal (mostly 4-5, some 1-2)."""
        return int(self.rng.choice([1, 2, 3, 4, 5], p=[0.04, 0.06, 0.10, 0.30, 0.50]))

    def sample_device_count(self) -> int:
        """Distinct devices used over the simulation window."""
        return int(self.rng.choice([1, 2, 3, 4], p=[0.60, 0.25, 0.10, 0.05]))

    def feature_vector(self) -> dict:
        """Return the node feature dictionary for this user.

        NOTE: chargeback_count is intentionally excluded from user-level features.
        It is encoded at the booking level (booking.chargeback_flag) so that only
        graph-aware models (GNNs) can access it via message passing. This forces
        tabular baselines to rely on weaker demographic/behavioural signals and
        creates a genuine graph-versus-tabular performance gap.
        """
        n_bookings = len(self.booking_ids)
        n_cancelled = self.cancellation_count
        return {
            "account_age_days":      self.account_age_days,
            "booking_count_30d":     min(n_bookings, 30),
            "cancellation_rate":     round(n_cancelled / max(n_bookings, 1), 3),
            "distinct_device_count": len(set(self.device_ids)),
            "distinct_ip_count":     len(set(self.ip_ids)),
            "country_code":          self.country_code,
            "is_loyalty_member":     int(self.is_loyalty_member),
            "avg_booking_value_usd": self.sample_booking_value(),
            "referral_count":        self.referral_count,
            "velocity_score":        round(min(n_bookings / max(self.account_age_days, 1) * 365, 50), 2),
        }
