"""
TravelFraudGraphBuilder

Orchestrates the full agent-based simulation and assembles the
heterogeneous property graph from legitimate travellers and fraud rings.

Output is a self-contained GraphData object holding:
  - Per-node-type feature matrices and label vectors
  - Edge index lists for every relation type
  - Ring membership metadata

Usage
-----
from travel_fraud_graphs.graph.builder import TravelFraudGraphBuilder

builder = TravelFraudGraphBuilder(
    n_users=5000,
    n_hotels=500,
    n_flights=800,
    n_ticketing_rings=20,
    n_ghost_hotel_rings=15,
    n_ato_rings=15,
    seed=42,
)
data = builder.build()
"""

from __future__ import annotations
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

from ..schema import (
    NODE_USER, NODE_DEVICE, NODE_IP, NODE_BOOKING, NODE_FLIGHT,
    NODE_HOTEL, NODE_REVIEW, NODE_PAYMENT, NODE_LOYALTY,
    LABEL_LEGITIMATE, LABEL_FRAUD,
    FRAUD_RING_TYPES,
)
from ..agents.traveler import TravelerAgent
from ..rings.ticketing import TicketingFraudRing
from ..rings.ghost_hotel import GhostHotelRing
from ..rings.account_takeover import AccountTakeoverRing


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class GraphData:
    """
    Container for the generated heterogeneous travel fraud graph.

    node_features[node_type]  : list[dict]   – one dict per node (feature name → value)
    node_labels[node_type]    : list[int]    – 0=legitimate, 1=fraud
    node_ring_ids[node_type]  : list[int]    – -1 if legitimate, else ring id
    node_ring_types[node_type]: list[int]    – see schema.FRAUD_RING_TYPES
    edges[relation]           : list[(src_idx, dst_idx)]  – local indices per type
    metadata                  : dict         – counts, seed, config
    """
    node_features:   Dict[str, List[dict]]         = field(default_factory=dict)
    node_labels:     Dict[str, List[int]]           = field(default_factory=dict)
    node_ring_ids:   Dict[str, List[int]]           = field(default_factory=dict)
    node_ring_types: Dict[str, List[int]]           = field(default_factory=dict)
    edges:           Dict[Tuple, List[Tuple[int,int]]] = field(default_factory=dict)
    metadata:        Dict[str, Any]                 = field(default_factory=dict)

    def num_nodes(self, ntype: str) -> int:
        return len(self.node_labels.get(ntype, []))

    def num_edges(self, rel: Tuple) -> int:
        return len(self.edges.get(rel, []))

    def fraud_ratio(self, ntype: str) -> float:
        labels = self.node_labels.get(ntype, [])
        if not labels:
            return 0.0
        return sum(labels) / len(labels)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class TravelFraudGraphBuilder:
    """
    Agent-based simulator that generates a Travel Fraud Graph.

    Parameters
    ----------
    n_users : int
        Total number of user accounts (legitimate + fraudulent).
    n_hotels : int
        Number of hotel listing nodes.
    n_flights : int
        Number of flight nodes.
    n_ticketing_rings : int
        Number of ticketing fraud rings to inject.
    n_ghost_hotel_rings : int
        Number of ghost hotel rings to inject.
    n_ato_rings : int
        Number of account takeover rings to inject.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_users:             int = 5000,
        n_hotels:            int = 500,
        n_flights:           int = 800,
        n_ticketing_rings:   int = 20,
        n_ghost_hotel_rings: int = 15,
        n_ato_rings:         int = 15,
        seed:                int = 42,
        ring_size_target:    int = -1,   # -1 = random per ring; >0 = target for difficulty study
    ):
        self.n_users             = n_users
        self.n_hotels            = n_hotels
        self.n_flights           = n_flights
        self.n_ticketing_rings   = n_ticketing_rings
        self.n_ghost_hotel_rings = n_ghost_hotel_rings
        self.n_ato_rings         = n_ato_rings
        self.seed                = seed
        self.ring_size_target    = ring_size_target

        self.rng = np.random.default_rng(seed)

        # Node pools (global indices)
        self._users:    List[TravelerAgent]  = []
        self._devices:  List[dict]           = []
        self._ips:      List[dict]           = []
        self._bookings: List[dict]           = []
        self._flights:  List[dict]           = []
        self._hotels:   List[dict]           = []
        self._reviews:  List[dict]           = []
        self._payments: List[dict]           = []
        self._loyalties: List[dict]          = []

        # Labels & ring membership
        self._labels:     Dict[str, List[int]] = defaultdict(list)
        self._ring_ids:   Dict[str, List[int]] = defaultdict(list)
        self._ring_types: Dict[str, List[int]] = defaultdict(list)

        # Edge lists
        self._edges: Dict[Tuple, List[Tuple[int, int]]] = defaultdict(list)

        # Ring objects
        self._ticketing_rings:   List[TicketingFraudRing]   = []
        self._ghost_hotel_rings: List[GhostHotelRing]       = []
        self._ato_rings:         List[AccountTakeoverRing]  = []

        # Running ring counter
        self._ring_counter = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_ring_id(self) -> int:
        rid = self._ring_counter
        self._ring_counter += 1
        return rid

    def _add_node(
        self,
        ntype: str,
        features: dict,
        is_fraud: bool,
        ring_id: int = -1,
        ring_type: int = 0,
    ) -> int:
        """Append a node to the appropriate list and return its local index."""
        pool = {
            NODE_USER:    self._users,
            NODE_DEVICE:  self._devices,
            NODE_IP:      self._ips,
            NODE_BOOKING: self._bookings,
            NODE_FLIGHT:  self._flights,
            NODE_HOTEL:   self._hotels,
            NODE_REVIEW:  self._reviews,
            NODE_PAYMENT: self._payments,
            NODE_LOYALTY: self._loyalties,
        }[ntype]

        idx = len(pool)
        pool.append(features)
        self._labels[ntype].append(LABEL_FRAUD if is_fraud else LABEL_LEGITIMATE)
        self._ring_ids[ntype].append(ring_id)
        self._ring_types[ntype].append(ring_type)
        return idx

    def _add_edge(self, rel: Tuple, src: int, dst: int):
        self._edges[rel].append((src, dst))

    # ------------------------------------------------------------------
    # Legitimate node generators
    # ------------------------------------------------------------------

    def _gen_legitimate_flights(self):
        airports = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                    111, 112, 113, 114, 115, 116, 117, 118, 119, 120]
        airlines = [1, 2, 3, 4, 5, 6, 7, 8]
        for _ in range(self.n_flights):
            orig, dest = self.rng.choice(airports, size=2, replace=False)
            feats = {
                "origin_airport":  int(orig),
                "dest_airport":    int(dest),
                "airline_code":    int(self.rng.choice(airlines)),
                "departure_unix":  int(self.rng.integers(1_650_000_000, 1_750_000_000)),
                "seat_class":      int(self.rng.choice([0, 1, 2, 3], p=[0.65, 0.15, 0.15, 0.05])),
                "base_price_usd":  round(float(np.exp(self.rng.normal(5.8, 0.5))), 2),
                "remaining_seats": int(self.rng.integers(0, 180)),
            }
            self._add_node(NODE_FLIGHT, feats, is_fraud=False)

    def _gen_legitimate_hotels(self, n: int):
        for _ in range(n):
            r = self.rng
            feats = {
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
            self._add_node(NODE_HOTEL, feats, is_fraud=False)

    def _gen_device(self, shared_users: int = 1, is_fraud: bool = False,
                    ring_id: int = -1, ring_type: int = 0) -> int:
        r = self.rng
        feats = {
            "device_type":       int(r.choice([0, 1, 2], p=[0.40, 0.50, 0.10])),
            "os_type":           int(r.choice([0, 1, 2, 3, 4], p=[0.30, 0.20, 0.20, 0.25, 0.05])),
            "shared_user_count": shared_users,
            "first_seen_days":   int(r.integers(1, 500)),
            "is_emulator":       0,
        }
        return self._add_node(NODE_DEVICE, feats, is_fraud, ring_id, ring_type)

    def _gen_ip(self, shared_users: int = 1, is_fraud: bool = False,
                ring_id: int = -1, ring_type: int = 0) -> int:
        r = self.rng
        feats = {
            "is_vpn":            int(r.random() < 0.10),
            "is_datacenter":     int(r.random() < 0.05),
            "country_code":      int(r.choice([840, 156, 276, 826, 250])),
            "shared_user_count": shared_users,
            "abuse_score":       int(r.integers(0, 20)),
        }
        return self._add_node(NODE_IP, feats, is_fraud, ring_id, ring_type)

    def _gen_payment(self, is_fraud: bool = False, ring_id: int = -1,
                     ring_type: int = 0, shared: int = 1) -> int:
        r = self.rng
        feats = {
            "card_type":            int(r.choice([0, 1, 2, 3], p=[0.50, 0.30, 0.10, 0.10])),
            "issuer_country_code":  int(r.choice([840, 156, 276, 826, 250])),
            "shared_user_count":    shared,
            "total_bookings":       int(r.integers(1, 30)),
            "chargeback_count":     int(r.integers(0, 3)),
            "is_compromised":       int(is_fraud and r.random() < 0.6),
        }
        return self._add_node(NODE_PAYMENT, feats, is_fraud, ring_id, ring_type)

    def _gen_loyalty(self, user_feats: dict, is_fraud: bool = False,
                     ring_id: int = -1, ring_type: int = 0,
                     override: Optional[dict] = None) -> int:
        r = self.rng
        tier = user_feats.get("status_tier", 0) if not is_fraud else 0
        feats = override if override else {
            "point_balance":        int(r.integers(0, 200_000)),
            "lifetime_points":      int(r.integers(5_000, 500_000)),
            "redemption_count_30d": int(r.integers(0, 5)),
            "transfer_count_30d":   int(r.integers(0, 2)),
            "account_age_days":     user_feats.get("account_age_days", 100),
            "status_tier":          tier,
            "suspicious_velocity":  0,
        }
        return self._add_node(NODE_LOYALTY, feats, is_fraud, ring_id, ring_type)

    # ------------------------------------------------------------------
    # Legitimate traveller simulation
    # ------------------------------------------------------------------

    def _simulate_legitimate_users(self, n: int):
        """Create n legitimate user agents and generate their bookings.

        Realistic device-sharing: ~15% of legitimate users share a device
        with 1-3 other legitimate users (family members, office colleagues).
        Calibrated to Forter (2024): ~10-20% of travel accounts share at
        least one device with another legitimate account.

        Legitimate chargebacks: ~2% of non-cancelled bookings result in a
        legitimate chargeback (disputed charges, merchant errors).
        Calibrated to Mastercard (2024): ~1.5-2.5% chargeback rate on
        legitimate travel transactions.
        """
        # Pre-build shared device pool for ~15% of legitimate users.
        # Groups of 2-4 users share one device (family / office scenario).
        p_share = 0.15
        n_sharing = int(n * p_share)
        legit_shared_dev_map: dict = {}   # uid → list of shared device_ids

        remaining = list(self.rng.permutation(n_sharing))
        while len(remaining) >= 2:
            g_size = min(int(self.rng.integers(2, 5)), len(remaining))
            group = remaining[:g_size]
            remaining = remaining[g_size:]
            shared_dev_id = self._gen_device(shared_users=g_size)
            for uid in group:
                legit_shared_dev_map.setdefault(uid, []).append(shared_dev_id)

        for uid in range(n):
            agent = TravelerAgent(user_id=uid, rng=self.rng)

            # Devices: own unique devices + optional shared device
            dev_count = agent.sample_device_count()
            dev_ids = [self._gen_device(shared_users=1) for _ in range(dev_count)]
            if uid in legit_shared_dev_map:
                dev_ids.extend(legit_shared_dev_map[uid])
            agent.device_ids = dev_ids

            # IPs (slightly more than devices)
            ip_count = dev_count + int(self.rng.integers(0, 3))
            ip_ids = [self._gen_ip(shared_users=1) for _ in range(ip_count)]
            agent.ip_ids = ip_ids

            # Payment cards
            n_cards = int(self.rng.choice([1, 2, 3], p=[0.70, 0.20, 0.10]))
            card_ids = [self._gen_payment() for _ in range(n_cards)]
            agent.payment_ids = card_ids

            # Loyalty account
            if agent.is_loyalty_member:
                lid = self._gen_loyalty(
                    user_feats={"account_age_days": agent.account_age_days,
                                "status_tier": agent.status_tier}
                )
                agent.loyalty_account_id = lid

            # Bookings
            n_bookings = agent.sample_booking_count()
            booking_ids = []
            for _ in range(n_bookings):
                cancelled = agent.sample_cancels()
                if cancelled:
                    agent.cancellation_count += 1
                # ~2% of non-cancelled bookings: legitimate chargeback
                # (disputed charge, merchant error, lost card).
                is_legit_cb = (not cancelled and self.rng.random() < 0.02)
                if is_legit_cb:
                    agent.chargeback_count += 1
                b_feats = {
                    "booking_value_usd":  agent.sample_booking_value(),
                    "lead_time_days":     agent.sample_lead_time_days(),
                    "duration_nights":    agent.sample_duration_nights(),
                    "is_cancelled":       int(cancelled),
                    "is_refunded":        int(cancelled and self.rng.random() < 0.7),
                    "chargeback_flag":    int(is_legit_cb),
                    "booking_channel":    int(self.rng.choice([0, 1, 2, 3], p=[0.50, 0.35, 0.10, 0.05])),
                    "passengers":         int(self.rng.integers(1, 5)),
                    "timestamp_unix":     int(self.rng.integers(1_600_000_000, 1_750_000_000)),
                }
                bid = self._add_node(NODE_BOOKING, b_feats, is_fraud=False)
                booking_ids.append(bid)

                # Booking → flight or hotel
                if self.rng.random() < 0.55:
                    fid = int(self.rng.integers(0, max(1, len(self._flights))))
                    self._add_edge((NODE_BOOKING, "for_flight", NODE_FLIGHT), bid, fid)
                else:
                    hid = int(self.rng.integers(0, max(1, len(self._hotels))))
                    self._add_edge((NODE_BOOKING, "for_hotel", NODE_HOTEL), bid, hid)

                # Payment
                cid = self.rng.choice(card_ids)
                self._add_edge((NODE_BOOKING, "paid_with", NODE_PAYMENT), bid, cid)

                # Review (hotel stays only)
                if not cancelled and agent.sample_review():
                    r_feats = {
                        "rating":             agent.sample_review_rating(),
                        "verified_booking":   1,
                        "review_length_chars": int(self.rng.integers(50, 800)),
                        "sentiment_score":    round(self.rng.uniform(-0.2, 1.0), 3),
                        "days_after_checkin": int(self.rng.integers(1, 30)),
                        "is_incentivised":    0,
                    }
                    rid_node = self._add_node(NODE_REVIEW, r_feats, is_fraud=False)
                    agent.review_ids.append(rid_node)

            agent.booking_ids = booking_ids

            # Add user node (features computed after bookings)
            user_idx = self._add_node(NODE_USER, agent.feature_vector(), is_fraud=False)

            # Edges from user
            for did in dev_ids:
                self._add_edge((NODE_USER, "uses_device", NODE_DEVICE), user_idx, did)
            for iid in ip_ids:
                self._add_edge((NODE_USER, "uses_ip", NODE_IP), user_idx, iid)
            for bid in booking_ids:
                self._add_edge((NODE_USER, "made", NODE_BOOKING), user_idx, bid)
            for cid in card_ids:
                self._add_edge((NODE_USER, "owns_card", NODE_PAYMENT), user_idx, cid)
            if agent.loyalty_account_id is not None:
                self._add_edge((NODE_USER, "has_loyalty", NODE_LOYALTY), user_idx, agent.loyalty_account_id)
            for rev_id in agent.review_ids:
                self._add_edge((NODE_USER, "wrote", NODE_REVIEW), user_idx, rev_id)

    # ------------------------------------------------------------------
    # Fraud ring injectors
    # ------------------------------------------------------------------

    def _inject_ticketing_rings(self):
        ring_type_id = 1  # FRAUD_RING_TYPES[1]
        for _ in range(self.n_ticketing_rings):
            ring_id = self._next_ring_id()
            ring = TicketingFraudRing(ring_id=ring_id, rng=self.rng,
                                      ring_size_override=self.ring_size_target)

            # Shared devices and IPs
            n_dev = ring.shared_device_count()
            n_ip  = ring.shared_ip_count()
            dev_ids = []
            for _ in range(n_dev):
                idx = self._add_node(
                    NODE_DEVICE, ring.device_features(), True, ring_id, ring_type_id)
                dev_ids.append(idx)
            ip_ids = []
            for _ in range(n_ip):
                idx = self._add_node(
                    NODE_IP, ring.ip_features(), True, ring_id, ring_type_id)
                ip_ids.append(idx)

            ring.shared_device_ids = dev_ids
            ring.shared_ip_ids     = ip_ids

            # Choose 1-3 target flights
            if len(self._flights) > 0:
                n_targets = int(self.rng.integers(1, 4))
                ring.target_flight_ids = [
                    int(self.rng.integers(0, len(self._flights)))
                    for _ in range(n_targets)
                ]

            # Orchestrator — always uses shared devices (orchestrates infrastructure)
            orch_feats = ring.fraudster_user_features(-1)
            ring.orchestrator_uid = self._add_node(
                NODE_USER, orch_feats, True, ring_id, ring_type_id)

            # Shared payment card for orchestrator
            orch_card = self._gen_payment(is_fraud=True, ring_id=ring_id,
                                          ring_type=ring_type_id, shared=ring.ring_size)
            self._add_edge((NODE_USER, "owns_card", NODE_PAYMENT), ring.orchestrator_uid, orch_card)
            for did in dev_ids:
                self._add_edge((NODE_USER, "uses_device", NODE_DEVICE), ring.orchestrator_uid, did)
            for iid in ip_ids:
                self._add_edge((NODE_USER, "uses_ip", NODE_IP), ring.orchestrator_uid, iid)

            # Satellites
            # p_use_own_device: ~25% of satellite members are "careful" fraudsters
            # who avoid the shared infrastructure and use their own device.
            # This is realistic: sophisticated ring members know shared devices
            # are a detection risk. Calibrated to industry observations that
            # ~20-30% of ring members use non-shared infrastructure.
            # p_use_own_ip: ~35% of satellites use a personal IP (e.g. mobile data)
            # rather than the ring's shared IP pool. Independent of device choice.
            # Prevents IP-channel from being a trivial perfect separator.
            p_use_own_device = 0.25
            p_use_own_ip = 0.35
            for _ in range(ring.ring_size - 1):
                sat_feats = ring.fraudster_user_features(-1)
                sat_uid = self._add_node(
                    NODE_USER, sat_feats, True, ring_id, ring_type_id)
                ring.satellite_uids.append(sat_uid)

                if self.rng.random() < p_use_own_device:
                    # "Careful" satellite: uses own unique device (no shared device edge)
                    own_dev = self._gen_device(shared_users=1)
                    self._add_edge((NODE_USER, "uses_device", NODE_DEVICE), sat_uid, own_dev)
                else:
                    # Standard satellite: shares ring infrastructure
                    for did in dev_ids:
                        self._add_edge((NODE_USER, "uses_device", NODE_DEVICE), sat_uid, did)
                if self.rng.random() < p_use_own_ip:
                    # "IP-careful" satellite: uses personal IP (mobile data / VPN)
                    own_ip = self._gen_ip(shared_users=1)
                    self._add_edge((NODE_USER, "uses_ip", NODE_IP), sat_uid, own_ip)
                else:
                    for iid in ip_ids:
                        self._add_edge((NODE_USER, "uses_ip", NODE_IP), sat_uid, iid)

                # Satellite shares orchestrator card (fraud signal)
                self._add_edge((NODE_USER, "owns_card", NODE_PAYMENT), sat_uid, orch_card)

                # Bookings
                n_bk = ring.bookings_per_satellite()
                cb_rate = ring.chargeback_rate()
                for _ in range(n_bk):
                    is_cb = self.rng.random() < cb_rate
                    b_feats = ring.booking_features(is_chargeback=is_cb)
                    bid = self._add_node(
                        NODE_BOOKING, b_feats, True, ring_id, ring_type_id)
                    ring.booking_ids.append(bid)
                    if is_cb:
                        ring.chargeback_booking_ids.append(bid)

                    self._add_edge((NODE_USER, "made", NODE_BOOKING), sat_uid, bid)
                    self._add_edge((NODE_BOOKING, "paid_with", NODE_PAYMENT), bid, orch_card)

                    if ring.target_flight_ids:
                        fid = self.rng.choice(ring.target_flight_ids)
                        self._add_edge((NODE_BOOKING, "for_flight", NODE_FLIGHT), bid, fid)

            self._ticketing_rings.append(ring)

    def _inject_ghost_hotel_rings(self):
        ring_type_id = 2
        for _ in range(self.n_ghost_hotel_rings):
            ring_id = self._next_ring_id()
            ring = GhostHotelRing(ring_id=ring_id, rng=self.rng,
                                  ring_size_override=self.ring_size_target)

            n_ghost  = ring.ghost_hotel_count()
            n_rev    = ring.reviewer_count()
            n_book   = ring.booker_count()
            n_dev    = ring.shared_device_count()
            n_ip     = ring.shared_ip_count()

            # Shared devices / IPs
            dev_ids = []
            for _ in range(n_dev):
                ring.shared_device_ids.append(
                    self._add_node(NODE_DEVICE, ring.device_features(), True, ring_id, ring_type_id)
                )
            ip_ids = ring.shared_ip_ids
            for _ in range(n_ip):
                ip_ids.append(
                    self._add_node(NODE_IP, ring.ip_features(), True, ring_id, ring_type_id)
                )
            dev_ids = ring.shared_device_ids

            # Ghost hotels
            ghost_hotel_ids = []
            for _ in range(n_ghost):
                hid = self._add_node(
                    NODE_HOTEL, ring.ghost_hotel_features(), True, ring_id, ring_type_id)
                ghost_hotel_ids.append(hid)
            ring.ghost_hotel_ids = ghost_hotel_ids

            # Reviewer accounts (~25% use own device, ~35% use own IP)
            p_own = 0.25
            p_own_ip = 0.35
            for _ in range(n_rev):
                u_feats = ring.fake_reviewer_features()
                uid = self._add_node(NODE_USER, u_feats, True, ring_id, ring_type_id)
                ring.reviewer_uids.append(uid)
                if self.rng.random() < p_own:
                    own_dev = self._gen_device(shared_users=1)
                    self._add_edge((NODE_USER, "uses_device", NODE_DEVICE), uid, own_dev)
                else:
                    for did in dev_ids:
                        self._add_edge((NODE_USER, "uses_device", NODE_DEVICE), uid, did)
                if self.rng.random() < p_own_ip:
                    own_ip = self._gen_ip(shared_users=1)
                    self._add_edge((NODE_USER, "uses_ip", NODE_IP), uid, own_ip)
                else:
                    for iid in ip_ids:
                        self._add_edge((NODE_USER, "uses_ip", NODE_IP), uid, iid)

                # Each reviewer posts to every ghost hotel (dense bipartite clique)
                for hid in ghost_hotel_ids:
                    r_feats = ring.fake_review_features()
                    rid_node = self._add_node(
                        NODE_REVIEW, r_feats, True, ring_id, ring_type_id)
                    ring.review_ids.append(rid_node)
                    self._add_edge((NODE_USER, "wrote", NODE_REVIEW), uid, rid_node)
                    self._add_edge((NODE_REVIEW, "about", NODE_HOTEL), rid_node, hid)

            # Booker accounts (~25% use own device, ~35% use own IP)
            for _ in range(n_book):
                u_feats = ring.fake_booker_features()
                uid = self._add_node(NODE_USER, u_feats, True, ring_id, ring_type_id)
                ring.booker_uids.append(uid)
                if self.rng.random() < p_own:
                    own_dev = self._gen_device(shared_users=1)
                    self._add_edge((NODE_USER, "uses_device", NODE_DEVICE), uid, own_dev)
                else:
                    for did in dev_ids:
                        self._add_edge((NODE_USER, "uses_device", NODE_DEVICE), uid, did)
                if self.rng.random() < p_own_ip:
                    own_ip = self._gen_ip(shared_users=1)
                    self._add_edge((NODE_USER, "uses_ip", NODE_IP), uid, own_ip)
                else:
                    for iid in ip_ids:
                        self._add_edge((NODE_USER, "uses_ip", NODE_IP), uid, iid)

                card_id = self._gen_payment(True, ring_id, ring_type_id, shared=n_book)
                self._add_edge((NODE_USER, "owns_card", NODE_PAYMENT), uid, card_id)

                # Bookings on ghost hotels
                for hid in ghost_hotel_ids:
                    b_feats = ring.fake_booking_features()
                    bid = self._add_node(
                        NODE_BOOKING, b_feats, True, ring_id, ring_type_id)
                    ring.booking_ids.append(bid)
                    self._add_edge((NODE_USER, "made", NODE_BOOKING), uid, bid)
                    self._add_edge((NODE_BOOKING, "for_hotel", NODE_HOTEL), bid, hid)
                    self._add_edge((NODE_BOOKING, "paid_with", NODE_PAYMENT), bid, card_id)

            self._ghost_hotel_rings.append(ring)

    def _inject_ato_rings(self):
        ring_type_id = 3
        for _ in range(self.n_ato_rings):
            ring_id = self._next_ring_id()
            ring = AccountTakeoverRing(ring_id=ring_id, rng=self.rng,
                                       ring_size_override=self.ring_size_target)

            n_comp  = ring.compromised_account_count()
            n_dev   = ring.attacker_device_count()
            n_ip    = ring.attacker_ip_count()
            n_mule  = ring.mule_loyalty_count()

            # Attacker devices / IPs
            for _ in range(n_dev):
                ring.attacker_device_ids.append(
                    self._add_node(NODE_DEVICE, ring.attacker_device_features(), True, ring_id, ring_type_id)
                )
            for _ in range(n_ip):
                ring.attacker_ip_ids.append(
                    self._add_node(NODE_IP, ring.attacker_ip_features(), True, ring_id, ring_type_id)
                )

            # Mule loyalty accounts (receive stolen points)
            for _ in range(n_mule):
                ring.mule_loyalty_ids.append(
                    self._add_node(NODE_LOYALTY, ring.mule_loyalty_features(), True, ring_id, ring_type_id)
                )

            # Compromised users (~25% accessed via victim's own device, ~35% via own IP)
            p_own_ato = 0.25
            p_own_ip_ato = 0.35
            for _ in range(n_comp):
                u_feats = ring.compromised_user_features()
                uid = self._add_node(NODE_USER, u_feats, True, ring_id, ring_type_id)
                ring.compromised_user_ids.append(uid)

                if self.rng.random() < p_own_ato:
                    # Account accessed via victim's own device (no attacker device edge)
                    own_dev = self._gen_device(shared_users=1)
                    self._add_edge((NODE_USER, "uses_device", NODE_DEVICE), uid, own_dev)
                else:
                    # Standard: attacker uses their own devices to access victim account
                    for did in ring.attacker_device_ids:
                        self._add_edge((NODE_USER, "uses_device", NODE_DEVICE), uid, did)
                if self.rng.random() < p_own_ip_ato:
                    # ATO via VPN/proxy that doesn't link to ring IP pool
                    own_ip = self._gen_ip(shared_users=1)
                    self._add_edge((NODE_USER, "uses_ip", NODE_IP), uid, own_ip)
                else:
                    for iid in ring.attacker_ip_ids:
                        self._add_edge((NODE_USER, "uses_ip", NODE_IP), uid, iid)

                # Compromised loyalty account → transfer to mule
                victim_loy_feats = ring.compromised_loyalty_features()
                vid_loy = self._add_node(
                    NODE_LOYALTY, victim_loy_feats, True, ring_id, ring_type_id)
                self._add_edge((NODE_USER, "has_loyalty", NODE_LOYALTY), uid, vid_loy)

                # Transfer to random mule
                mule_lid = self.rng.choice(ring.mule_loyalty_ids)
                self._add_edge(
                    (NODE_LOYALTY, "transferred_to", NODE_LOYALTY), vid_loy, mule_lid)
                ring.loyalty_transfer_pairs.append((vid_loy, mule_lid))

                # Fraudulent booking
                b_feats = ring.fraudulent_booking_features()
                bid = self._add_node(NODE_BOOKING, b_feats, True, ring_id, ring_type_id)
                ring.fraudulent_booking_ids.append(bid)
                self._add_edge((NODE_USER, "made", NODE_BOOKING), uid, bid)

                # Flight or hotel
                if self.rng.random() < 0.60 and len(self._flights) > 0:
                    fid = int(self.rng.integers(0, len(self._flights)))
                    self._add_edge((NODE_BOOKING, "for_flight", NODE_FLIGHT), bid, fid)
                elif len(self._hotels) > 0:
                    hid = int(self.rng.integers(0, len(self._hotels)))
                    self._add_edge((NODE_BOOKING, "for_hotel", NODE_HOTEL), bid, hid)

                card_id = self._gen_payment(True, ring_id, ring_type_id, shared=n_comp)
                self._add_edge((NODE_USER, "owns_card", NODE_PAYMENT), uid, card_id)
                self._add_edge((NODE_BOOKING, "paid_with", NODE_PAYMENT), bid, card_id)

            self._ato_rings.append(ring)

    # ------------------------------------------------------------------
    # Build orchestration
    # ------------------------------------------------------------------

    def build(self) -> GraphData:
        """Run the full simulation and return a GraphData object."""

        # 1. Legitimate infrastructure
        self._gen_legitimate_flights()

        # Reserve ~80 % of hotels for legitimate properties
        n_legit_hotels = int(self.n_hotels * 0.80)
        self._gen_legitimate_hotels(n_legit_hotels)

        # 2. Inject fraud rings (before legitimate users so node indices are set)
        self._inject_ticketing_rings()
        self._inject_ghost_hotel_rings()
        self._inject_ato_rings()

        # 3. Legitimate users fill remaining user quota
        total_fraud_users = (
            sum(len(r.satellite_uids) + 1 for r in self._ticketing_rings)
            + sum(len(r.reviewer_uids) + len(r.booker_uids) for r in self._ghost_hotel_rings)
            + sum(len(r.compromised_user_ids) for r in self._ato_rings)
        )
        n_legit_users = max(0, self.n_users - total_fraud_users)
        self._simulate_legitimate_users(n_legit_users)

        # 4. Assemble GraphData
        node_features   = {}
        node_labels     = {}
        node_ring_ids   = {}
        node_ring_types = {}

        pool_map = {
            NODE_USER:    self._users,
            NODE_DEVICE:  self._devices,
            NODE_IP:      self._ips,
            NODE_BOOKING: self._bookings,
            NODE_FLIGHT:  self._flights,
            NODE_HOTEL:   self._hotels,
            NODE_REVIEW:  self._reviews,
            NODE_PAYMENT: self._payments,
            NODE_LOYALTY: self._loyalties,
        }
        for ntype, pool in pool_map.items():
            node_features[ntype]   = pool if not isinstance(pool[0] if pool else None, TravelerAgent) else [a.feature_vector() for a in pool]
            node_labels[ntype]     = self._labels[ntype]
            node_ring_ids[ntype]   = self._ring_ids[ntype]
            node_ring_types[ntype] = self._ring_types[ntype]

        # Fix user pool (TravelerAgent objects need feature_vector call)
        if self._users and isinstance(self._users[0], TravelerAgent):
            node_features[NODE_USER] = [u.feature_vector() for u in self._users]
        else:
            node_features[NODE_USER] = self._users  # dicts already

        metadata = {
            "seed":                  self.seed,
            "n_users_total":         len(self._users),
            "n_hotels_total":        len(self._hotels),
            "n_flights_total":       len(self._flights),
            "n_bookings_total":      len(self._bookings),
            "n_devices_total":       len(self._devices),
            "n_ips_total":           len(self._ips),
            "n_reviews_total":       len(self._reviews),
            "n_payments_total":      len(self._payments),
            "n_loyalties_total":     len(self._loyalties),
            "n_ticketing_rings":     self.n_ticketing_rings,
            "n_ghost_hotel_rings":   self.n_ghost_hotel_rings,
            "n_ato_rings":           self.n_ato_rings,
            "total_rings":           self._ring_counter,
            "fraud_user_ratio":      round(sum(self._labels[NODE_USER]) / max(len(self._labels[NODE_USER]), 1), 4),
        }

        return GraphData(
            node_features=node_features,
            node_labels=node_labels,
            node_ring_ids=node_ring_ids,
            node_ring_types=node_ring_types,
            edges=dict(self._edges),
            metadata=metadata,
        )
