"""
Graph schema definitions for the Travel Fraud Graph dataset.

Node types, edge types, feature descriptors, and label conventions
used throughout the generator. Intended as the single source of truth
for downstream exporters and documentation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Node type identifiers
# ---------------------------------------------------------------------------
NODE_USER = "user"
NODE_DEVICE = "device"
NODE_IP = "ip_address"
NODE_BOOKING = "booking"
NODE_FLIGHT = "flight"
NODE_HOTEL = "hotel"
NODE_REVIEW = "review"
NODE_PAYMENT = "payment_card"
NODE_LOYALTY = "loyalty_account"

ALL_NODE_TYPES: List[str] = [
    NODE_USER, NODE_DEVICE, NODE_IP, NODE_BOOKING,
    NODE_FLIGHT, NODE_HOTEL, NODE_REVIEW, NODE_PAYMENT, NODE_LOYALTY,
]

# ---------------------------------------------------------------------------
# Edge type identifiers  (src_type, relation, dst_type)
# ---------------------------------------------------------------------------
EDGE_USER_BOOKING    = (NODE_USER,    "made",         NODE_BOOKING)
EDGE_USER_DEVICE     = (NODE_USER,    "uses_device",  NODE_DEVICE)
EDGE_USER_IP         = (NODE_USER,    "uses_ip",      NODE_IP)
EDGE_USER_LOYALTY    = (NODE_USER,    "has_loyalty",  NODE_LOYALTY)
EDGE_USER_PAYMENT    = (NODE_USER,    "owns_card",    NODE_PAYMENT)
EDGE_USER_REVIEW     = (NODE_USER,    "wrote",        NODE_REVIEW)
EDGE_BOOKING_FLIGHT  = (NODE_BOOKING, "for_flight",   NODE_FLIGHT)
EDGE_BOOKING_HOTEL   = (NODE_BOOKING, "for_hotel",    NODE_HOTEL)
EDGE_BOOKING_PAYMENT = (NODE_BOOKING, "paid_with",    NODE_PAYMENT)
EDGE_REVIEW_HOTEL    = (NODE_REVIEW,  "about",        NODE_HOTEL)
EDGE_USER_REFERRED   = (NODE_USER,    "referred",     NODE_USER)
EDGE_LOYALTY_TRANSFER = (NODE_LOYALTY, "transferred_to", NODE_LOYALTY)

ALL_EDGE_TYPES: List[Tuple[str, str, str]] = [
    EDGE_USER_BOOKING,
    EDGE_USER_DEVICE,
    EDGE_USER_IP,
    EDGE_USER_LOYALTY,
    EDGE_USER_PAYMENT,
    EDGE_USER_REVIEW,
    EDGE_BOOKING_FLIGHT,
    EDGE_BOOKING_HOTEL,
    EDGE_BOOKING_PAYMENT,
    EDGE_REVIEW_HOTEL,
    EDGE_USER_REFERRED,
    EDGE_LOYALTY_TRANSFER,
]

# ---------------------------------------------------------------------------
# Feature schemas  (name -> description)
# ---------------------------------------------------------------------------
USER_FEATURES: Dict[str, str] = {
    "account_age_days":      "Days since account creation",
    "booking_count_30d":     "Number of bookings in last 30 days",
    "cancellation_rate":     "Fraction of bookings cancelled (0-1)",
    "chargeback_count":      "Total chargebacks ever filed",
    "distinct_device_count": "Number of distinct devices used in 90d",
    "distinct_ip_count":     "Number of distinct IP addresses in 90d",
    "country_code":          "ISO-3166-1 numeric country code",
    "is_loyalty_member":     "Binary: has active loyalty account",
    "avg_booking_value_usd": "Average booking value in USD",
    "referral_count":        "Number of users referred",
    "velocity_score":        "Bookings per hour (max 24h window)",
}

DEVICE_FEATURES: Dict[str, str] = {
    "device_type":       "0=desktop, 1=mobile, 2=tablet",
    "os_type":           "0=Windows, 1=macOS, 2=iOS, 3=Android, 4=Linux",
    "shared_user_count": "Number of distinct users linked to this device",
    "first_seen_days":   "Days since first observed",
    "is_emulator":       "Binary: detected as emulator/virtual device",
}

IP_FEATURES: Dict[str, str] = {
    "is_vpn":              "Binary: known VPN/proxy exit node",
    "is_datacenter":       "Binary: belongs to a datacenter ASN",
    "country_code":        "ISO-3166-1 numeric country code of IP geolocation",
    "shared_user_count":   "Number of distinct users from this IP in 30d",
    "abuse_score":         "0-100 abuse confidence from threat intelligence",
}

BOOKING_FEATURES: Dict[str, str] = {
    "booking_value_usd":    "Total booking value in USD",
    "lead_time_days":       "Days between booking and travel date",
    "duration_nights":      "Number of nights (0 for flight-only)",
    "is_cancelled":         "Binary: booking was cancelled",
    "is_refunded":          "Binary: refund was issued",
    "chargeback_flag":      "Binary: chargeback was raised",
    "booking_channel":      "0=web, 1=mobile, 2=api, 3=agent",
    "passengers":           "Number of passengers / guests",
    "timestamp_unix":       "Unix timestamp of booking creation",
}

FLIGHT_FEATURES: Dict[str, str] = {
    "origin_airport":       "IATA origin code (encoded integer)",
    "dest_airport":         "IATA destination code (encoded integer)",
    "airline_code":         "IATA airline code (encoded integer)",
    "departure_unix":       "Unix timestamp of scheduled departure",
    "seat_class":           "0=economy, 1=premium, 2=business, 3=first",
    "base_price_usd":       "Base price per seat in USD",
    "remaining_seats":      "Seats available at time of booking",
}

HOTEL_FEATURES: Dict[str, str] = {
    "hotel_class":          "Star rating 1-5",
    "review_count":         "Total number of reviews",
    "avg_rating":           "Average rating 1.0-5.0",
    "listing_age_days":     "Days since property was listed",
    "cancellation_policy":  "0=flexible, 1=moderate, 2=strict",
    "country_code":         "ISO-3166-1 numeric country code",
    "is_ghost":             "Binary: synthetic ghost listing (fraud)",
    "photo_count":          "Number of photos on listing",
    "price_percentile":     "Price percentile vs comparable hotels (0-100)",
}

REVIEW_FEATURES: Dict[str, str] = {
    "rating":               "Integer rating 1-5",
    "verified_booking":     "Binary: reviewer has verified booking",
    "review_length_chars":  "Length of review text in characters",
    "sentiment_score":      "Sentiment score -1.0 to 1.0",
    "days_after_checkin":   "Days between checkout and review posted",
    "is_incentivised":      "Binary: review linked to reward redemption",
}

PAYMENT_FEATURES: Dict[str, str] = {
    "card_type":            "0=credit, 1=debit, 2=prepaid, 3=virtual",
    "issuer_country_code":  "ISO-3166-1 numeric country of issuing bank",
    "shared_user_count":    "Number of distinct users who used this card",
    "total_bookings":       "Total bookings charged to this card",
    "chargeback_count":     "Total chargebacks on this card",
    "is_compromised":       "Binary: card flagged as compromised",
}

LOYALTY_FEATURES: Dict[str, str] = {
    "point_balance":        "Current loyalty point balance",
    "lifetime_points":      "Total points ever earned",
    "redemption_count_30d": "Redemptions in last 30 days",
    "transfer_count_30d":   "Point transfers in last 30 days",
    "account_age_days":     "Days since loyalty enrollment",
    "status_tier":          "0=base, 1=silver, 2=gold, 3=platinum",
    "suspicious_velocity":  "Binary: high velocity transfer/redeem detected",
}

ALL_FEATURE_SCHEMAS: Dict[str, Dict[str, str]] = {
    NODE_USER:    USER_FEATURES,
    NODE_DEVICE:  DEVICE_FEATURES,
    NODE_IP:      IP_FEATURES,
    NODE_BOOKING: BOOKING_FEATURES,
    NODE_FLIGHT:  FLIGHT_FEATURES,
    NODE_HOTEL:   HOTEL_FEATURES,
    NODE_REVIEW:  REVIEW_FEATURES,
    NODE_PAYMENT: PAYMENT_FEATURES,
    NODE_LOYALTY: LOYALTY_FEATURES,
}

# ---------------------------------------------------------------------------
# Label conventions
# ---------------------------------------------------------------------------
LABEL_LEGITIMATE = 0
LABEL_FRAUD      = 1

FRAUD_RING_TYPES = {
    0: "legitimate",
    1: "ticketing_fraud_ring",
    2: "ghost_hotel_ring",
    3: "account_takeover_ring",
}
