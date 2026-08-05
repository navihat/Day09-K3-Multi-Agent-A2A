"""Deterministic EC_POLICY_V1 rule engine (README section 4).

Numbers that feed the score (refund amounts, entity ids, evidence ids) must
never come from an LLM guess — they are computed here from the verified CSV
rows the Order/Seller, Delivery and Payment agents collected. The Policy
agent node wraps this pure function; the LLM only narrates the outcome.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from . import config

FREIGHT_TOLERANCE_BRL = 0.10

_ROOT_CAUSE_BY_ISSUE = {
    "canceled_order_paid": "ORDER_CANCELED_AFTER_PAYMENT",
    "unavailable_order_paid": "ORDER_UNAVAILABLE_AFTER_PAYMENT",
    "late_delivery_seller": "SELLER_HANDOFF_AFTER_LIMIT",
    "late_delivery_logistics": "CARRIER_DELIVERED_AFTER_ESTIMATE",
    "valid_split_payment": "MULTIPLE_PAYMENTS_RECONCILED",
    "unsupported_late_claim": "DELIVERY_WITHIN_ESTIMATE",
}

_ACTION_BY_ISSUE = {
    "canceled_order_paid": "issue_full_refund",
    "unavailable_order_paid": "issue_full_refund",
    "late_delivery_seller": "refund_freight",
    "late_delivery_logistics": "refund_freight",
    "valid_split_payment": "explain_valid_split_payment",
    "unsupported_late_claim": "reject_late_refund",
}

_ACTION_REQUIRED_ISSUES = {
    "canceled_order_paid",
    "unavailable_order_paid",
    "late_delivery_seller",
    "late_delivery_logistics",
}


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    return datetime.fromisoformat(ts)


def _after(a: datetime | None, b: datetime | None) -> bool:
    """True only if both timestamps exist and a is strictly after b."""
    if a is None or b is None:
        return False
    return a > b


def apply_policy(
    order: dict[str, Any] | None,
    items: list[dict[str, Any]],
    payments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate the priority-ordered rule table for one case.

    Returns a plain dict with everything the Verifier agent needs to build
    the final CaseOutput: primary_issue, case_status, confidence,
    responsible parties, root cause, financial totals and action.
    """
    payment_total = round(sum(p["payment_value"] for p in payments), 2)
    item_total = round(sum(i["price"] for i in items), 2) if items else 0.0
    freight_total = round(sum(i["freight_value"] for i in items), 2) if items else 0.0

    order_status = (order or {}).get("order_status")
    delivered_customer = _parse((order or {}).get("order_delivered_customer_date"))
    estimated = _parse((order or {}).get("order_estimated_delivery_date"))
    delivered_carrier = _parse((order or {}).get("order_delivered_carrier_date"))

    confidence_penalty = 0.0
    if order is None:
        confidence_penalty += 0.4

    late_to_customer = _after(delivered_customer, estimated)

    # Sellers whose handoff to the carrier happened after their own item's
    # shipping_limit_date (README section 4 convention for multi-item orders).
    late_sellers: list[str] = []
    on_time_sellers: list[str] = []
    for item in items:
        limit = _parse(item["shipping_limit_date"])
        if _after(delivered_carrier, limit):
            if item["seller_id"] not in late_sellers:
                late_sellers.append(item["seller_id"])
        else:
            if item["seller_id"] not in on_time_sellers:
                on_time_sellers.append(item["seller_id"])

    payment_matches_items = (
        abs(payment_total - round(item_total + freight_total, 2)) <= FREIGHT_TOLERANCE_BRL
    )

    responsible_parties: list[dict[str, str]] = []
    primary_issue: str
    refund: float

    if order_status == "canceled" and payment_total > 0:
        primary_issue = "canceled_order_paid"
        responsible_parties = [{"party_type": "platform", "party_id": "OLIST_PLATFORM"}]
        refund = payment_total
    elif order_status == "unavailable" and payment_total > 0:
        primary_issue = "unavailable_order_paid"
        responsible_parties = [{"party_type": "platform", "party_id": "OLIST_PLATFORM"}]
        refund = payment_total
    elif late_to_customer and late_sellers:
        primary_issue = "late_delivery_seller"
        responsible_parties = [{"party_type": "seller", "party_id": sid} for sid in late_sellers[:3]]
        refund = freight_total
    elif late_to_customer and not late_sellers:
        primary_issue = "late_delivery_logistics"
        responsible_parties = [{"party_type": "logistics_provider", "party_id": "LOGISTICS_PROVIDER"}]
        refund = freight_total
    elif len(payments) >= 2 and payment_matches_items:
        primary_issue = "valid_split_payment"
        responsible_parties = []
        refund = 0.0
    else:
        primary_issue = "unsupported_late_claim"
        if not payment_matches_items:
            confidence_penalty += 0.15
        responsible_parties = []
        refund = 0.0

    root_cause_code = _ROOT_CAUSE_BY_ISSUE[primary_issue]
    action = _ACTION_BY_ISSUE[primary_issue]
    case_status = "action_required" if primary_issue in _ACTION_REQUIRED_ISSUES else "no_action"
    confidence = round(max(0.5, 0.97 - confidence_penalty), 2)

    return {
        "primary_issue": primary_issue,
        "case_status": case_status,
        "confidence": confidence,
        "root_cause_code": root_cause_code,
        "responsible_parties": responsible_parties,
        "financial_resolution": {
            "currency": "BRL",
            "item_total_brl": item_total,
            "freight_total_brl": freight_total,
            "payment_total_brl": payment_total,
            "recommended_refund_brl": round(refund, 2),
        },
        "resolution_actions": [action],
        "late_sellers": late_sellers,
        "late_to_customer": late_to_customer,
    }
