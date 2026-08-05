"""Builds evidence_ids and affected_entities strictly from verified rows
(README section 5). Every id here must be reconstructible from the CSVs —
never from LLM free text — otherwise it is a false positive under the
grading rule.
"""
from __future__ import annotations

from typing import Any

from . import config


def build_affected_entities(
    order_id: str,
    items: list[dict[str, Any]],
    payments: list[dict[str, Any]],
    responsible_parties: list[dict[str, str]],
) -> dict[str, list[str]]:
    seller_ids = [p["party_id"] for p in responsible_parties if p["party_type"] == "seller"]
    if not seller_ids:
        seller_ids = sorted({i["seller_id"] for i in items})

    return {
        "order_ids": [order_id][: config.MAX_ENTITY_IDS],
        "item_ids": [f"{order_id}:{i['order_item_id']}" for i in items][: config.MAX_ENTITY_IDS],
        "seller_ids": seller_ids[: config.MAX_ENTITY_IDS],
        "payment_ids": [f"{order_id}:{p['payment_sequential']}" for p in payments][
            : config.MAX_ENTITY_IDS
        ],
    }


def build_evidence_ids(
    order_id: str,
    items: list[dict[str, Any]],
    payments: list[dict[str, Any]],
    responsible_parties: list[dict[str, str]],
    root_cause_code: str,
) -> list[str]:
    evidence = [f"order:{order_id}"]

    seller_ids = [p["party_id"] for p in responsible_parties if p["party_type"] == "seller"]
    if not seller_ids:
        seller_ids = sorted({i["seller_id"] for i in items})

    for item in items:
        evidence.append(f"item:{order_id}:{item['order_item_id']}")
    for payment in payments:
        evidence.append(f"payment:{order_id}:{payment['payment_sequential']}")
    for seller_id in seller_ids:
        evidence.append(f"seller:{seller_id}")
    evidence.append(f"policy:{root_cause_code}")

    # Trim while always keeping order + policy evidence (the two ids that
    # anchor the decision) inside the max-10 cap.
    if len(evidence) > config.MAX_EVIDENCE_IDS:
        head = [evidence[0]]
        tail = [evidence[-1]]
        middle = evidence[1:-1][: config.MAX_EVIDENCE_IDS - 2]
        evidence = head + middle + tail
    return evidence
