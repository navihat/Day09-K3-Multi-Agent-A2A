"""Verifier Agent: hard gate before a file is written.

Deliberately has no LLM step — README scores a hard gate (0 points) for
cases with malformed/non-existent evidence ids, so the last check before
writing a file must be deterministic, not a model's best guess. It
re-derives every evidence id from the same verified rows the other agents
used and rejects anything that doesn't match.
"""
from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from .. import config
from ..data_store import load_data_store
from ..schemas import CaseOutput
from ..trace_logger import log_event
from .state import CaseState

_ID_PATTERNS = {
    "order": re.compile(r"^order:(?P<order_id>[^:]+)$"),
    "item": re.compile(r"^item:(?P<order_id>[^:]+):(?P<item_id>[^:]+)$"),
    "payment": re.compile(r"^payment:(?P<order_id>[^:]+):(?P<seq>[^:]+)$"),
    "seller": re.compile(r"^seller:(?P<seller_id>[^:]+)$"),
    "policy": re.compile(r"^policy:(?P<code>[^:]+)$"),
}

_KNOWN_ROOT_CAUSES = {
    "SELLER_HANDOFF_AFTER_LIMIT",
    "CARRIER_DELIVERED_AFTER_ESTIMATE",
    "ORDER_CANCELED_AFTER_PAYMENT",
    "ORDER_UNAVAILABLE_AFTER_PAYMENT",
    "MULTIPLE_PAYMENTS_RECONCILED",
    "DELIVERY_WITHIN_ESTIMATE",
}


def _verify_evidence_ids(evidence_ids: list[str], order_id: str, items: list[dict], payments: list[dict]) -> list[str]:
    store = load_data_store()
    item_ids = {str(i["order_item_id"]) for i in items}
    payment_seqs = {str(p["payment_sequential"]) for p in payments}
    issues: list[str] = []

    for eid in evidence_ids:
        kind = eid.split(":", 1)[0]
        pattern = _ID_PATTERNS.get(kind)
        if pattern is None:
            issues.append(f"unknown_evidence_kind:{eid}")
            continue
        m = pattern.match(eid)
        if not m:
            issues.append(f"malformed_evidence_id:{eid}")
            continue

        if kind == "order":
            if m.group("order_id") != order_id or store.order(order_id) is None:
                issues.append(f"nonexistent_order_evidence:{eid}")
        elif kind == "item":
            if m.group("order_id") != order_id or m.group("item_id") not in item_ids:
                issues.append(f"nonexistent_item_evidence:{eid}")
        elif kind == "payment":
            if m.group("order_id") != order_id or m.group("seq") not in payment_seqs:
                issues.append(f"nonexistent_payment_evidence:{eid}")
        elif kind == "seller":
            if store.seller(m.group("seller_id")) is None:
                issues.append(f"nonexistent_seller_evidence:{eid}")
        elif kind == "policy":
            if m.group("code") not in _KNOWN_ROOT_CAUSES:
                issues.append(f"unknown_policy_code:{eid}")

    return issues


def verifier_node(state: CaseState) -> CaseState:
    draft = state["draft_output"]
    order_id = state["customer_request"]["claimed_order_id"]

    issues = _verify_evidence_ids(draft["evidence_ids"], order_id, state["items"], state["payments"])

    validated: dict[str, Any] | None = None
    try:
        validated = CaseOutput.model_validate(draft).model_dump()
    except ValidationError as exc:
        issues.append(f"schema_validation_error:{exc}")

    hard_gate_failed = bool(issues)
    log_event(
        case_id=state["case_id"],
        agent="verifier",
        event="verify",
        data={"issues": issues, "hard_gate_failed": hard_gate_failed},
    )

    return {
        **state,
        "verifier_issues": issues,
        "output": validated if not hard_gate_failed else draft,
        "hard_gate_failed": hard_gate_failed,
    }
