"""Coordinator Agent: case intake (resolve claimed_order_id against the
dataset) and final output aggregation. Pure orchestration/routing — no LLM,
per README's own description of the Coordinator's job ("nhận case, giao
việc và tổng hợp output").
"""
from __future__ import annotations

from typing import Any

from .. import evidence
from ..data_store import load_data_store
from ..trace_logger import log_event
from .state import CaseState


def intake_node(state: CaseState) -> CaseState:
    store = load_data_store()
    order_id = state["customer_request"]["claimed_order_id"]

    order = store.order(order_id)
    items = store.order_items_for(order_id) if order else []
    payments = store.payments_for(order_id) if order else []

    log_event(
        case_id=state["case_id"],
        agent="coordinator",
        event="intake",
        data={
            "claimed_order_id": order_id,
            "order_found": order is not None,
            "item_count": len(items),
            "payment_count": len(payments),
        },
    )

    return {**state, "order": order, "items": items, "payments": payments}


def aggregate_node(state: CaseState) -> CaseState:
    order_id = state["customer_request"]["claimed_order_id"]
    policy_result = state["policy_result"]

    affected_entities = evidence.build_affected_entities(
        order_id, state["items"], state["payments"], policy_result["responsible_parties"]
    )
    evidence_ids = evidence.build_evidence_ids(
        order_id,
        state["items"],
        state["payments"],
        policy_result["responsible_parties"],
        policy_result["root_cause_code"],
    )

    draft: dict[str, Any] = {
        "case_id": state["case_id"],
        "assessment": {
            "primary_issue": policy_result["primary_issue"],
            "case_status": policy_result["case_status"],
            "confidence": policy_result["confidence"],
        },
        "affected_entities": affected_entities,
        "root_cause_analysis": {
            "ranked_causes": [{"cause_code": policy_result["root_cause_code"], "rank": 1}],
            "responsible_parties": policy_result["responsible_parties"],
        },
        "evidence_ids": evidence_ids,
        "financial_resolution": policy_result["financial_resolution"],
        "resolution_actions": policy_result["resolution_actions"],
    }

    log_event(case_id=state["case_id"], agent="coordinator", event="aggregate", data=draft)
    return {**state, "draft_output": draft}
