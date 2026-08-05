"""Delivery Agent: compares actual delivery timing against the estimate.

Access scope: orders.csv delivery/estimate timestamps only. No item, seller
or payment fields.
"""
from __future__ import annotations

from ..llm import run_agent_note
from ..trace_logger import log_event
from .state import CaseState

SYSTEM_PROMPT = (
    "You are the Delivery agent in an e-commerce dispute pipeline. You are "
    "given the order's delivery timestamps, already verified against the "
    "dataset. State plainly whether the order was delivered to the "
    "customer after the estimated delivery date. Do not invent dates."
)


def delivery_node(state: CaseState) -> CaseState:
    order = state["order"] or {}
    evidence = {
        "order_delivered_customer_date": order.get("order_delivered_customer_date"),
        "order_estimated_delivery_date": order.get("order_estimated_delivery_date"),
        "order_delivered_carrier_date": order.get("order_delivered_carrier_date"),
    }

    note, meta = run_agent_note(
        agent_name="delivery", system_prompt=SYSTEM_PROMPT, evidence=evidence
    )
    log_event(
        case_id=state["case_id"],
        agent="delivery",
        event="analyze",
        data={"evidence": evidence, "note": note.model_dump(), "llm": meta},
    )
    return {**state, "delivery_note": note.model_dump()}
