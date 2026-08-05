"""Payment Agent: reconciles payment rows against item + freight totals.

Access scope: order_payments.csv rows plus the item/freight totals already
computed by the Order & Seller agent. Does not see delivery timestamps.
"""
from __future__ import annotations

from ..llm import run_agent_note
from ..trace_logger import log_event
from .state import CaseState

SYSTEM_PROMPT = (
    "You are the Payment agent in an e-commerce dispute pipeline. You are "
    "given payment rows and the item+freight total, already computed from "
    "the dataset. State whether the payments reconcile with the order "
    "total (within 0.10 BRL) and whether this is a split payment (2+ "
    "rows). Do not invent amounts."
)


def payment_node(state: CaseState) -> CaseState:
    items = state["items"]
    payments = state["payments"]
    item_total = round(sum(i["price"] for i in items), 2)
    freight_total = round(sum(i["freight_value"] for i in items), 2)
    payment_total = round(sum(p["payment_value"] for p in payments), 2)

    evidence = {
        "payment_rows": [
            {"payment_sequential": p["payment_sequential"], "payment_value": p["payment_value"]}
            for p in payments
        ],
        "payment_total_brl": payment_total,
        "item_total_brl": item_total,
        "freight_total_brl": freight_total,
        "expected_total_brl": round(item_total + freight_total, 2),
    }

    note, meta = run_agent_note(
        agent_name="payment", system_prompt=SYSTEM_PROMPT, evidence=evidence
    )
    log_event(
        case_id=state["case_id"],
        agent="payment",
        event="analyze",
        data={"evidence": evidence, "note": note.model_dump(), "llm": meta},
    )
    return {**state, "payment_note": note.model_dump()}
