"""Order & Seller Agent: order status, items and seller handoff timing.

Access scope: orders.csv (status only) + order_items.csv + sellers.csv,
already resolved by the Coordinator into state["order"]/state["items"].
This agent does not see payment rows.
"""
from __future__ import annotations

from .. import config
from ..llm import run_agent_note
from ..trace_logger import log_event
from .state import CaseState

SYSTEM_PROMPT = (
    "You are the Order & Seller agent in an e-commerce dispute pipeline. "
    "You are given the order status and its item/seller rows, already "
    "verified against the dataset. Summarize whether the order looks "
    "canceled/unavailable, and whether any seller handed the item to the "
    "carrier after that item's shipping_limit_date. Do not invent dates or "
    "ids that are not in the evidence."
)


def order_seller_node(state: CaseState) -> CaseState:
    order = state["order"]
    items = state["items"]

    seller_ids = sorted({i["seller_id"] for i in items})
    evidence = {
        "order_status": (order or {}).get("order_status"),
        "order_delivered_carrier_date": (order or {}).get("order_delivered_carrier_date"),
        "items": [
            {
                "order_item_id": i["order_item_id"],
                "seller_id": i["seller_id"],
                "shipping_limit_date": i["shipping_limit_date"],
            }
            for i in items
        ],
        "distinct_seller_ids": seller_ids,
    }

    note, meta = run_agent_note(
        agent_name="order_seller", system_prompt=SYSTEM_PROMPT, evidence=evidence
    )
    log_event(
        case_id=state["case_id"],
        agent="order_seller",
        event="analyze",
        data={"evidence": evidence, "note": note.model_dump(), "llm": meta},
    )
    return {**state, "order_seller_note": note.model_dump()}
