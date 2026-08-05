"""Policy Agent: applies EC_POLICY_V1 to the evidence handed off by the
Order & Seller, Delivery and Payment agents.

The rule table (README section 4) is a strict priority-ordered lookup, so
the decision itself is computed deterministically by policy.apply_policy —
an LLM judgment call here would risk picking the wrong branch and failing
the hard gate. The model's role is limited to producing a short rationale
for the trace, grounded in the three upstream notes.
"""
from __future__ import annotations

from .. import policy
from ..llm import run_agent_note
from ..trace_logger import log_event
from .state import CaseState

SYSTEM_PROMPT = (
    "You are the Policy agent in an e-commerce dispute pipeline. You are "
    "given the deterministic EC_POLICY_V1 outcome plus the upstream "
    "agents' notes. Write a one- or two-sentence rationale for why this "
    "outcome follows from the evidence. Do not change the outcome or "
    "invent new facts."
)


def policy_node(state: CaseState) -> CaseState:
    result = policy.apply_policy(state["order"], state["items"], state["payments"])

    evidence = {
        "policy_outcome": {
            "primary_issue": result["primary_issue"],
            "root_cause_code": result["root_cause_code"],
            "recommended_refund_brl": result["financial_resolution"]["recommended_refund_brl"],
        },
        "order_seller_note": state.get("order_seller_note"),
        "delivery_note": state.get("delivery_note"),
        "payment_note": state.get("payment_note"),
    }

    note, meta = run_agent_note(agent_name="policy", system_prompt=SYSTEM_PROMPT, evidence=evidence)
    log_event(
        case_id=state["case_id"],
        agent="policy",
        event="apply_policy",
        data={"result": result, "note": note.model_dump(), "llm": meta},
    )
    return {**state, "policy_result": result, "policy_note": note.model_dump()}
