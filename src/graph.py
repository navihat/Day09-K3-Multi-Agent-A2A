"""Wires the six agents into a single LangGraph pipeline:

coordinator.intake -> order_seller -> delivery -> payment -> policy
  -> coordinator.aggregate -> verifier -> END

This linear chain is the handoff path: each node reads only the state keys
upstream agents produced and writes its own, mirroring README section 7's
suggested architecture.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from .agents.coordinator import aggregate_node, intake_node
from .agents.delivery import delivery_node
from .agents.order_seller import order_seller_node
from .agents.payment import payment_node
from .agents.policy_agent import policy_node
from .agents.state import CaseState
from .agents.verifier import verifier_node


def build_graph():
    graph = StateGraph(CaseState)

    graph.add_node("coordinator_intake", intake_node)
    graph.add_node("order_seller_agent", order_seller_node)
    graph.add_node("delivery_agent", delivery_node)
    graph.add_node("payment_agent", payment_node)
    graph.add_node("policy_agent", policy_node)
    graph.add_node("coordinator_aggregate", aggregate_node)
    graph.add_node("verifier_agent", verifier_node)

    graph.set_entry_point("coordinator_intake")
    graph.add_edge("coordinator_intake", "order_seller_agent")
    graph.add_edge("order_seller_agent", "delivery_agent")
    graph.add_edge("delivery_agent", "payment_agent")
    graph.add_edge("payment_agent", "policy_agent")
    graph.add_edge("policy_agent", "coordinator_aggregate")
    graph.add_edge("coordinator_aggregate", "verifier_agent")
    graph.add_edge("verifier_agent", END)

    return graph.compile()
