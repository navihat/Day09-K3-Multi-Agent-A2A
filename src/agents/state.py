"""Shared state object handed off between agent nodes in the LangGraph.

This *is* the handoff contract: each node reads the fields upstream agents
filled in and writes its own — nothing is passed through a single shared
prompt.
"""
from __future__ import annotations

from typing import Any, TypedDict


class CaseState(TypedDict, total=False):
    case_id: str
    opened_at: str
    customer_request: dict[str, Any]
    policy_version: str

    order: dict[str, Any] | None
    items: list[dict[str, Any]]
    payments: list[dict[str, Any]]

    order_seller_note: dict[str, Any]
    delivery_note: dict[str, Any]
    payment_note: dict[str, Any]
    policy_result: dict[str, Any]
    policy_note: dict[str, Any]

    draft_output: dict[str, Any]
    verifier_issues: list[str]
    output: dict[str, Any] | None
    hard_gate_failed: bool
