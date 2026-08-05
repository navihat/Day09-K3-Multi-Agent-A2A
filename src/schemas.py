"""Pydantic models for the case input and the required output schema
(README section 3 and 6). These are the contract every agent hands off
through — the Verifier agent uses them to hard-validate before a file is
written.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from . import config


class CustomerRequest(BaseModel):
    language: str
    message: str
    claimed_order_id: str


class CaseInput(BaseModel):
    case_id: str
    opened_at: str
    customer_request: CustomerRequest
    policy_version: str


class Assessment(BaseModel):
    primary_issue: Literal[
        "canceled_order_paid",
        "unavailable_order_paid",
        "late_delivery_seller",
        "late_delivery_logistics",
        "valid_split_payment",
        "unsupported_late_claim",
    ]
    case_status: Literal["action_required", "no_action"]
    confidence: float = Field(ge=0.0, le=1.0)


class AffectedEntities(BaseModel):
    order_ids: list[str] = Field(default_factory=list, max_length=config.MAX_ENTITY_IDS)
    item_ids: list[str] = Field(default_factory=list, max_length=config.MAX_ENTITY_IDS)
    seller_ids: list[str] = Field(default_factory=list, max_length=config.MAX_ENTITY_IDS)
    payment_ids: list[str] = Field(default_factory=list, max_length=config.MAX_ENTITY_IDS)


class RankedCause(BaseModel):
    cause_code: str
    rank: int


class ResponsibleParty(BaseModel):
    party_type: Literal["platform", "seller", "logistics_provider"]
    party_id: str


class RootCauseAnalysis(BaseModel):
    ranked_causes: list[RankedCause] = Field(default_factory=list, max_length=config.MAX_ROOT_CAUSES)
    responsible_parties: list[ResponsibleParty] = Field(
        default_factory=list, max_length=config.MAX_RESPONSIBLE_PARTIES
    )


class FinancialResolution(BaseModel):
    currency: Literal["BRL"] = "BRL"
    item_total_brl: float
    freight_total_brl: float
    payment_total_brl: float
    recommended_refund_brl: float

    @field_validator(
        "item_total_brl", "freight_total_brl", "payment_total_brl", "recommended_refund_brl"
    )
    @classmethod
    def _round2(cls, v: float) -> float:
        return round(v, 2)


class CaseOutput(BaseModel):
    case_id: str
    assessment: Assessment
    affected_entities: AffectedEntities
    root_cause_analysis: RootCauseAnalysis
    evidence_ids: list[str] = Field(default_factory=list, max_length=config.MAX_EVIDENCE_IDS)
    financial_resolution: FinancialResolution
    resolution_actions: list[str] = Field(default_factory=list, max_length=config.MAX_ACTIONS)
