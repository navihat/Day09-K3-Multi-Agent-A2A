"""Thin Groq LLM wrapper shared by every LLM-backed agent node.

Each domain agent (Order&Seller, Delivery, Payment, Policy) calls the model
to *reason and narrate* over evidence that Python has already computed
deterministically from the CSVs. The model never invents the numbers that
feed the score — see policy.py / evidence.py for those. If the call fails
after retries, the agent falls back to a deterministic templated note
instead of blocking the whole case, and the failure is recorded in the
trace rather than hidden.
"""
from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from . import config


class AgentNote(BaseModel):
    summary: str = Field(description="1-2 sentence plain-language finding for this domain.")
    flags: list[str] = Field(
        default_factory=list, description="Short anomaly tags, empty if nothing unusual."
    )


_client = None


def _get_llm():
    global _client
    if _client is None:
        from langchain_groq import ChatGroq

        _client = ChatGroq(model=config.GROQ_MODEL_NAME, temperature=config.LLM_TEMPERATURE)
    return _client


def run_agent_note(
    *, agent_name: str, system_prompt: str, evidence: dict[str, Any], retries: int = 2
) -> tuple[AgentNote, dict[str, Any]]:
    """Ask the LLM to narrate over already-computed evidence.

    Returns (note, trace_meta). trace_meta always includes ok/latency_ms/error
    so the caller can write an honest trace line regardless of outcome.
    """
    import json

    llm = _get_llm().with_structured_output(AgentNote)
    user_content = (
        "Verified evidence extracted from the dataset (do not invent facts beyond this):\n"
        + json.dumps(evidence, ensure_ascii=False, default=str)
    )
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        start = time.monotonic()
        try:
            result: AgentNote = llm.invoke(
                [("system", system_prompt), ("human", user_content)]
            )
            latency_ms = round((time.monotonic() - start) * 1000, 1)
            return result, {
                "ok": True,
                "latency_ms": latency_ms,
                "attempt": attempt + 1,
                "model": config.GROQ_MODEL_NAME,
            }
        except Exception as exc:  # noqa: BLE001 - external API, broad by design
            last_error = exc
            time.sleep(0.5 * (attempt + 1))

    fallback = AgentNote(
        summary=f"{agent_name}: LLM call failed after {retries + 1} attempts; "
        "proceeding with deterministic evidence only.",
        flags=["llm_unavailable"],
    )
    return fallback, {
        "ok": False,
        "error": str(last_error),
        "model": config.GROQ_MODEL_NAME,
    }
