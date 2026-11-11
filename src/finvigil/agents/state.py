"""Pydantic schemas and LangGraph state definitions for FinVigil investigations."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, model_validator


class CaseFile(BaseModel):
    """Structured case file produced by the Case Writer agent with mandatory citations and SHA-256."""

    case_id: str
    account_id: str
    holder_name: str
    executive_summary: str
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_band: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    model_version: str = Field(min_length=3)
    cited_transaction_ids: list[str] = Field(min_length=1)
    cited_policy_ids: list[str] = Field(min_length=1)
    key_findings: list[str] = Field(min_length=1)
    prompt_injection_detected: bool = False
    recommended_action: Literal["FREEZE_ACCOUNT", "MONITOR_ONLY", "DISMISS_ALERT"]
    requires_human_approval: bool
    case_file_sha256: str = Field(min_length=64, max_length=64)
    llm_model_used: str = "offline"
    budget_exhausted_warning: str | None = None

    @model_validator(mode="after")
    def _validate_freeze_consistency(self) -> CaseFile:
        """Ensure FREEZE_ACCOUNT always requires human approval."""
        if self.recommended_action == "FREEZE_ACCOUNT" and not self.requires_human_approval:
            raise ValueError("FREEZE_ACCOUNT recommendation must set requires_human_approval=True.")
        return self


class InvestigationState(TypedDict, total=False):
    """LangGraph shared state across Supervisor, Sub-Agents, and Human Approval."""

    case_id: str
    account_id: str
    analyst_query: str
    as_of_step: int | None
    db_url: str | None
    budget: dict[str, Any]
    ledger_findings: dict[str, Any] | None
    risk_assessment: dict[str, Any] | None
    policy_citations: list[dict[str, Any]] | None
    case_file: dict[str, Any] | None
    next_agent: str
    status: Literal[
        "IN_PROGRESS",
        "AWAITING_HUMAN_APPROVAL",
        "APPROVED_AND_FROZEN",
        "REJECTED_BY_HUMAN",
        "COMPLETED_NO_FREEZE",
    ]
    human_decision: dict[str, Any] | None
    action_receipt: dict[str, Any] | None
    trace_events: list[dict[str, Any]]
