"""FastAPI server with authenticated case-bound HITL approval and read-only default."""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from finvigil import __version__
from finvigil.agents.graph import (
    CaseHashMismatchError,
    FreezeDisabledError,
    get_checkpointed_state,
    resume_investigation_after_approval,
    run_investigation,
)
from finvigil.auth import verify_compliance_officer_token
from finvigil.config import settings
from finvigil.ledger_client import sync_from_java_ledger_postgres

app = FastAPI(
    title="FinVigil API",
    description=(
        "Autonomous Multi-Agent AML & Fraud Investigation Engine with sqlglot SQL Guard, "
        "deterministic XGBoost scoring, DeepSeek/Qwen LLM support, and authenticated HITL checkpoints."
    ),
    version=__version__,
)


class InvestigationRequest(BaseModel):
    """Request payload for initiating an account investigation."""

    query: str = Field(
        default="Why was account 4471 flagged, and should we freeze it?",
        min_length=3,
    )
    account_id: str | None = None
    case_id: str | None = None
    as_of_step: int | None = Field(default=None, ge=1)
    max_steps: int | None = Field(default=None, ge=1, le=25)
    max_tokens: int | None = Field(default=None, ge=100, le=100_000)
    max_cost_eur: float | None = Field(default=None, gt=0.0, le=5.0)


class ApprovalRequest(BaseModel):
    """Authenticated compliance officer decision bound to the exact CaseFile SHA-256."""

    decision: Literal["APPROVE", "REJECT"] = Field(
        ...,
        description="Explicit decision required (no default). Must be 'APPROVE' or 'REJECT'.",
    )
    case_file_sha256: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 digest of the reviewed CaseFile to prevent TOCTOU/tampering.",
    )
    notes: str = Field(default="Verified transaction citations and EU AMLR/PSD2 triggers.")
    idempotency_key: str | None = None


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Return service health, active risk model version, LLM provider, and freeze mode."""
    return {
        "status": "ok",
        "service": settings.service_name,
        "version": __version__,
        "risk_model_version": settings.risk_model_version,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "freeze_actions_enabled": settings.enable_freeze_actions,
    }


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    """Expose Prometheus observability metrics."""
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/v1/investigations")
def create_investigation(req: InvestigationRequest) -> dict[str, Any]:
    """Run a read-only investigation and return the cited CaseFile, SHA-256, and budget telemetry."""
    try:
        state = run_investigation(
            req.query,
            account_id=req.account_id,
            case_id=req.case_id,
            as_of_step=req.as_of_step,
            max_steps=req.max_steps,
            max_tokens=req.max_tokens,
            max_cost_eur=req.max_cost_eur,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "case_id": state["case_id"],
        "account_id": state["account_id"],
        "status": state["status"],
        "case_file": state["case_file"],
        "budget": state["budget"],
        "trace_events": state["trace_events"],
    }


@app.post("/v1/investigations/stream")
def stream_investigation(req: InvestigationRequest) -> StreamingResponse:
    """Stream each agent step and the final CaseFile via Server-Sent Events (SSE)."""
    try:
        state = run_investigation(
            req.query,
            account_id=req.account_id,
            case_id=req.case_id,
            as_of_step=req.as_of_step,
            max_steps=req.max_steps,
            max_tokens=req.max_tokens,
            max_cost_eur=req.max_cost_eur,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _event_generator() -> Generator[str, None, None]:
        for evt in state.get("trace_events", []):
            yield f"event: agent_step\ndata: {json.dumps(evt)}\n\n"
        final_payload = {
            "case_id": state["case_id"],
            "account_id": state["account_id"],
            "status": state["status"],
            "case_file": state["case_file"],
            "budget": state["budget"],
        }
        yield f"event: case_file\ndata: {json.dumps(final_payload)}\n\n"

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


@app.get("/v1/investigations/{case_id}")
def get_investigation(case_id: str) -> dict[str, Any]:
    """Load a checkpointed investigation by case_id from the SQLite LangGraph checkpointer."""
    state = get_checkpointed_state(case_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")
    return dict(state)


@app.post("/v1/investigations/{case_id}/approve")
def approve_or_reject_investigation(
    case_id: str,
    req: ApprovalRequest,
    authenticated_officer: str = Depends(verify_compliance_officer_token),
    idempotency_key_header: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict[str, Any]:
    """Resume a paused investigation with authenticated officer identity and CaseFile hash verification."""
    effective_idem_key = idempotency_key_header or req.idempotency_key
    approved = req.decision == "APPROVE"
    try:
        resumed = resume_investigation_after_approval(
            case_id,
            approved=approved,
            authenticated_reviewer=authenticated_officer,
            case_file_sha256=req.case_file_sha256,
            notes=req.notes,
            idempotency_key=effective_idem_key,
        )
    except FreezeDisabledError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except CaseHashMismatchError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "case_id": resumed["case_id"],
        "account_id": resumed["account_id"],
        "status": resumed["status"],
        "human_decision": resumed["human_decision"],
        "action_receipt": resumed.get("action_receipt"),
        "trace_events": resumed["trace_events"],
    }


@app.post("/v1/ledger/sync-java")
def sync_java_ledger_endpoint(
    authenticated_officer: str = Depends(verify_compliance_officer_token),
) -> dict[str, Any]:
    """Read-only sync of accounts, postings, and payments from Java 21 payments-lab PostgreSQL 18."""
    counts = sync_from_java_ledger_postgres()
    return {
        "status": "synced",
        "synced_by": authenticated_officer,
        "java_ledger_database_url": settings.java_ledger_database_url.split("@")[-1],
        **counts,
    }
