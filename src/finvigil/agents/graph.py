"""LangGraph multi-agent workflow with SQLite BaseCheckpointSaver and native HITL resume."""

from __future__ import annotations

import hmac
import re
import uuid
from pathlib import Path
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from finvigil.agents.budget import BudgetSnapshot
from finvigil.agents.case_writer import case_writer_node
from finvigil.agents.checkpointer import SqliteLangGraphSaver
from finvigil.agents.ledger_analyst import ledger_analyst_node
from finvigil.agents.policy_researcher import policy_researcher_node
from finvigil.agents.risk_scorer import risk_scorer_node
from finvigil.agents.state import InvestigationState
from finvigil.agents.supervisor import supervisor_node
from finvigil.config import settings
from finvigil.observability import (
    INVESTIGATION_COST_HISTOGRAM,
    INVESTIGATIONS_TOTAL,
    trace_agent_step,
)
from finvigil.tools.ledger_actions import execute_idempotent_freeze


class CaseHashMismatchError(ValueError):
    """Raised when an approval request's case_file_sha256 does not match the checkpointed CaseFile."""


class FreezeDisabledError(PermissionError):
    """Raised when freeze execution is attempted while enable_freeze_actions is False."""


def human_approval_node(state: InvestigationState) -> dict[str, Any]:
    """Process authenticated human compliance officer decision inside LangGraph."""
    decision = state.get("human_decision")
    if not decision:
        return {"status": "AWAITING_HUMAN_APPROVAL"}

    approved = bool(decision.get("approved", False))
    reviewer = str(decision.get("reviewer", "authenticated-officer"))
    trace_events = list(state.get("trace_events", []))

    if approved:
        trace_events.append(
            {
                "step": state["budget"]["steps_used"],
                "agent": "human_approval",
                "summary": (
                    f"Authenticated approval GRANTED by {reviewer} to freeze account "
                    f"{state['account_id']} (bound to case_file_sha256={decision.get('case_file_sha256', '')[:12]}...)."
                ),
                "cost_eur": state["budget"]["cost_eur"],
            }
        )
        return {"status": "IN_PROGRESS", "trace_events": trace_events}

    trace_events.append(
        {
            "step": state["budget"]["steps_used"],
            "agent": "human_approval",
            "summary": f"Authenticated decision REJECTED by {reviewer}; account {state['account_id']} remains ACTIVE.",
            "cost_eur": state["budget"]["cost_eur"],
        }
    )
    return {"status": "REJECTED_BY_HUMAN", "trace_events": trace_events}


def execute_freeze_node(state: InvestigationState) -> dict[str, Any]:
    """Call the idempotent ledger action tool to freeze the account after approval."""
    case_id = state["case_id"]
    account_id = state["account_id"]
    db_url = state.get("db_url")
    decision = state.get("human_decision") or {}
    case_file = state.get("case_file") or {}
    reviewer = str(decision.get("reviewer", "authenticated-officer"))
    notes = str(decision.get("notes", "Approved following FinVigil AML investigation."))
    idem_key = decision.get("idempotency_key")
    case_hash = str(decision.get("case_file_sha256") or case_file.get("case_file_sha256", ""))

    with trace_agent_step(
        "ledger_action_api",
        case_id=case_id,
        account_id=account_id,
        step_number=state["budget"]["steps_used"] + 1,
    ) as span:
        receipt = execute_idempotent_freeze(
            case_id=case_id,
            account_id=account_id,
            approved_by=reviewer,
            case_file_sha256=case_hash,
            reason=notes,
            idempotency_key=idem_key,
            db_url=db_url,
        )
        span.set_attribute("finvigil.idempotency_key", receipt.idempotency_key)
        span.set_attribute("finvigil.was_replayed", receipt.was_replayed)

        trace_events = list(state.get("trace_events", []))
        trace_events.append(
            {
                "step": state["budget"]["steps_used"],
                "agent": "ledger_action_api",
                "summary": (
                    f"Executed idempotent FREEZE_ACCOUNT on {account_id} "
                    f"(key={receipt.idempotency_key}, replayed={receipt.was_replayed})."
                ),
                "cost_eur": state["budget"]["cost_eur"],
            }
        )

        return {
            "action_receipt": receipt.model_dump(),
            "status": "APPROVED_AND_FROZEN",
            "trace_events": trace_events,
        }


def _route_from_supervisor(
    state: InvestigationState,
) -> Literal[
    "ledger_analyst",
    "risk_scorer",
    "policy_researcher",
    "case_writer",
    "human_approval",
    "__end__",
]:
    next_agent = state.get("next_agent", "case_writer")
    if next_agent == "ledger_analyst":
        return "ledger_analyst"
    if next_agent == "risk_scorer":
        return "risk_scorer"
    if next_agent == "policy_researcher":
        return "policy_researcher"
    if next_agent == "case_writer":
        return "case_writer"

    case_file = state.get("case_file") or {}
    if case_file.get("recommended_action") == "FREEZE_ACCOUNT":
        return "human_approval"
    return "__end__"


def _route_after_human_approval(
    state: InvestigationState,
) -> Literal["execute_freeze", "__end__"]:
    decision = state.get("human_decision") or {}
    if decision.get("approved"):
        return "execute_freeze"
    return "__end__"


def build_investigation_graph(checkpointer: BaseCheckpointSaver[Any] | None = None) -> Any:
    """Build and compile the FinVigil LangGraph workflow with SQLite-backed HITL interrupt."""
    workflow = StateGraph(InvestigationState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("ledger_analyst", ledger_analyst_node)
    workflow.add_node("risk_scorer", risk_scorer_node)
    workflow.add_node("policy_researcher", policy_researcher_node)
    workflow.add_node("case_writer", case_writer_node)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("execute_freeze", execute_freeze_node)

    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {
            "ledger_analyst": "ledger_analyst",
            "risk_scorer": "risk_scorer",
            "policy_researcher": "policy_researcher",
            "case_writer": "case_writer",
            "human_approval": "human_approval",
            "__end__": END,
        },
    )
    workflow.add_edge("ledger_analyst", "supervisor")
    workflow.add_edge("risk_scorer", "supervisor")
    workflow.add_edge("policy_researcher", "supervisor")
    workflow.add_edge("case_writer", "supervisor")
    workflow.add_conditional_edges(
        "human_approval",
        _route_after_human_approval,
        {
            "execute_freeze": "execute_freeze",
            "__end__": END,
        },
    )
    workflow.add_edge("execute_freeze", END)

    saver = checkpointer or SqliteLangGraphSaver()
    return workflow.compile(checkpointer=saver, interrupt_before=["human_approval"])


def extract_account_id_from_query(query: str, explicit_account_id: str | None = None) -> str:
    """Extract account ID or UUID from natural-language analyst query if not explicitly passed."""
    if explicit_account_id:
        return explicit_account_id.strip()
    match = re.search(r"\b(?:account|acct|#)\s*([A-Za-z0-9_-]{3,64})\b", query, re.IGNORECASE)
    if match:
        return match.group(1)
    uuid_match = re.search(
        r"\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b",
        query,
        re.IGNORECASE,
    )
    if uuid_match:
        return uuid_match.group(1)
    digits = re.search(r"\b(\d{4,10})\b", query)
    if digits:
        return digits.group(1)
    raise ValueError("Could not identify target account_id from query.")


def get_checkpointed_state(
    case_id: str,
    *,
    checkpoint_file: Path | None = None,
) -> InvestigationState | None:
    """Retrieve a thread's state directly from the LangGraph SQLite checkpointer."""
    saver = SqliteLangGraphSaver(checkpoint_file)
    graph = build_investigation_graph(checkpointer=saver)
    config: RunnableConfig = {"configurable": {"thread_id": case_id}}
    snapshot = graph.get_state(config)
    if snapshot is None or not snapshot.values:
        return None
    typed_vals: InvestigationState = dict(snapshot.values)  # type: ignore[assignment]
    return typed_vals


def run_investigation(
    analyst_query: str,
    *,
    account_id: str | None = None,
    case_id: str | None = None,
    as_of_step: int | None = None,
    db_url: str | None = None,
    max_steps: int | None = None,
    max_tokens: int | None = None,
    max_cost_eur: float | None = None,
    checkpoint_file: Path | None = None,
) -> InvestigationState:
    """Run an investigation up to completion or the Human Approval LangGraph interrupt."""
    resolved_account_id = extract_account_id_from_query(analyst_query, account_id)
    resolved_case_id = case_id or f"CASE-{resolved_account_id}-{uuid.uuid4().hex[:6].upper()}"

    budget = BudgetSnapshot(
        max_steps=max_steps or settings.max_steps_per_request,
        max_tokens=max_tokens or settings.max_tokens_per_request,
        max_cost_eur=max_cost_eur or settings.max_cost_eur_per_request,
    )

    initial_state: InvestigationState = {
        "case_id": resolved_case_id,
        "account_id": resolved_account_id,
        "analyst_query": analyst_query,
        "as_of_step": as_of_step,
        "db_url": db_url,
        "budget": budget.model_dump(),
        "ledger_findings": None,
        "risk_assessment": None,
        "policy_citations": None,
        "case_file": None,
        "next_agent": "ledger_analyst",
        "status": "IN_PROGRESS",
        "human_decision": None,
        "action_receipt": None,
        "trace_events": [],
    }

    saver = SqliteLangGraphSaver(checkpoint_file)
    saver.delete_thread(resolved_case_id)
    graph = build_investigation_graph(checkpointer=saver)
    config: RunnableConfig = {"configurable": {"thread_id": resolved_case_id}}
    final_state: InvestigationState = graph.invoke(initial_state, config=config)

    case_file = final_state.get("case_file") or {}
    rec_action = str(case_file.get("recommended_action", "UNKNOWN"))
    status = str(final_state.get("status", "COMPLETED"))
    INVESTIGATIONS_TOTAL.labels(recommended_action=rec_action, status=status).inc()
    INVESTIGATION_COST_HISTOGRAM.observe(float(final_state["budget"]["cost_eur"]))

    return final_state


def resume_investigation_after_approval(
    case_id: str,
    *,
    approved: bool,
    authenticated_reviewer: str,
    case_file_sha256: str,
    notes: str = "Reviewed case file and citations.",
    idempotency_key: str | None = None,
    db_url: str | None = None,
    checkpoint_file: Path | None = None,
) -> InvestigationState:
    """Resume a paused investigation natively via LangGraph checkpointer (survives OS restarts)."""
    if approved and not settings.enable_freeze_actions:
        raise FreezeDisabledError(
            "Account freeze execution is disabled in v1 read-only mode "
            "(set FINVIGIL_ENABLE_FREEZE_ACTIONS=true to enable)."
        )

    saver = SqliteLangGraphSaver(checkpoint_file)
    graph = build_investigation_graph(checkpointer=saver)
    config: RunnableConfig = {"configurable": {"thread_id": case_id}}

    snapshot = graph.get_state(config)
    if snapshot is None or not snapshot.values:
        raise KeyError(f"No saved LangGraph checkpoint found for case_id '{case_id}'.")

    saved_case_file = snapshot.values.get("case_file") or {}
    expected_hash = str(saved_case_file.get("case_file_sha256", ""))
    if not expected_hash or not hmac.compare_digest(
        case_file_sha256.strip().encode("utf-8"),
        expected_hash.encode("utf-8"),
    ):
        raise CaseHashMismatchError(
            f"Case file SHA-256 mismatch for case '{case_id}': approval must bind to exact reviewed CaseFile."
        )

    state_patch: dict[str, Any] = {
        "human_decision": {
            "approved": approved,
            "reviewer": authenticated_reviewer,
            "case_file_sha256": expected_hash,
            "notes": notes,
            "idempotency_key": idempotency_key,
        }
    }
    if db_url is not None:
        state_patch["db_url"] = db_url

    merged_for_approval: InvestigationState = {**snapshot.values, **state_patch}  # type: ignore[typeddict-item]
    approval_updates = human_approval_node(merged_for_approval)

    # Native LangGraph state update as_node="human_approval" + graph continuation to execute_freeze
    graph.update_state(config, {**state_patch, **approval_updates}, as_node="human_approval")
    resumed_state: InvestigationState = graph.invoke(None, config=config)
    return resumed_state
