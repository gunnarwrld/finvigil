"""Supervisor agent planning and routing sub-agents under strict step/token/EUR budgets."""

from __future__ import annotations

import json
from typing import Any

from finvigil.agents.budget import BudgetSnapshot
from finvigil.agents.state import InvestigationState
from finvigil.llm import count_tokens
from finvigil.observability import trace_agent_step


def supervisor_node(state: InvestigationState) -> dict[str, Any]:
    """Route to the next sub-agent or force Case Writer synthesis if budget is exhausted."""
    if state.get("case_file") is not None:
        return {"next_agent": "route_post_case"}

    case_id = state["case_id"]
    account_id = state["account_id"]
    budget = BudgetSnapshot.model_validate(state["budget"])

    with trace_agent_step(
        "supervisor",
        case_id=case_id,
        account_id=account_id,
        step_number=budget.steps_used + 1,
    ) as span:
        routing_state_summary = json.dumps(
            {
                "case_id": case_id,
                "account_id": account_id,
                "has_ledger_findings": state.get("ledger_findings") is not None,
                "has_risk_assessment": state.get("risk_assessment") is not None,
                "has_policy_citations": state.get("policy_citations") is not None,
                "steps_used": budget.steps_used,
            }
        )
        in_tok = count_tokens(routing_state_summary)

        # Essential steps (ledger_analyst and risk_scorer) run first; if budget is tight or
        # exhausted afterwards, skip optional RAG (policy_researcher) first!
        projected_budget = budget.record_step(in_tokens=in_tok, out_tokens=12)
        if state.get("ledger_findings") is None:
            next_agent = "ledger_analyst"
        elif state.get("risk_assessment") is None:
            next_agent = "risk_scorer"
        elif state.get("policy_citations") is None and not projected_budget.is_exhausted:
            next_agent = "policy_researcher"
        else:
            next_agent = "case_writer"

        out_tok = count_tokens(next_agent)
        updated_budget = budget.record_step(in_tokens=in_tok, out_tokens=out_tok)

        span.set_attribute("finvigil.budget.steps_used", updated_budget.steps_used)
        span.set_attribute("finvigil.budget.cost_eur", updated_budget.cost_eur)
        span.set_attribute("finvigil.budget.is_exhausted", updated_budget.is_exhausted)
        span.set_attribute("finvigil.supervisor.next_agent", next_agent)

        trace_events = list(state.get("trace_events", []))
        trace_events.append(
            {
                "step": updated_budget.steps_used,
                "agent": "supervisor",
                "summary": (
                    f"Routing to '{next_agent}' "
                    f"(steps={updated_budget.steps_used}/{updated_budget.max_steps}, "
                    f"tokens={updated_budget.tokens_used}/{updated_budget.max_tokens}, "
                    f"cost=€{updated_budget.cost_eur:.6f}/€{updated_budget.max_cost_eur:.4f})."
                ),
                "tokens": in_tok + out_tok,
                "cost_eur": updated_budget.cost_eur,
            }
        )

        return {
            "next_agent": next_agent,
            "budget": updated_budget.model_dump(),
            "trace_events": trace_events,
        }
