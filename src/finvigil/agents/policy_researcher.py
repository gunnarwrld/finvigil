"""Policy Researcher sub-agent retrieving EU AMLR, PSD2, FATF, and EU AI Act guidance."""

from __future__ import annotations

import json
from typing import Any

from finvigil.agents.budget import BudgetSnapshot
from finvigil.agents.state import InvestigationState
from finvigil.llm import invoke_llm
from finvigil.observability import trace_agent_step
from finvigil.tools.policy_retriever import retrieve_aml_policies


def policy_researcher_node(state: InvestigationState) -> dict[str, Any]:
    """Retrieve relevant AML and PSD2 policy chunks with official CELEX/FATF references."""
    account_id = state["account_id"]
    case_id = state["case_id"]
    db_url = state.get("db_url")
    budget = BudgetSnapshot.model_validate(state["budget"])

    ledger_findings = state.get("ledger_findings") or {}
    risk_assessment = state.get("risk_assessment") or {}

    alert_rules = " ".join(
        str(a.get("trigger_rule", "")) + " " + str(a.get("alert_summary", ""))
        for a in ledger_findings.get("alerts", [])
    )
    drivers = " ".join(risk_assessment.get("top_risk_drivers", []))
    inj_flag = (
        "untrusted memo prompt injection override"
        if ledger_findings.get("prompt_injection_detected")
        else ""
    )
    rag_query = f"{alert_rules} {drivers} {inj_flag} freeze human approval".strip()

    with trace_agent_step(
        "policy_researcher",
        case_id=case_id,
        account_id=account_id,
        step_number=budget.steps_used + 1,
    ) as span:
        citations = retrieve_aml_policies(rag_query, top_k=3, db_url=db_url)
        default_summary = "Retrieved regulatory citations: " + ", ".join(
            f"{c.chunk_id} ({c.celex_or_source_ref})" for c in citations
        )
        llm_resp = invoke_llm(
            system_prompt="Select and summarize applicable EU AMLR, PSD2, FATF, and EU AI Act provisions.",
            user_prompt=json.dumps(
                {
                    "rag_query": rag_query,
                    "retrieved_chunks": [c.model_dump() for c in citations],
                }
            ),
            fallback_output=default_summary,
        )
        updated_budget = budget.record_step(
            in_tokens=llm_resp.input_tokens,
            out_tokens=llm_resp.output_tokens,
        )

        span.set_attribute("gen_ai.request.model", llm_resp.model_used)
        span.set_attribute("gen_ai.usage.input_tokens", llm_resp.input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", llm_resp.output_tokens)
        span.set_attribute("finvigil.policy_chunks_retrieved", len(citations))

        trace_events = list(state.get("trace_events", []))
        trace_events.append(
            {
                "step": updated_budget.steps_used,
                "agent": "policy_researcher",
                "model": llm_resp.model_used,
                "summary": default_summary,
                "tokens": llm_resp.input_tokens + llm_resp.output_tokens,
                "cost_eur": updated_budget.cost_eur,
            }
        )

        return {
            "policy_citations": [c.model_dump() for c in citations],
            "budget": updated_budget.model_dump(),
            "trace_events": trace_events,
        }
