"""Risk Scorer sub-agent invoking the versioned XGBoost model (never scores via LLM)."""

from __future__ import annotations

import json
from typing import Any

from finvigil.agents.budget import BudgetSnapshot
from finvigil.agents.state import InvestigationState
from finvigil.llm import count_tokens
from finvigil.observability import trace_agent_step
from finvigil.tools.risk_model import score_account_risk


def risk_scorer_node(state: InvestigationState) -> dict[str, Any]:
    """Run the deterministic XGBoost fraud/AML model for the target account."""
    account_id = state["account_id"]
    case_id = state["case_id"]
    db_url = state.get("db_url")
    as_of_step = state.get("as_of_step")
    budget = BudgetSnapshot.model_validate(state["budget"])

    with trace_agent_step(
        "risk_scorer",
        case_id=case_id,
        account_id=account_id,
        step_number=budget.steps_used + 1,
    ) as span:
        score_res = score_account_risk(
            account_id,
            as_of_step=as_of_step,
            db_url=db_url,
        )
        # Deterministic tool step: count exact BPE tokens of tool input/output serialized into state
        in_tok = count_tokens(json.dumps({"account_id": account_id, "as_of_step": as_of_step}))
        out_tok = count_tokens(score_res.model_dump_json())
        updated_budget = budget.record_step(in_tokens=in_tok, out_tokens=out_tok)

        span.set_attribute("finvigil.model_version", score_res.model_version)
        span.set_attribute("finvigil.risk_score", score_res.risk_score)
        span.set_attribute("finvigil.risk_band", score_res.risk_band)
        span.set_attribute("gen_ai.usage.input_tokens", in_tok)
        span.set_attribute("gen_ai.usage.output_tokens", out_tok)

        trace_events = list(state.get("trace_events", []))
        trace_events.append(
            {
                "step": updated_budget.steps_used,
                "agent": "risk_scorer",
                "model": score_res.model_version,
                "summary": (
                    f"XGBoost ({score_res.model_version}) scored account {account_id} at "
                    f"{score_res.risk_score:.4f} ({score_res.risk_band}, as_of_step={score_res.as_of_step})."
                ),
                "tokens": in_tok + out_tok,
                "cost_eur": updated_budget.cost_eur,
            }
        )

        return {
            "risk_assessment": score_res.model_dump(),
            "budget": updated_budget.model_dump(),
            "trace_events": trace_events,
        }
