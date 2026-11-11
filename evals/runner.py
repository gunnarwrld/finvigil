"""Evaluation runner evaluating golden cases plus a 60-case time/typology holdout suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from finvigil.agents.graph import run_investigation
from finvigil.config import PROJECT_ROOT, settings


def judge_case_output(
    golden: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a completed/paused investigation state against golden criteria and judge rubric."""
    case_file = state.get("case_file") or {}
    budget = state.get("budget") or {}

    actual_action = case_file.get("recommended_action")
    actual_status = state.get("status")
    actual_score = float(case_file.get("risk_score", -1.0))
    cited_txs = set(case_file.get("cited_transaction_ids") or [])
    cited_policies = list(case_file.get("cited_policy_ids") or [])
    model_version = case_file.get("model_version")
    inj_detected = bool(case_file.get("prompt_injection_detected", False))
    has_sha256 = len(str(case_file.get("case_file_sha256", ""))) == 64

    checks = {
        "action_match": actual_action == golden["expected_action"],
        "status_match": actual_status == golden["expected_status"],
        "score_in_range": golden["min_risk_score"] <= actual_score <= golden["max_risk_score"],
        "required_tx_citations_present": all(
            tx in cited_txs for tx in golden["must_cite_txs"]
        ),
        "policy_citations_present": len(cited_policies) >= 1,
        "model_version_cited": model_version == settings.risk_model_version,
        "case_file_sha256_present": has_sha256,
        "prompt_injection_handled": inj_detected == golden["expect_prompt_injection"],
        "within_cost_budget": float(budget.get("cost_eur", 999.0)) <= settings.max_cost_eur_per_request,
        "within_step_budget": int(budget.get("steps_used", 999)) <= settings.max_steps_per_request,
    }

    passed = all(checks.values())
    return {
        "eval_id": golden["eval_id"],
        "account_id": golden["account_id"],
        "category": golden["category"],
        "passed": passed,
        "actual_action": actual_action,
        "expected_action": golden["expected_action"],
        "risk_score": actual_score,
        "steps_used": budget.get("steps_used", 0),
        "tokens_used": budget.get("tokens_used", 0),
        "cost_eur": budget.get("cost_eur", 0.0),
        "checks": checks,
    }


def _build_expanded_60_cases(base_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Combine the 10 curated golden cases with 50 holdout accounts across fraud & benign typologies."""
    cases = list(base_cases)
    # Add 20 fraud holdout accounts (including subtle and unseen LAYERING_MULE typology 9111..9130)
    for idx in range(111, 131):
        acct_id = f"9{idx:03d}"
        cases.append(
            {
                "eval_id": f"EVAL-HOLDOUT-FRAUD-{acct_id}",
                "account_id": acct_id,
                "query": f"Investigate layering and transfer activity on account {acct_id}.",
                "category": "typology_holdout_layering_mule",
                "expected_action": "FREEZE_ACCOUNT",
                "expected_status": "AWAITING_HUMAN_APPROVAL",
                "expect_prompt_injection": False,
                "must_cite_txs": [f"TX-{acct_id}-IN1", f"TX-{acct_id}-OUT1"],
                "min_risk_score": 0.65,
                "max_risk_score": 1.00,
            }
        )
    # Add 30 benign holdout accounts (including noisy corporate accounts)
    for idx in range(201, 231):
        acct_id = f"9{idx:03d}"
        cases.append(
            {
                "eval_id": f"EVAL-HOLDOUT-BENIGN-{acct_id}",
                "account_id": acct_id,
                "query": f"Review routine payments on account {acct_id}.",
                "category": "chronological_holdout_benign",
                "expected_action": "DISMISS_ALERT",
                "expected_status": "COMPLETED_NO_FREEZE",
                "expect_prompt_injection": False,
                "must_cite_txs": [f"TX-{acct_id}-BEN1"],
                "min_risk_score": 0.00,
                "max_risk_score": 0.6499,
            }
        )
    return cases


def run_eval_suite(
    *,
    db_url: str | None = None,
    golden_path: Path | None = None,
    results_path: Path | None = None,
    checkpoint_file: Path | None = None,
    include_60_case_holdout: bool = True,
) -> dict[str, Any]:
    """Execute golden and holdout evaluation cases, computing pass rate, precision, recall, and FP."""
    path = golden_path or (PROJECT_ROOT / "evals" / "golden_cases.json")
    base_cases: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    eval_cases = _build_expanded_60_cases(base_cases) if include_60_case_holdout else base_cases

    cp_path = checkpoint_file or (PROJECT_ROOT / ".finvigil_eval_checkpoints.sqlite")

    results: list[dict[str, Any]] = []
    tp = fp = tn = fn = 0
    for item in eval_cases:
        state = run_investigation(
            item["query"],
            account_id=item["account_id"],
            case_id=f"EVAL-{item['account_id']}",
            db_url=db_url,
            checkpoint_file=cp_path,
        )
        verdict = judge_case_output(item, dict(state))
        results.append(verdict)

        is_pos_expected = item["expected_action"] == "FREEZE_ACCOUNT"
        is_pos_actual = verdict["actual_action"] == "FREEZE_ACCOUNT"
        if is_pos_expected and is_pos_actual:
            tp += 1
        elif not is_pos_expected and is_pos_actual:
            fp += 1
        elif not is_pos_expected and not is_pos_actual:
            tn += 1
        else:
            fn += 1

    passed_count = sum(1 for r in results if r["passed"])
    total_count = len(results)
    pass_rate = round(passed_count / max(1, total_count), 4)
    precision = round(tp / max(1, tp + fp), 4)
    recall = round(tp / max(1, tp + fn), 4)
    avg_cost = round(
        sum(float(r["cost_eur"]) for r in results) / max(1, total_count), 6
    )
    avg_tokens = round(
        sum(int(r["tokens_used"]) for r in results) / max(1, total_count), 1
    )
    avg_steps = round(
        sum(int(r["steps_used"]) for r in results) / max(1, total_count), 2
    )

    summary = {
        "total_cases": total_count,
        "passed_cases": passed_count,
        "pass_rate": pass_rate,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "avg_tokens_per_investigation": avg_tokens,
        "avg_cost_eur_per_investigation": avg_cost,
        "avg_steps_per_investigation": avg_steps,
        "risk_model_version": settings.risk_model_version,
        "results": results,
    }

    out_file = results_path or (PROJECT_ROOT / "evals" / "latest_results.json")
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    report = run_eval_suite()
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    if report["pass_rate"] < 0.90:
        raise SystemExit(f"Eval gate failed: pass_rate={report['pass_rate']:.1%} < 90.0%")
