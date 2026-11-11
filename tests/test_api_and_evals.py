"""Tests for FastAPI endpoints, Bearer Auth, Case-Bound Hash, SSE streaming, and 60-Case Eval Gate."""

from __future__ import annotations

import pytest
from evals.runner import run_eval_suite
from fastapi.testclient import TestClient

from finvigil.api.app import app
from finvigil.config import settings

client = TestClient(app)
VALID_BEARER = {"Authorization": "Bearer tok-nordea-aml-lead-991"}


def test_health_and_prometheus_metrics_endpoints() -> None:
    """Verify /health and /metrics endpoints return valid status and telemetry."""
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "ok"

    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    assert "finvigil_agent_steps_total" in metrics_resp.text


def test_approve_endpoint_enforces_bearer_auth_and_case_hash_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify /approve rejects unauthenticated calls, enforces read-only default, and checks SHA-256."""
    resp = client.post(
        "/v1/investigations",
        json={
            "query": "Check account 5590 for unauthorized drain",
            "account_id": "5590",
            "case_id": "API-CASE-5590",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "AWAITING_HUMAN_APPROVAL"
    case_hash = body["case_file"]["case_file_sha256"]

    # 1. Unauthenticated request -> 401 Unauthorized
    unauth = client.post(
        "/v1/investigations/API-CASE-5590/approve",
        json={"decision": "APPROVE", "case_file_sha256": case_hash},
    )
    assert unauth.status_code == 401

    # 2. V1 Read-only mode -> 403 Forbidden when freeze is disabled
    monkeypatch.setattr(settings, "enable_freeze_actions", False)
    readonly_block = client.post(
        "/v1/investigations/API-CASE-5590/approve",
        headers=VALID_BEARER,
        json={"decision": "APPROVE", "case_file_sha256": case_hash},
    )
    assert readonly_block.status_code == 403

    # 3. Enable freeze actions; wrong case_file_sha256 -> 409 Conflict
    monkeypatch.setattr(settings, "enable_freeze_actions", True)
    bad_hash = client.post(
        "/v1/investigations/API-CASE-5590/approve",
        headers=VALID_BEARER,
        json={"decision": "APPROVE", "case_file_sha256": "f" * 64},
    )
    assert bad_hash.status_code == 409

    # 4. Valid Bearer token + matching case_file_sha256 -> 200 OK (reviewer derived from token)
    approve_1 = client.post(
        "/v1/investigations/API-CASE-5590/approve",
        headers={**VALID_BEARER, "Idempotency-Key": "api-idem-5590-1"},
        json={"decision": "APPROVE", "case_file_sha256": case_hash, "notes": "Approve freeze."},
    )
    assert approve_1.status_code == 200
    assert approve_1.json()["status"] == "APPROVED_AND_FROZEN"
    assert (
        approve_1.json()["action_receipt"]["approved_by"]
        == "officer:anna.lindqvist@nordea.example"
    )
    assert approve_1.json()["action_receipt"]["was_replayed"] is False

    # 5. Idempotent retry -> was_replayed=True
    approve_2 = client.post(
        "/v1/investigations/API-CASE-5590/approve",
        headers={**VALID_BEARER, "Idempotency-Key": "api-idem-5590-1"},
        json={"decision": "APPROVE", "case_file_sha256": case_hash, "notes": "Retry."},
    )
    assert approve_2.status_code == 200
    assert approve_2.json()["action_receipt"]["was_replayed"] is True


def test_sse_streaming_endpoint() -> None:
    """Verify POST /v1/investigations/stream emits agent_step and case_file SSE events."""
    with client.stream(
        "POST",
        "/v1/investigations/stream",
        json={"query": "Review payroll account 2048", "account_id": "2048"},
    ) as response:
        assert response.status_code == 200
        raw_stream = "".join(response.iter_text())
        assert "event: agent_step" in raw_stream
        assert "event: case_file" in raw_stream
        assert "DISMISS_ALERT" in raw_stream


def test_60_case_holdout_eval_suite_passes_ci_gate(
    seeded_environment: str,
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Run the 60-case evaluation benchmark (curated + time/typology holdouts) and assert >= 90% pass."""
    summary = run_eval_suite(
        db_url=seeded_environment,
        checkpoint_file=tmp_path / "eval_cp.sqlite",  # type: ignore[operator]
        include_60_case_holdout=True,
    )
    assert summary["total_cases"] == 60
    assert summary["pass_rate"] >= 0.90
    assert summary["precision"] >= 0.90
    assert summary["recall"] >= 0.85
    assert summary["avg_cost_eur_per_investigation"] <= 0.15
