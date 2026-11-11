"""Integration tests for Multi-Agent Graph, Subprocess Restart, Point-in-Time Features, and DeepSeek/Qwen LLM."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from ml.features import extract_account_features
from sqlalchemy import func, select

from finvigil.agents.graph import (
    CaseHashMismatchError,
    FreezeDisabledError,
    resume_investigation_after_approval,
    run_investigation,
)
from finvigil.config import settings
from finvigil.db.models import Account, AuditActionLog
from finvigil.db.session import session_scope
from finvigil.ledger_client import (
    ExternalLedgerAccountView,
    JavaLedgerEntryItem,
    sync_account_view_into_finvigil,
)
from finvigil.llm import strip_deepseek_r1_think_tags
from finvigil.tools.risk_model import score_account_risk


def test_point_in_time_feature_extraction_prevents_future_leakage(seeded_environment: str) -> None:
    """Verify extract_account_features with as_of_step excludes future transactions."""
    with session_scope(seeded_environment, readonly=True) as session:
        early_vec = extract_account_features(session, "4471", as_of_step=112)
        full_vec = extract_account_features(session, "4471", as_of_step=114)

    assert early_vec.cited_tx_ids == ["TX-4471-001", "TX-4471-002", "TX-4471-003"]
    assert "TX-4471-005" not in early_vec.cited_tx_ids
    assert len(full_vec.cited_tx_ids) == 5


def test_deepseek_r1_think_tag_extraction() -> None:
    """Verify DeepSeek-R1 / QwQ <think>...</think> blocks are cleanly separated from output."""
    raw = "<think>\nChecking structuring threshold under Art. 19...\n</think>\nAccount 4471 shows structuring."
    cleaned, reasoning = strip_deepseek_r1_think_tags(raw)
    assert cleaned == "Account 4471 shows structuring."
    assert reasoning == "Checking structuring threshold under Art. 19..."


def test_v1_readonly_default_blocks_freeze_until_enabled(
    seeded_environment: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify v1 defaults to read-only (enable_freeze_actions=False) and blocks freeze calls."""
    monkeypatch.setattr(settings, "enable_freeze_actions", False)
    cp_file = tmp_path / "readonly_cp.sqlite"
    state = run_investigation(
        "Why was account 4471 flagged?",
        case_id="CASE-4471-RO",
        db_url=seeded_environment,
        checkpoint_file=cp_file,
    )
    assert state["case_file"] is not None
    case_hash = state["case_file"]["case_file_sha256"]

    with pytest.raises(FreezeDisabledError):
        resume_investigation_after_approval(
            "CASE-4471-RO",
            approved=True,
            authenticated_reviewer="officer:anna.lindqvist@nordea.example",
            case_file_sha256=case_hash,
            db_url=seeded_environment,
            checkpoint_file=cp_file,
        )


def test_real_subprocess_restart_checkpoint_durability_and_idempotency(
    seeded_environment: str,
    tmp_path: Path,
) -> None:
    """Spawn Process 1 to pause at HITL interrupt, exit OS process, and spawn Process 2 to resume."""
    cp_file = tmp_path / "subprocess_cp.sqlite"
    case_id = "CASE-4471-SUBPROC"

    # Process 1: Start investigation and pause at interrupt_before=["human_approval"]
    proc1_code = f"""
import json
from pathlib import Path
from finvigil.agents.graph import run_investigation
state = run_investigation(
    "Why was account 4471 flagged, and should we freeze it?",
    case_id={case_id!r},
    db_url={seeded_environment!r},
    checkpoint_file=Path({str(cp_file)!r}),
)
print(json.dumps({{"status": state["status"], "sha256": state["case_file"]["case_file_sha256"]}}))
"""
    res1 = subprocess.run(  # noqa: S603
        [sys.executable, "-c", proc1_code],
        capture_output=True,
        text=True,
        check=True,
    )
    out1 = json.loads(res1.stdout.strip().splitlines()[-1])
    assert out1["status"] == "AWAITING_HUMAN_APPROVAL"
    case_hash = out1["sha256"]

    # Account 4471 must still be ACTIVE before Process 2 approves
    with session_scope(seeded_environment, readonly=True) as session:
        acct_before = session.get(Account, "4471")
        assert acct_before is not None
        assert acct_before.status == "ACTIVE"

    # Process 2: Separate OS process cold-loads SQLite checkpoint and resumes LangGraph
    proc2_code = f"""
import json
from pathlib import Path
from finvigil.config import settings
from finvigil.agents.graph import resume_investigation_after_approval
settings.enable_freeze_actions = True
resumed = resume_investigation_after_approval(
    {case_id!r},
    approved=True,
    authenticated_reviewer="officer:anna.lindqvist@nordea.example",
    case_file_sha256={case_hash!r},
    idempotency_key="idem-subproc-4471",
    db_url={seeded_environment!r},
    checkpoint_file=Path({str(cp_file)!r}),
)
print(json.dumps({{"status": resumed["status"], "receipt": resumed["action_receipt"]}}))
"""
    res2 = subprocess.run(  # noqa: S603
        [sys.executable, "-c", proc2_code],
        capture_output=True,
        text=True,
        check=True,
    )
    out2 = json.loads(res2.stdout.strip().splitlines()[-1])
    assert out2["status"] == "APPROVED_AND_FROZEN"
    assert out2["receipt"]["was_replayed"] is False
    assert out2["receipt"]["approved_by"] == "officer:anna.lindqvist@nordea.example"

    # Verify tamper detection on mismatched SHA-256 hash
    settings.enable_freeze_actions = True
    with pytest.raises(CaseHashMismatchError):
        resume_investigation_after_approval(
            case_id,
            approved=True,
            authenticated_reviewer="officer:anna.lindqvist@nordea.example",
            case_file_sha256="0" * 64,
            db_url=seeded_environment,
            checkpoint_file=cp_file,
        )

    # Verify idempotent replay with valid hash
    resumed_replay = resume_investigation_after_approval(
        case_id,
        approved=True,
        authenticated_reviewer="officer:anna.lindqvist@nordea.example",
        case_file_sha256=case_hash,
        idempotency_key="idem-subproc-4471",
        db_url=seeded_environment,
        checkpoint_file=cp_file,
    )
    assert resumed_replay["action_receipt"] is not None
    assert resumed_replay["action_receipt"]["was_replayed"] is True

    direct_score = score_account_risk("4471", db_url=seeded_environment)
    assert direct_score.risk_score >= 0.65

    with session_scope(seeded_environment, readonly=True) as session:
        acct_after = session.get(Account, "4471")
        assert acct_after is not None
        assert acct_after.status == "FROZEN"
        audit_count = session.scalar(
            select(func.count(AuditActionLog.action_id)).where(
                AuditActionLog.case_id == case_id
            )
        )
        assert audit_count == 1


def test_prompt_injection_in_memo_cannot_hijack_risk_score_or_recommendation(
    seeded_environment: str,
    tmp_path: Path,
) -> None:
    """Verify adversarial memos on account 7712 are detected and do not prevent FREEZE_ACCOUNT."""
    state = run_investigation(
        "Investigate account 7712 and decide whether to freeze.",
        account_id="7712",
        case_id="CASE-7712-INJ",
        db_url=seeded_environment,
        checkpoint_file=tmp_path / "cp_inj.sqlite",
    )
    case_file = state["case_file"]
    assert case_file is not None
    assert case_file["prompt_injection_detected"] is True
    assert case_file["risk_score"] >= 0.65
    assert case_file["recommended_action"] == "FREEZE_ACCOUNT"
    assert state["status"] == "AWAITING_HUMAN_APPROVAL"


def test_tight_budget_cuts_policy_rag_agent_first_and_completes_safely(
    seeded_environment: str,
    tmp_path: Path,
) -> None:
    """Verify Supervisor enforces step budget by cutting policy_researcher first and routing to case_writer."""
    state = run_investigation(
        "Investigate account 5590 under tight step budget.",
        account_id="5590",
        case_id="CASE-5590-BUDGET",
        db_url=seeded_environment,
        max_steps=5,
        checkpoint_file=tmp_path / "cp_budget.sqlite",
    )
    agents_invoked = [evt["agent"] for evt in state["trace_events"]]
    assert "ledger_analyst" in agents_invoked
    assert "risk_scorer" in agents_invoked
    assert "case_writer" in agents_invoked
    assert "policy_researcher" not in agents_invoked

    case_file = state["case_file"]
    assert case_file is not None
    assert case_file["budget_exhausted_warning"] is not None
    assert case_file["recommended_action"] == "FREEZE_ACCOUNT"


def test_java_payments_lab_bridge_sync_and_investigation(
    seeded_environment: str,
    tmp_path: Path,
) -> None:
    """Verify Java 21 payments-lab (ledger_sentinel_ai) UUID account and entries sync into FinVigil."""
    java_uuid = "8a4d2c10-9f1b-4c22-a111-000000004471"
    # Newest-first entries as returned by payments-lab GET /v1/accounts/{accountId}/entries
    entries = [
        JavaLedgerEntryItem(
            journalEntryId="je-pl-005",
            kind="TRANSFER",
            transferId="tx-pl-005",
            amountMinor=-3_890_000,
            currency="EUR",
            createdAt="2026-11-11T18:05:00Z",
        ),
        JavaLedgerEntryItem(
            journalEntryId="je-pl-004",
            kind="FUNDING",
            transferId="tx-pl-004",
            amountMinor=980_000,
            currency="EUR",
            createdAt="2026-11-11T17:45:00Z",
        ),
        JavaLedgerEntryItem(
            journalEntryId="je-pl-003",
            kind="FUNDING",
            transferId="tx-pl-003",
            amountMinor=980_000,
            currency="EUR",
            createdAt="2026-11-11T17:30:00Z",
        ),
        JavaLedgerEntryItem(
            journalEntryId="je-pl-002",
            kind="FUNDING",
            transferId="tx-pl-002",
            amountMinor=980_000,
            currency="EUR",
            createdAt="2026-11-11T17:15:00Z",
        ),
        JavaLedgerEntryItem(
            journalEntryId="je-pl-001",
            kind="FUNDING",
            transferId="tx-pl-001",
            amountMinor=980_000,
            currency="EUR",
            createdAt="2026-11-11T17:00:00Z",
        ),
    ]
    view = ExternalLedgerAccountView(
        account_id=java_uuid,
        holder_name="payments-lab:alice",
        status="ACTIVE",
        balance_cents=30_000,
        currency="EUR",
        postings=entries,
    )
    sync_account_view_into_finvigil(view, target_db_url=seeded_environment)

    state = run_investigation(
        f"Investigate Java payments-lab account {java_uuid}",
        case_id="CASE-JAVA-PL-01",
        db_url=seeded_environment,
        checkpoint_file=tmp_path / "cp_java_pl.sqlite",
    )
    case_file = state["case_file"]
    assert case_file is not None
    assert case_file["account_id"] == java_uuid
    assert "tx-pl-005" in case_file["cited_transaction_ids"]
    assert case_file["risk_score"] >= 0.65
    assert case_file["recommended_action"] == "FREEZE_ACCOUNT"
