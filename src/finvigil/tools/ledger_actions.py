"""Idempotent ledger action API for account freeze execution after authenticated human approval."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime

from pydantic import BaseModel
from sqlalchemy import select

from finvigil.db.models import Account, AuditActionLog, IdempotencyRecord
from finvigil.db.session import session_scope


class AccountFreezeReceipt(BaseModel):
    """Receipt returned by an idempotent account freeze action."""

    action_id: str
    case_id: str
    account_id: str
    action_type: str = "FREEZE_ACCOUNT"
    previous_status: str
    new_status: str
    approved_by: str
    case_file_sha256: str
    idempotency_key: str
    was_replayed: bool = False
    executed_at: str


def build_idempotency_key(
    case_id: str,
    account_id: str,
    action_type: str = "FREEZE_ACCOUNT",
) -> str:
    """Generate a deterministic idempotency key from (case_id, account_id, action_type)."""
    raw = f"{case_id.strip()}::{account_id.strip()}::{action_type.strip().upper()}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return f"idem-{account_id}-{digest}"


def execute_idempotent_freeze(
    *,
    case_id: str,
    account_id: str,
    approved_by: str,
    case_file_sha256: str,
    reason: str,
    idempotency_key: str | None = None,
    db_url: str | None = None,
) -> AccountFreezeReceipt:
    """Freeze a ledger account idempotently so retried agent/API calls cannot freeze twice."""
    idem_key = idempotency_key or build_idempotency_key(case_id, account_id, "FREEZE_ACCOUNT")

    with session_scope(db_url, readonly=False) as session:
        existing = session.scalar(
            select(IdempotencyRecord).where(IdempotencyRecord.idempotency_key == idem_key)
        )
        if existing is not None:
            stored_data = json.loads(existing.response_payload_json)
            stored_data["was_replayed"] = True
            return AccountFreezeReceipt.model_validate(stored_data)

        account = session.get(Account, account_id)
        if account is None:
            raise ValueError(f"Cannot freeze non-existent account '{account_id}'.")

        prev_status = account.status
        account.status = "FROZEN"

        action_id = f"ACT-{uuid.uuid4().hex[:12].upper()}"
        now_iso = datetime.now(UTC).isoformat()

        receipt = AccountFreezeReceipt(
            action_id=action_id,
            case_id=case_id,
            account_id=account_id,
            action_type="FREEZE_ACCOUNT",
            previous_status=prev_status,
            new_status="FROZEN",
            approved_by=approved_by,
            case_file_sha256=case_file_sha256,
            idempotency_key=idem_key,
            was_replayed=False,
            executed_at=now_iso,
        )

        session.add(
            AuditActionLog(
                action_id=action_id,
                case_id=case_id,
                account_id=account_id,
                action_type="FREEZE_ACCOUNT",
                approved_by=approved_by,
                case_file_sha256=case_file_sha256,
                idempotency_key=idem_key,
                previous_status=prev_status,
                new_status="FROZEN",
                reason=reason,
            )
        )
        session.add(
            IdempotencyRecord(
                idempotency_key=idem_key,
                action_type="FREEZE_ACCOUNT",
                target_account_id=account_id,
                case_id=case_id,
                actor=approved_by,
                response_payload_json=receipt.model_dump_json(),
            )
        )

    return receipt
