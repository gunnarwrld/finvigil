"""SQLAlchemy 2.0 ORM models for the double-entry bank ledger (integer minor-unit accounting)."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import BigInteger, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Base declarative class for all FinVigil ORM models."""


class Account(Base):
    """Bank account on the ledger using exact integer minor units (cents)."""

    __tablename__ = "accounts"

    account_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    holder_name: Mapped[str] = mapped_column(String(128), nullable=False)
    account_type: Mapped[str] = mapped_column(String(32), nullable=False, default="RETAIL")
    country_code: Mapped[str] = mapped_column(String(8), nullable=False, default="SE")
    kyc_tier: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    account_age_days: Mapped[int] = mapped_column(Integer, nullable=False, default=365)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="ACTIVE")
    balance_cents: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    balance_eur: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    flagged_reason: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class JournalEntry(Base):
    """Double-entry accounting journal entry header."""

    __tablename__ = "journal_entries"

    entry_id: Mapped[str] = mapped_column(String(48), primary_key=True)
    tx_id: Mapped[str] = mapped_column(
        String(48), ForeignKey("transactions.tx_id"), nullable=False, index=True
    )
    description: Mapped[str] = mapped_column(String(256), nullable=False)
    posted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    is_balanced: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class Posting(Base):
    """Individual DEBIT or CREDIT leg belonging to a double-entry JournalEntry (integer cents)."""

    __tablename__ = "postings"

    posting_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    entry_id: Mapped[str] = mapped_column(
        String(48), ForeignKey("journal_entries.entry_id"), nullable=False, index=True
    )
    account_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("accounts.account_id"), nullable=False, index=True
    )
    direction: Mapped[str] = mapped_column(String(12), nullable=False)  # DEBIT or CREDIT
    amount_cents: Mapped[int] = mapped_column(BigInteger, nullable=False)
    amount_eur: Mapped[float] = mapped_column(Float, nullable=False)
    running_balance_cents: Mapped[int] = mapped_column(BigInteger, nullable=False)
    running_balance_eur: Mapped[float] = mapped_column(Float, nullable=False)


class Transaction(Base):
    """PaySim-modelled transaction with integer cents, typology metadata, and untrusted memo."""

    __tablename__ = "transactions"

    tx_id: Mapped[str] = mapped_column(String(48), primary_key=True)
    step_hour: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    tx_type: Mapped[str] = mapped_column(
        String(24), nullable=False, index=True
    )  # TRANSFER, CASH_OUT, PAYMENT, CASH_IN, DEBIT
    amount_cents: Mapped[int] = mapped_column(BigInteger, nullable=False)
    amount_eur: Mapped[float] = mapped_column(Float, nullable=False)
    sender_account_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("accounts.account_id"), nullable=False, index=True
    )
    sender_old_balance_cents: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sender_new_balance_cents: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sender_old_balance: Mapped[float] = mapped_column(Float, nullable=False)
    sender_new_balance: Mapped[float] = mapped_column(Float, nullable=False)
    receiver_account_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("accounts.account_id"), nullable=False, index=True
    )
    receiver_old_balance_cents: Mapped[int] = mapped_column(BigInteger, nullable=False)
    receiver_new_balance_cents: Mapped[int] = mapped_column(BigInteger, nullable=False)
    receiver_old_balance: Mapped[float] = mapped_column(Float, nullable=False)
    receiver_new_balance: Mapped[float] = mapped_column(Float, nullable=False)
    counterparty_country: Mapped[str] = mapped_column(String(8), nullable=False, default="SE")
    channel: Mapped[str] = mapped_column(String(24), nullable=False, default="MOBILE")
    typology: Mapped[str] = mapped_column(String(32), nullable=False, default="BENIGN")
    memo: Mapped[str] = mapped_column(Text, nullable=False, default="")
    is_flagged_aml: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_fraud_label: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class AmlAlert(Base):
    """AML or fraud rule alert raised on an account."""

    __tablename__ = "aml_alerts"

    alert_id: Mapped[str] = mapped_column(String(48), primary_key=True)
    account_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("accounts.account_id"), nullable=False, index=True
    )
    trigger_rule: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    alert_summary: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class PolicyChunk(Base):
    """AML & PSD2 regulatory policy chunk with official CELEX/FATF citation and embedding."""

    __tablename__ = "policy_chunks"

    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    regulation: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    celex_or_source_ref: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    section_title: Mapped[str] = mapped_column(String(160), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    keywords: Mapped[str] = mapped_column(String(256), nullable=False, default="")
    embedding_json: Mapped[str] = mapped_column(Text, nullable=False)


class IdempotencyRecord(Base):
    """Idempotency store preventing duplicate ledger freeze/unfreeze actions."""

    __tablename__ = "idempotency_records"

    idempotency_key: Mapped[str] = mapped_column(String(128), primary_key=True)
    action_type: Mapped[str] = mapped_column(String(32), nullable=False)
    target_account_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    case_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    actor: Mapped[str] = mapped_column(String(64), nullable=False)
    response_payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class AuditActionLog(Base):
    """Immutable audit log of approved ledger actions."""

    __tablename__ = "audit_action_logs"

    action_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    account_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    action_type: Mapped[str] = mapped_column(String(32), nullable=False)
    approved_by: Mapped[str] = mapped_column(String(64), nullable=False)
    case_file_sha256: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    previous_status: Mapped[str] = mapped_column(String(24), nullable=False)
    new_status: Mapped[str] = mapped_column(String(24), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
