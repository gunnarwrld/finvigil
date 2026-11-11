"""Read-only HTTP and PostgreSQL bridge connecting FinVigil to Java 21 payments-lab (ledger_sentinel_ai)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

from finvigil.config import settings
from finvigil.db.models import Account, AmlAlert, JournalEntry, Posting, Transaction
from finvigil.db.session import session_scope


class JavaLedgerEntryItem(BaseModel):
    """Single statement posting item from payments-lab GET /v1/accounts/{accountId}/entries."""

    journal_entry_id: str = Field(alias="journalEntryId")
    kind: str  # FUNDING | TRANSFER | REVERSAL
    transfer_id: str | None = Field(default=None, alias="transferId")
    amount_minor: int = Field(alias="amountMinor")
    currency: str = "EUR"
    created_at: str = Field(alias="createdAt")

    model_config = {"populate_by_name": True}


class JavaLedgerTransferView(BaseModel):
    """Transfer representation from payments-lab GET /v1/transfers/{transferId}."""

    transfer_id: str = Field(alias="id")
    status: str  # POSTED | DECLINED | REVIEW_REQUIRED | REVERSED
    decline_reason: str | None = Field(default=None, alias="declineReason")
    source_account_id: str = Field(alias="sourceAccountId")
    destination_account_id: str = Field(alias="destinationAccountId")
    amount_minor: int = Field(alias="amountMinor")
    currency: str = "EUR"
    reference: str | None = None
    created_at: str = Field(alias="createdAt")

    model_config = {"populate_by_name": True}


class ExternalLedgerAccountView(BaseModel):
    """Read-only projection fetched from Java payments-lab (/v1/accounts/{accountId})."""

    account_id: str
    holder_name: str
    status: str
    balance_cents: int
    currency: str = "EUR"
    created_at: str | None = None
    postings: list[JavaLedgerEntryItem] = Field(default_factory=list)


class JavaLedgerReadOnlyClient:
    """HTTP client matching the OpenAPI 3.1 contract of Java 21 payments-lab (ledger_sentinel_ai)."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        self.base_url = (base_url or settings.java_ledger_base_url).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.java_ledger_api_key
        self.timeout_seconds = timeout_seconds

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def check_readiness(self) -> dict[str, Any]:
        """Check GET /health/ready on the Java payments-lab server."""
        with httpx.Client(timeout=self.timeout_seconds) as client:
            resp = client.get(f"{self.base_url}/health/ready")
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            return data

    def fetch_account_view(self, account_id: str, *, max_entries: int = 100) -> ExternalLedgerAccountView:
        """Fetch account balance/status and statement postings from Java payments-lab."""
        with httpx.Client(timeout=self.timeout_seconds, headers=self._headers()) as client:
            acct_resp = client.get(f"{self.base_url}/v1/accounts/{account_id}")
            acct_resp.raise_for_status()
            acct_data = acct_resp.json()

            items: list[JavaLedgerEntryItem] = []
            cursor: str | None = None
            while len(items) < max_entries:
                params: dict[str, Any] = {"limit": min(50, max_entries - len(items))}
                if cursor:
                    params["cursor"] = cursor
                entries_resp = client.get(
                    f"{self.base_url}/v1/accounts/{account_id}/entries",
                    params=params,
                )
                entries_resp.raise_for_status()
                page = entries_resp.json()
                for raw_item in page.get("items", []):
                    items.append(JavaLedgerEntryItem.model_validate(raw_item))
                cursor = page.get("nextCursor")
                if not cursor:
                    break

        resolved_id = str(acct_data.get("id") or acct_data.get("accountId") or account_id)
        balance_minor = int(
            acct_data.get("balanceMinor")
            if "balanceMinor" in acct_data
            else acct_data.get("balanceCents", 0)
        )
        return ExternalLedgerAccountView(
            account_id=resolved_id,
            holder_name=str(acct_data.get("holderName", f"payments-lab:{resolved_id[:8]}")),
            status=str(acct_data.get("status", "ACTIVE")),
            balance_cents=balance_minor,
            currency=str(acct_data.get("currency", "EUR")),
            created_at=acct_data.get("createdAt"),
            postings=items,
        )

    def fetch_transfer(self, transfer_id: str) -> JavaLedgerTransferView:
        """Fetch a single transfer from GET /v1/transfers/{transferId}."""
        with httpx.Client(timeout=self.timeout_seconds, headers=self._headers()) as client:
            resp = client.get(f"{self.base_url}/v1/transfers/{transfer_id}")
            resp.raise_for_status()
            return JavaLedgerTransferView.model_validate(resp.json())


def sync_account_view_into_finvigil(
    view: ExternalLedgerAccountView,
    *,
    transfers: list[JavaLedgerTransferView] | None = None,
    target_db_url: str | None = None,
) -> str:
    """Ingest a read-only Java payments-lab account + entries/transfers into FinVigil for AML investigation."""
    with session_scope(target_db_url, readonly=False) as session:
        existing = session.get(Account, view.account_id)
        if existing is None:
            existing = Account(
                account_id=view.account_id,
                holder_name=view.holder_name,
                account_type="RETAIL",
                country_code="EU",
                kyc_tier=1,
                account_age_days=60,
                status=view.status,
                balance_cents=view.balance_cents,
                balance_eur=round(view.balance_cents / 100.0, 2),
                created_at=datetime.now(UTC),
            )
            session.add(existing)
        else:
            existing.status = view.status
            existing.balance_cents = view.balance_cents
            existing.balance_eur = round(view.balance_cents / 100.0, 2)

        for sys_id, sys_name in (
            ("SYSTEM-FUNDING", "payments-lab SYSTEM Funding"),
            ("EXTERNAL-COUNTERPARTY", "payments-lab Counterparty"),
        ):
            if session.get(Account, sys_id) is None:
                session.add(
                    Account(
                        account_id=sys_id,
                        holder_name=sys_name,
                        account_type="CORPORATE",
                        country_code="CY",
                        kyc_tier=2,
                        account_age_days=365,
                        status="ACTIVE",
                        balance_cents=0,
                        balance_eur=0.0,
                        created_at=datetime.now(UTC),
                    )
                )

        # Map transfers if provided, otherwise synthesize from statement entries
        if transfers:
            for step_idx, tr in enumerate(transfers, start=100):
                if session.get(Transaction, tr.transfer_id) is not None:
                    continue
                for cp_id in (tr.source_account_id, tr.destination_account_id):
                    if session.get(Account, cp_id) is None:
                        session.add(
                            Account(
                                account_id=cp_id,
                                holder_name=f"payments-lab:{cp_id[:8]}",
                                account_type="RETAIL",
                                country_code="CY" if cp_id != view.account_id else "EU",
                                kyc_tier=1,
                                account_age_days=60,
                                status="ACTIVE",
                                balance_cents=0,
                                balance_eur=0.0,
                                created_at=datetime.now(UTC),
                            )
                        )
                flagged = (
                    tr.status in ("REVIEW_REQUIRED", "DECLINED")
                    or (850_000 <= tr.amount_minor < 1_000_000)
                    or tr.amount_minor >= 2_500_000
                )
                is_drain = tr.source_account_id == view.account_id and tr.amount_minor >= 2_500_000
                session.add(
                    Transaction(
                        tx_id=tr.transfer_id,
                        step_hour=step_idx,
                        tx_type="CASH_OUT" if is_drain else "TRANSFER",
                        amount_cents=tr.amount_minor,
                        amount_eur=round(tr.amount_minor / 100.0, 2),
                        sender_account_id=tr.source_account_id,
                        receiver_account_id=tr.destination_account_id,
                        sender_old_balance_cents=tr.amount_minor + view.balance_cents,
                        sender_new_balance_cents=view.balance_cents,
                        receiver_old_balance_cents=0,
                        receiver_new_balance_cents=tr.amount_minor,
                        sender_old_balance=round((tr.amount_minor + view.balance_cents) / 100.0, 2),
                        sender_new_balance=round(view.balance_cents / 100.0, 2),
                        receiver_old_balance=0.0,
                        receiver_new_balance=round(tr.amount_minor / 100.0, 2),
                        counterparty_country="CY" if is_drain else "EU",
                        memo=tr.reference or f"payments-lab transfer ({tr.status})",
                        is_flagged_aml=flagged,
                        typology="PAYMENTS_LAB_TRANSFER",
                        created_at=datetime.now(UTC),
                    )
                )
        else:
            for step_idx, entry in enumerate(reversed(view.postings), start=100):
                tx_id = entry.transfer_id or entry.journal_entry_id
                if session.get(Transaction, tx_id) is not None:
                    continue
                abs_cents = abs(entry.amount_minor)
                is_credit = entry.amount_minor > 0
                sender_id = "SYSTEM-FUNDING" if is_credit else view.account_id
                receiver_id = view.account_id if is_credit else "EXTERNAL-COUNTERPARTY"
                is_drain = (not is_credit) and abs_cents >= 2_500_000
                flagged = (850_000 <= abs_cents < 1_000_000) or is_drain
                session.add(
                    Transaction(
                        tx_id=tx_id,
                        step_hour=step_idx,
                        tx_type="CASH_OUT" if is_drain else "TRANSFER",
                        amount_cents=abs_cents,
                        amount_eur=round(abs_cents / 100.0, 2),
                        sender_account_id=sender_id,
                        receiver_account_id=receiver_id,
                        sender_old_balance_cents=abs_cents + view.balance_cents,
                        sender_new_balance_cents=view.balance_cents,
                        receiver_old_balance_cents=0,
                        receiver_new_balance_cents=abs_cents,
                        sender_old_balance=round((abs_cents + view.balance_cents) / 100.0, 2),
                        sender_new_balance=round(view.balance_cents / 100.0, 2),
                        receiver_old_balance=0.0,
                        receiver_new_balance=round(abs_cents / 100.0, 2),
                        counterparty_country="CY" if is_drain else "EU",
                        memo=f"payments-lab journal entry {entry.journal_entry_id} ({entry.kind})",
                        is_flagged_aml=flagged,
                        typology="PAYMENTS_LAB_JOURNAL",
                        created_at=datetime.now(UTC),
                    )
                )

        # Reconstruct running balance from newest-first statement entries
        running_cents = view.balance_cents
        seen_je_ids: set[str] = set()
        for idx, entry in enumerate(view.postings):
            posting_id = f"PL-POST-{view.account_id[:8]}-{idx + 1:03d}"
            if session.get(Posting, posting_id) is not None:
                continue
            tx_ref = entry.transfer_id or entry.journal_entry_id
            if (
                entry.journal_entry_id not in seen_je_ids
                and session.get(JournalEntry, entry.journal_entry_id) is None
            ):
                seen_je_ids.add(entry.journal_entry_id)
                session.add(
                    JournalEntry(
                        entry_id=entry.journal_entry_id,
                        tx_id=tx_ref,
                        description=f"payments-lab {entry.kind}",
                        is_balanced=True,
                    )
                )
            direction = "CREDIT" if entry.amount_minor > 0 else "DEBIT"
            abs_cents = abs(entry.amount_minor)
            session.add(
                Posting(
                    posting_id=posting_id,
                    entry_id=entry.journal_entry_id,
                    account_id=view.account_id,
                    direction=direction,
                    amount_cents=abs_cents,
                    amount_eur=round(abs_cents / 100.0, 2),
                    running_balance_cents=running_cents,
                    running_balance_eur=round(running_cents / 100.0, 2),
                )
            )
            running_cents -= entry.amount_minor

    return view.account_id


def sync_from_java_ledger_postgres(
    *,
    java_db_url: str | None = None,
    target_db_url: str | None = None,
) -> dict[str, int]:
    """Read accounts, postings, and payments from Java payments-lab PostgreSQL 18 (V1..V6 schema)."""
    source_url = java_db_url or settings.java_ledger_database_url
    engine = create_engine(source_url, future=True)

    with engine.connect() as conn:
        if engine.dialect.name == "postgresql":
            conn.execute(text("SET TRANSACTION READ ONLY"))

        accounts_rows = conn.execute(
            text(
                """
                SELECT a.id, COALESCE(c.name, 'SYS-' || a.kind) AS holder_name,
                       a.kind, a.currency, a.status, a.balance_minor, a.created_at
                FROM accounts a
                LEFT JOIN callers c ON c.id = a.owner_id
                """
            )
        ).mappings().all()

        postings_rows = conn.execute(
            text(
                """
                SELECT p.id, p.journal_entry_id, p.account_id, p.currency, p.amount_minor,
                       je.kind AS entry_kind, je.created_at
                FROM postings p
                JOIN journal_entries je ON je.id = p.journal_entry_id
                ORDER BY p.id ASC
                """
            )
        ).mappings().all()

        payments_rows = conn.execute(
            text(
                """
                SELECT id, source_account_id, dest_account_id, currency, amount_minor,
                       reference, status, decline_reason, journal_entry_id, created_at
                FROM payments
                ORDER BY created_at ASC
                """
            )
        ).mappings().all()

    synced_accounts = 0
    synced_postings = 0
    synced_payments = 0

    with session_scope(target_db_url, readonly=False) as session:
        running_balances: dict[str, int] = {}
        for row in accounts_rows:
            acct_id = str(row["id"])
            bal_cents = int(row["balance_minor"])
            running_balances[acct_id] = 0
            if session.get(Account, acct_id) is None:
                session.add(
                    Account(
                        account_id=acct_id,
                        holder_name=str(row["holder_name"]),
                        account_type="CORPORATE" if row["kind"] == "SYSTEM" else "RETAIL",
                        country_code="EU",
                        kyc_tier=3 if row["kind"] == "SYSTEM" else 1,
                        account_age_days=365 if row["kind"] == "SYSTEM" else 60,
                        status=str(row["status"]),
                        balance_cents=bal_cents,
                        balance_eur=round(bal_cents / 100.0, 2),
                        created_at=datetime.now(UTC),
                    )
                )
                synced_accounts += 1

        for idx, pay in enumerate(payments_rows, start=200):
            tx_id = str(pay["id"])
            if session.get(Transaction, tx_id) is not None:
                continue
            amt_cents = int(pay["amount_minor"])
            status_str = str(pay["status"])
            flagged = status_str in ("REVIEW_REQUIRED", "DECLINED") or amt_cents >= 900_000
            session.add(
                Transaction(
                    tx_id=tx_id,
                    step_hour=idx,
                    tx_type="TRANSFER",
                    amount_cents=amt_cents,
                    amount_eur=round(amt_cents / 100.0, 2),
                    sender_account_id=str(pay["source_account_id"]),
                    receiver_account_id=str(pay["dest_account_id"]),
                    sender_old_balance_cents=amt_cents,
                    sender_new_balance_cents=0,
                    receiver_old_balance_cents=0,
                    receiver_new_balance_cents=amt_cents,
                    sender_old_balance=round(amt_cents / 100.0, 2),
                    sender_new_balance=0.0,
                    receiver_old_balance=0.0,
                    receiver_new_balance=round(amt_cents / 100.0, 2),
                    memo=str(pay["reference"] or f"payments-lab transfer ({status_str})"),
                    is_flagged_aml=flagged,
                    typology="PAYMENTS_LAB_LIVE",
                    created_at=datetime.now(UTC),
                )
            )
            if flagged:
                alert_id = f"ALT-PL-{tx_id[:8]}"
                if session.get(AmlAlert, alert_id) is None:
                    session.add(
                        AmlAlert(
                            alert_id=alert_id,
                            account_id=str(pay["source_account_id"]),
                            trigger_rule=f"PAYMENTS_LAB_{status_str}",
                            severity="HIGH",
                            alert_summary=(
                                f"Java payments-lab transfer {tx_id} of €{amt_cents / 100.0:,.2f} "
                                f"marked {status_str} (reason={pay['decline_reason']})."
                            ),
                            created_at=datetime.now(UTC),
                        )
                    )
            synced_payments += 1

        seen_je_ids: set[str] = set()
        for p_row in postings_rows:
            posting_id = f"PG-POST-{p_row['id']}"
            if session.get(Posting, posting_id) is not None:
                continue
            acct_id = str(p_row["account_id"])
            je_id = str(p_row["journal_entry_id"])
            signed_cents = int(p_row["amount_minor"])
            running_balances[acct_id] = running_balances.get(acct_id, 0) + signed_cents
            abs_cents = abs(signed_cents)
            run_cents = running_balances[acct_id]
            if je_id not in seen_je_ids and session.get(JournalEntry, je_id) is None:
                seen_je_ids.add(je_id)
                # Ensure a synthetic transaction row exists if this was a FUNDING entry
                if session.get(Transaction, je_id) is None:
                    session.add(
                        Transaction(
                            tx_id=je_id,
                            step_hour=150,
                            tx_type="TRANSFER",
                            amount_cents=abs_cents,
                            amount_eur=round(abs_cents / 100.0, 2),
                            sender_account_id=acct_id,
                            receiver_account_id=acct_id,
                            sender_old_balance_cents=0,
                            sender_new_balance_cents=run_cents,
                            receiver_old_balance_cents=0,
                            receiver_new_balance_cents=run_cents,
                            sender_old_balance=0.0,
                            sender_new_balance=round(run_cents / 100.0, 2),
                            receiver_old_balance=0.0,
                            receiver_new_balance=round(run_cents / 100.0, 2),
                            memo=f"payments-lab journal {p_row['entry_kind']}",
                            is_flagged_aml=False,
                            typology="PAYMENTS_LAB_JOURNAL",
                            created_at=datetime.now(UTC),
                        )
                    )
                session.add(
                    JournalEntry(
                        entry_id=je_id,
                        tx_id=je_id,
                        description=f"payments-lab {p_row['entry_kind']}",
                        is_balanced=True,
                    )
                )
            session.add(
                Posting(
                    posting_id=posting_id,
                    entry_id=je_id,
                    account_id=acct_id,
                    direction="CREDIT" if signed_cents > 0 else "DEBIT",
                    amount_cents=abs_cents,
                    amount_eur=round(abs_cents / 100.0, 2),
                    running_balance_cents=run_cents,
                    running_balance_eur=round(run_cents / 100.0, 2),
                )
            )
            synced_postings += 1

    return {
        "synced_accounts": synced_accounts,
        "synced_postings": synced_postings,
        "synced_payments": synced_payments,
    }
