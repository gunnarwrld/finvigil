"""Unit and integration tests for Hard Problem #1: sqlglot Safe SQL Guard."""

from __future__ import annotations

import pytest

from finvigil.tools.sql_guard import (
    UnsafeSQLError,
    execute_guarded_sql,
    validate_and_rewrite_sql,
)


@pytest.mark.parametrize(
    "malicious_sql",
    [
        "",
        "DROP TABLE accounts",
        "DELETE FROM transactions WHERE account_id = '4471'",
        "UPDATE accounts SET status = 'ACTIVE' WHERE account_id = '4471'",
        "INSERT INTO accounts (account_id, holder_name) VALUES ('9999', 'Hacker')",
        "SELECT * FROM accounts; DROP TABLE accounts",
        "SELECT * FROM accounts -- comment injection",
        "SELECT * FROM accounts /* block comment */",
        "SELECT * FROM pg_catalog.pg_tables",
        "SELECT * FROM information_schema.tables",
        "SELECT * FROM idempotency_records",
        "SELECT * FROM policy_chunks",
        "SELECT account_id, is_fraud_label FROM transactions",
        "SELECT pg_sleep(10) FROM accounts",
        "SELECT pg_read_file('/etc/passwd') FROM accounts",
        "SELECT * FROM accounts FOR UPDATE",
    ],
)
def test_sql_guard_rejects_unsafe_queries(malicious_sql: str) -> None:
    """Verify sqlglot AST guard rejects DML/DDL, stacked queries, non-allowlisted tables, and functions."""
    with pytest.raises(UnsafeSQLError):
        validate_and_rewrite_sql(malicious_sql)


def test_sql_guard_allows_select_and_clamps_row_limit() -> None:
    """Verify valid SELECT queries are rewritten with a bounded LIMIT <= max_rows."""
    validated = validate_and_rewrite_sql(
        "SELECT account_id, balance_eur FROM accounts WHERE status = 'ACTIVE' LIMIT 5000",
        max_rows=50,
    )
    assert validated.effective_limit == 50
    assert validated.referenced_tables == ("accounts",)
    assert "LIMIT 50" in validated.safe_sql


def test_sql_guard_allows_cte_and_sanitizes_untrusted_memos(seeded_environment: str) -> None:
    """Verify CTE queries execute and wrap untrusted memos while neutralizing tag breakout attempts."""
    cte_query = """
    WITH acct_txs AS (
        SELECT tx_id, amount_eur, memo
        FROM transactions
        WHERE sender_account_id = '7712' OR receiver_account_id = '7712'
    )
    SELECT tx_id, amount_eur, memo FROM acct_txs ORDER BY tx_id ASC
    """
    result = execute_guarded_sql(cte_query, db_url=seeded_environment)
    assert result["row_count"] == 3
    memos = [row["memo"] for row in result["rows"]]
    assert all(m.startswith("<untrusted_transaction_memo>") for m in memos)
    assert all(m.endswith("</untrusted_transaction_memo>") for m in memos)
    # Tag breakout attempt inside TX-7712-002 must be neutralized to [REDACTED_TAG]
    assert "[REDACTED_TAG]" in memos[1]
