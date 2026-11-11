# ADR 002: Defense-in-Depth SQL Guard with `sqlglot` AST Parsing

## Status
Accepted

## Context
Allowing an LLM agent to inspect a relational ledger introduces risks of SQL injection, accidental DML/DDL mutations, unbounded full-table scans, or exfiltration of ground-truth ML labels (`is_fraud_label`).

## Decision
Implement a 4-layer defense in `src/finvigil/tools/sql_guard.py`:
1. **AST Verification (`sqlglot`)**: Parse every query into a Postgres AST; reject multi-statement strings, comments, locking clauses (`FOR UPDATE`), and any non-`SELECT` statement.
2. **Table, Column & Function Allowlisting**: Only `accounts`, `journal_entries`, `postings`, `transactions`, and `aml_alerts` may be queried (plus CTE aliases). Ground-truth ML training columns (`is_fraud_label`) and dangerous functions (`pg_sleep`, `pg_read_file`, `dblink`) are rejected at the AST level.
3. **Hard Row Cap & Read-Only Timeout**: Every query AST is rewritten with `LIMIT <= 100` and executed inside a read-only database session (`SET LOCAL TRANSACTION READ ONLY; SET LOCAL statement_timeout = '2000ms';`).
4. **Prompt-Injection Delimiting**: Untrusted `memo` strings from `transactions` are sanitized and wrapped inside `<untrusted_transaction_memo>` tags before reaching any agent context.
