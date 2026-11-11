"""Guarded read-only SQL tool powered by sqlglot AST verification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import sqlglot
from sqlalchemy import text
from sqlglot import exp

from finvigil.config import settings
from finvigil.db.session import session_scope

ALLOWED_TABLES: frozenset[str] = frozenset(
    {
        "accounts",
        "journal_entries",
        "postings",
        "transactions",
        "aml_alerts",
    }
)

FORBIDDEN_FUNCTIONS: frozenset[str] = frozenset(
    {
        "pg_sleep",
        "pg_read_file",
        "pg_read_binary_file",
        "pg_ls_dir",
        "pg_stat_file",
        "dblink",
        "dblink_exec",
        "lo_import",
        "lo_export",
        "load_extension",
        "randomblob",
    }
)

FORBIDDEN_COLUMNS: frozenset[str] = frozenset({"is_fraud_label"})

FORBIDDEN_AST_TYPES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.Create,
    exp.Command,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Pragma,
    exp.Set,
    exp.Into,
)


class UnsafeSQLError(ValueError):
    """Raised when a candidate SQL query violates FinVigil's read-only AST policy."""


@dataclass(frozen=True)
class ValidatedSQL:
    """Result of sqlglot validation and row-limit normalization."""

    original_sql: str
    safe_sql: str
    referenced_tables: tuple[str, ...]
    effective_limit: int


def sanitize_untrusted_memo(raw_memo: str) -> str:
    """Neutralize closing-tag breakout attempts and wrap untrusted memo text."""
    cleaned = raw_memo.replace("</untrusted_transaction_memo>", "[REDACTED_TAG]")
    cleaned = cleaned.replace("<untrusted_transaction_memo>", "[REDACTED_TAG]")
    return f"<untrusted_transaction_memo>{cleaned}</untrusted_transaction_memo>"


def validate_and_rewrite_sql(
    raw_sql: str,
    *,
    max_rows: int | None = None,
    dialect: str = "postgres",
) -> ValidatedSQL:
    """Parse SQL with sqlglot, enforce SELECT-only & table allowlist, and cap LIMIT."""
    row_cap = max_rows or settings.sql_max_rows
    candidate = raw_sql.strip()
    if not candidate:
        raise UnsafeSQLError("SQL query cannot be empty.")

    # Reject SQL comments that may hide stacked directives or obfuscate prompts
    if "--" in candidate or "/*" in candidate or "*/" in candidate:
        raise UnsafeSQLError("SQL comments ('--' or '/* */') are not permitted.")

    # Check for stacked semicolons before/after parsing
    stripped_no_trailing = candidate.rstrip(";").strip()
    if ";" in stripped_no_trailing:
        raise UnsafeSQLError("Multiple SQL statements are forbidden; only a single SELECT is allowed.")

    try:
        parsed_statements = sqlglot.parse(stripped_no_trailing, read=dialect)
    except Exception as exc:
        raise UnsafeSQLError(f"SQL syntax error during sqlglot parse: {exc}") from exc

    if len(parsed_statements) != 1 or parsed_statements[0] is None:
        raise UnsafeSQLError("Exactly one valid SQL SELECT statement is required.")

    root = parsed_statements[0]

    # 1. Verify root expression is a SELECT (not UNION with DML or raw command)
    if not isinstance(root, exp.Select):
        raise UnsafeSQLError(
            f"Only SELECT statements are permitted (got {root.__class__.__name__})."
        )

    # 2. Scan entire AST for forbidden statement/clause node types (e.g. SELECT ... INTO)
    for forbidden_node in root.find_all(*FORBIDDEN_AST_TYPES):
        raise UnsafeSQLError(
            f"Forbidden SQL operation detected in AST: {forbidden_node.__class__.__name__}"
        )

    # Reject locking clauses (FOR UPDATE / FOR SHARE)
    if root.args.get("locks"):
        raise UnsafeSQLError("Row-locking clauses (FOR UPDATE / FOR SHARE) are forbidden.")

    # 3. Collect CTE alias names so WITH queries don't trigger false table violations
    cte_names: set[str] = set()
    for cte in root.find_all(exp.CTE):
        alias = cte.alias_or_name
        if alias:
            cte_names.add(alias.lower())

    # 4. Validate all referenced tables against ALLOWED_TABLES
    referenced_tables: list[str] = []
    for table_node in root.find_all(exp.Table):
        schema_name = (table_node.db or "").lower()
        table_name = (table_node.name or "").lower()

        if schema_name and schema_name not in {"public", "main"}:
            raise UnsafeSQLError(f"Access to schema '{schema_name}' is forbidden.")

        if table_name in cte_names:
            continue

        if table_name not in ALLOWED_TABLES:
            raise UnsafeSQLError(
                f"Table '{table_name}' is not in the ledger allowlist {sorted(ALLOWED_TABLES)}."
            )
        if table_name not in referenced_tables:
            referenced_tables.append(table_name)

    if not referenced_tables:
        raise UnsafeSQLError(
            "Query must reference at least one allowlisted ledger table."
        )

    # 5. Block forbidden columns (e.g. ground-truth ML training labels)
    for col_node in root.find_all(exp.Column):
        col_name = (col_node.name or "").lower()
        if col_name in FORBIDDEN_COLUMNS:
            raise UnsafeSQLError(
                f"Column '{col_name}' is restricted and cannot be queried by agents."
            )

    # 6. Block dangerous database functions
    for func_node in root.find_all(exp.Func):
        func_name = func_node.sql_name().lower()
        if func_name in FORBIDDEN_FUNCTIONS or any(
            bad in func_name for bad in FORBIDDEN_FUNCTIONS
        ):
            raise UnsafeSQLError(f"Function '{func_name}' is forbidden by security policy.")

    for anon_node in root.find_all(exp.Anonymous):
        anon_name = (anon_node.name or "").lower()
        if anon_name in FORBIDDEN_FUNCTIONS:
            raise UnsafeSQLError(f"Function '{anon_name}' is forbidden by security policy.")

    # 7. Enforce hard LIMIT cap (<= row_cap)
    existing_limit_node = root.args.get("limit")
    effective_limit = row_cap
    if isinstance(existing_limit_node, exp.Limit) and existing_limit_node.expression is not None:
        limit_text = existing_limit_node.expression.name
        if limit_text and re.fullmatch(r"\d+", limit_text):
            requested_limit = int(limit_text)
            effective_limit = min(requested_limit, row_cap)

    rewritten_ast = root.limit(effective_limit)
    safe_sql = rewritten_ast.sql(dialect="postgres")

    return ValidatedSQL(
        original_sql=raw_sql,
        safe_sql=safe_sql,
        referenced_tables=tuple(referenced_tables),
        effective_limit=effective_limit,
    )


def execute_guarded_sql(
    raw_sql: str,
    *,
    db_url: str | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Validate SQL via sqlglot and execute it inside a read-only, timeout-bounded session."""
    validated = validate_and_rewrite_sql(raw_sql, max_rows=max_rows)

    with session_scope(db_url, readonly=True) as session:
        bind = session.get_bind()
        dialect_name = bind.dialect.name
        if dialect_name == "postgresql":
            session.execute(text("SET LOCAL TRANSACTION READ ONLY;"))
            session.execute(
                text(f"SET LOCAL statement_timeout = '{int(settings.sql_statement_timeout_ms)}ms';")
            )

        result = session.execute(text(validated.safe_sql))
        columns = list(result.keys())
        fetched = result.fetchmany(validated.effective_limit)

        rows: list[dict[str, Any]] = []
        for record in fetched:
            row_dict = dict(zip(columns, record, strict=False))
            # Exclude restricted columns if SELECT * was used on transactions
            row_dict.pop("is_fraud_label", None)
            # Wrap untrusted memo fields in explicit XML tags to prevent prompt injection
            if "memo" in row_dict and isinstance(row_dict["memo"], str) and row_dict["memo"]:
                row_dict["memo"] = sanitize_untrusted_memo(row_dict["memo"])
            rows.append(row_dict)

    return {
        "safe_sql": validated.safe_sql,
        "referenced_tables": list(validated.referenced_tables),
        "row_count": len(rows),
        "effective_limit": validated.effective_limit,
        "rows": rows,
    }
