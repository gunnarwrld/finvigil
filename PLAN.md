# Plan: gunnarwrld/finvigil
Goal: Harden security, checkpointing, streaming, CI gates, and test coverage for the FinVigil AML & fraud investigation engine.     Milestone: v0.2 Production Hardening, 2026-12-18

## Tasks
| id | task | type | size | state |
|----|------|------|------|-------|
| T01 | Write PLAN.md + STATUS.md from the README and code | docs | S | done |
| T02 | Remove local editable path in requirements.txt and block default demo Bearer tokens in staging/production | security | S | todo |
| T03 | Replace pickle serialization in SqliteLangGraphSaver with safe serde and enable SQLite WAL mode | security | M | todo |
| T04 | Support bound SQL parameters in execute_guarded_sql to remove S608 interpolation in ledger_analyst.py | refactor | M | todo |
| T05 | Add offline unit tests for JavaLedgerReadOnlyClient, invoke_llm, REJECT approvals, and GET case 404/200 | tests | M | todo |
| T06 | Stream real incremental LangGraph node events in POST /v1/investigations/stream instead of post-run replay | feature | M | todo |
| T07 | Align CI eval gate threshold/label and set FINVIGIL_LLM_PROVIDER=offline in GitHub Actions workflow | ci | S | todo |
| T08 | Add PostgreSQL pgvector similarity query path in policy_retriever.py with SQLite fallback | refactor | M | todo |

## Done when
- All planned tasks (T01–T08) are marked `done` with small, reviewable commits.
- `ruff check src ml data evals tests`, `mypy src ml data evals tests`, and `pytest -v` pass with zero errors.
- `python -m evals.runner` passes the 60-case holdout gate (`>= 90%` pass rate, precision `>= 90%`, recall `>= 85%`).
- No hardcoded local paths, unsafe pickle deserialization, or production demo auth tokens remain.
