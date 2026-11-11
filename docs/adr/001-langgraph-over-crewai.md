# ADR 001: LangGraph over CrewAI / AutoGen for Banking Investigations

## Status
Accepted

## Context
Financial crime and AML investigations require deterministic state transitions, strict step/token/EUR budget enforcement, and durable checkpointing across multi-hour human-in-the-loop (HITL) compliance pauses.

## Decision
We chose **LangGraph** (`StateGraph` with typed `InvestigationState` and checkpoint interrupts) over autonomous conversational frameworks (CrewAI, AutoGen).

## Consequences
- **Explicit Graph Control**: Every transition returns through the `supervisor` node, where `BudgetSnapshot` enforces hard step (`<= 8`), token (`<= 12,000`), and Euro (`<= €0.15`) ceilings.
- **Native HITL Interrupts**: When `CaseFile.recommended_action == "FREEZE_ACCOUNT"`, execution pauses before `human_approval`, checkpoints state to Redis/disk, and resumes cleanly hours later via `POST /v1/investigations/{case_id}/approve`.
- **Graceful Degradation**: When step or token budgets run low, the Supervisor deterministically skips the optional `policy_researcher` node first and routes directly to `case_writer`.
