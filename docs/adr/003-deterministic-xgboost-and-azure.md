# ADR 003: Deterministic XGBoost Risk Scoring & Azure Deployment

## Status
Accepted

## Context
1. **Risk Scoring**: LLMs are uncalibrated for numerical probability estimation and vulnerable to adversarial narratives in free-text payment memos.
2. **Cloud Target**: Nordic (Nordea) and Iberian/European (BBVA) banking cloud workloads predominantly standardize on Microsoft Azure.

## Decision
1. **LLM Never Scores**: Quantitative fraud/AML scoring is performed exclusively by a versioned **XGBoost** classifier (`xgb-paysim-v1.2.0`) trained on double-entry ledger behavioral ratios (split prior to scaling, benchmarked against a `LogisticRegression` baseline, and logged to MLflow). Free-text `memo` fields are excluded from model features.
2. **Azure Container Apps + Azure Database for PostgreSQL**: Infrastructure is provisioned via Terraform (`infra/terraform/main.tf`) in `swedencentral` with `pgvector` enabled and Azure Cache for Redis backing LangGraph checkpoints.
