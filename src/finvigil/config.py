"""Configuration settings for FinVigil."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_prefix="FINVIGIL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Environment & V1 Read-Only Safety Mode
    environment: Literal["local", "test", "staging", "production"] = "local"
    service_name: str = "finvigil"
    enable_freeze_actions: bool = Field(
        default=False,
        description=(
            "V1 defaults to read-only (False). Set True only when authenticated "
            "case-bound freeze execution is explicitly enabled."
        ),
    )

    # Ledger Source ("local_db" for seeded Postgres/SQLite, "java_ledger_api" for Java payments-lab API)
    ledger_source: Literal["local_db", "java_ledger_api", "java_postgres"] = "local_db"
    java_ledger_base_url: str = "http://localhost:8080"
    java_ledger_api_key: str = ""
    java_ledger_database_url: str = (
        "postgresql+psycopg://payments:payments-local-only@127.0.0.1:55432/payments"
    )

    # Database (Postgres 16 in Docker/Azure, SQLite default for local/test runs)
    database_url: str = Field(
        default=f"sqlite:///{PROJECT_ROOT / 'finvigil.db'}",
        description="Primary read-write SQLAlchemy database URL.",
    )
    readonly_database_url: str | None = Field(
        default=None,
        description="Read-only database role URL used by the Ledger Analyst SQL tool.",
    )
    checkpoint_db_path: Path = PROJECT_ROOT / "finvigil_checkpoints.sqlite"

    # Redis
    redis_url: str = "redis://localhost:56379/0"

    # LLM Provider: supports Open-Source Chinese models (Qwen 2.5 7B, DeepSeek-V3/R1) via Ollama or API
    llm_provider: Literal["offline", "deepseek", "qwen", "ollama", "azure"] = "ollama"
    llm_base_url: str = Field(
        default="http://localhost:11434/v1",
        description=(
            "OpenAI-compatible base URL: e.g. http://localhost:11434/v1 (Ollama), "
            "https://api.deepseek.com/v1 (DeepSeek), or "
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1 (Alibaba Qwen)."
        ),
    )
    llm_api_key: str = ""
    llm_model: str = Field(
        default="qwen2.5:7b",
        description="Model ID, e.g. 'qwen2.5:7b', 'deepseek-chat', 'deepseek-reasoner', or 'deepseek-r1:8b'.",
    )
    llm_timeout_seconds: float = 30.0

    # Authentication for Human-in-the-Loop Approval Endpoint
    # Maps Bearer token -> authenticated compliance officer identity
    auth_tokens_json: str = Field(
        default='{"tok-nordea-aml-lead-991": "officer:anna.lindqvist@nordea.example", "tok-bbva-compliance-442": "officer:carlos.vega@bbva.example"}',
        description="JSON map of valid Bearer tokens to verified compliance officer IDs.",
    )

    # Hard Budgets (enforced by Supervisor per investigation request)
    max_steps_per_request: int = 8
    max_tokens_per_request: int = 12_000
    max_cost_eur_per_request: float = 0.15
    # DeepSeek-V3 / Qwen-2.5 / GPT-4o-mini realistic token pricing in EUR per 1k tokens
    input_cost_eur_per_1k_tokens: float = 0.00027
    output_cost_eur_per_1k_tokens: float = 0.00110

    # Safe SQL Guard
    sql_max_rows: int = 100
    sql_statement_timeout_ms: int = 2000

    # Deterministic XGBoost Risk Model
    risk_model_path: Path = PROJECT_ROOT / "ml" / "artifacts" / "xgboost_fraud_v1.json"
    risk_model_version: str = "xgb-paysim-v2.0.0"
    freeze_risk_threshold: float = 0.65
    mlflow_tracking_uri: str = "http://localhost:5000"

    @property
    def effective_readonly_url(self) -> str:
        """Return the read-only URL if configured, otherwise fallback to database_url."""
        return self.readonly_database_url or self.database_url


settings = Settings()
