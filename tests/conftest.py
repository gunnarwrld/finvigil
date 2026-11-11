"""Shared pytest fixtures seeding an isolated test database and XGBoost model."""

from __future__ import annotations

from pathlib import Path

import pytest
from data.seed_ledger import seed_database
from ml.train import train_and_register_model

from finvigil.config import settings


@pytest.fixture(scope="session", autouse=True)
def seeded_environment(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Seed a session-scoped SQLite ledger and train the chronological XGBoost artifact."""
    db_dir: Path = tmp_path_factory.mktemp("finvigil_test_db")
    db_file = db_dir / "test_ledger.db"
    cp_file = db_dir / "test_checkpoints.sqlite"
    db_url = f"sqlite:///{db_file}"

    settings.database_url = db_url
    settings.readonly_database_url = db_url
    settings.checkpoint_db_path = cp_file
    settings.llm_provider = "offline"

    seed_database(db_url=db_url, seed=42)
    train_and_register_model(db_url=db_url, log_to_mlflow=False)
    return db_url
