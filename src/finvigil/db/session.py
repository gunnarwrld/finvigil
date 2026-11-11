"""Database engine and session factories (read-write and guarded read-only)."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from finvigil.config import settings
from finvigil.db.models import Base

_engines: dict[str, Engine] = {}


def get_engine(url: str | None = None, *, readonly: bool = False) -> Engine:
    """Create or retrieve a cached SQLAlchemy Engine."""
    target_url = url or (settings.effective_readonly_url if readonly else settings.database_url)
    cache_key = f"{target_url}::ro={readonly}"
    if cache_key in _engines:
        return _engines[cache_key]

    connect_args: dict[str, Any] = {}
    if target_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        connect_args["timeout"] = max(1.0, settings.sql_statement_timeout_ms / 1000.0)

    engine = create_engine(target_url, connect_args=connect_args, future=True)

    if readonly and target_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _set_sqlite_query_only(dbapi_connection: Any, _connection_record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA query_only = ON;")
            cursor.close()

    _engines[cache_key] = engine
    return engine


def init_db(url: str | None = None) -> None:
    """Create all tables in the target database."""
    engine = get_engine(url, readonly=False)
    Base.metadata.create_all(bind=engine)


@contextmanager
def session_scope(url: str | None = None, *, readonly: bool = False) -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""
    engine = get_engine(url, readonly=readonly)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    session = factory()
    try:
        yield session
        if not readonly:
            session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
