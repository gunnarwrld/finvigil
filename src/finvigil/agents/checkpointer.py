"""SQLite-backed LangGraph BaseCheckpointSaver surviving OS process restarts."""

from __future__ import annotations

import pickle
import sqlite3
import threading
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.memory import InMemorySaver

from finvigil.config import settings


class SqliteLangGraphSaver(InMemorySaver):
    """Persistent LangGraph checkpointer backed by SQLite across OS processes."""

    def __init__(self, db_path: Path | None = None) -> None:
        super().__init__()
        self._lock = threading.RLock()
        self.db_path = Path(db_path or settings.checkpoint_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_table()
        self._sync_from_disk()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), timeout=5.0)

    def _init_table(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS langgraph_thread_state (
                    thread_id TEXT PRIMARY KEY,
                    storage_bytes BLOB NOT NULL,
                    writes_bytes BLOB NOT NULL,
                    blobs_bytes BLOB NOT NULL
                )
                """
            )
            conn.commit()

    def _sync_from_disk(self, thread_id: str | None = None) -> None:
        """Hydrate in-memory structures from SQLite on checkpointer initialization."""
        with self._lock:
            with self._connect() as conn:
                if thread_id is not None:
                    rows = conn.execute(
                        "SELECT thread_id, storage_bytes, writes_bytes, blobs_bytes "
                        "FROM langgraph_thread_state WHERE thread_id = ?",
                        (thread_id,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT thread_id, storage_bytes, writes_bytes, blobs_bytes "
                        "FROM langgraph_thread_state"
                    ).fetchall()

            for tid, s_blob, w_blob, b_blob in rows:
                # Restricted to internal LangGraph serde tuples persisted in local SQLite
                ns_dict = pickle.loads(s_blob)  # noqa: S301
                self.storage[tid] = defaultdict(dict, ns_dict)
                thread_writes = pickle.loads(w_blob)  # noqa: S301
                for k, v in thread_writes.items():
                    self.writes[k] = v
                thread_blobs = pickle.loads(b_blob)  # noqa: S301
                for k, v in thread_blobs.items():
                    self.blobs[k] = v

    def _flush_thread_to_disk(self, thread_id: str) -> None:
        """Persist a thread's checkpoints, pending writes, and channel blobs to SQLite."""
        with self._lock:
            ns_dict = {ns: dict(cp_map) for ns, cp_map in self.storage.get(thread_id, {}).items()}
            thread_writes = {k: dict(v) for k, v in self.writes.items() if k[0] == thread_id}
            thread_blobs = {k: v for k, v in self.blobs.items() if k[0] == thread_id}

            s_blob = pickle.dumps(ns_dict, protocol=pickle.HIGHEST_PROTOCOL)
            w_blob = pickle.dumps(thread_writes, protocol=pickle.HIGHEST_PROTOCOL)
            b_blob = pickle.dumps(thread_blobs, protocol=pickle.HIGHEST_PROTOCOL)

            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO langgraph_thread_state (thread_id, storage_bytes, writes_bytes, blobs_bytes)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(thread_id) DO UPDATE SET
                        storage_bytes = excluded.storage_bytes,
                        writes_bytes = excluded.writes_bytes,
                        blobs_bytes = excluded.blobs_bytes
                    """,
                    (thread_id, s_blob, w_blob, b_blob),
                )
                conn.commit()

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        with self._lock:
            return super().get_tuple(config)

    def delete_thread(self, thread_id: str) -> None:
        with self._lock:
            super().delete_thread(thread_id)
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM langgraph_thread_state WHERE thread_id = ?",
                    (thread_id,),
                )
                conn.commit()

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        with self._lock:
            updated_config = super().put(config, checkpoint, metadata, new_versions)
            thread_id = str(config["configurable"]["thread_id"])
            self._flush_thread_to_disk(thread_id)
            return updated_config

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        with self._lock:
            super().put_writes(config, writes, task_id, task_path)
            thread_id = str(config["configurable"]["thread_id"])
            self._flush_thread_to_disk(thread_id)
