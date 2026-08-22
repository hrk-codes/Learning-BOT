from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


_CHECKPOINTERS: dict[Path, SqliteSaver] = {}
_CHECKPOINTER_LOCK = threading.RLock()


def build_sqlite_checkpointer(path: Path) -> SqliteSaver:
    """Create a local durable checkpointer for this single-process learning app.

    A checkpointer persists graph state after each graph step. SQLite makes an
    approval pause survive a Streamlit rerun or local application restart without
    introducing a distributed workflow database. Production multi-worker systems
    should move this responsibility to a managed Postgres-backed saver.
    """

    resolved_path = path.resolve()
    with _CHECKPOINTER_LOCK:
        cached = _CHECKPOINTERS.get(resolved_path)
        if cached is not None:
            return cached

        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        # Streamlit reruns the script whenever a widget changes. Reopening a saver
        # on each rerun can leave SQLite waiting on an earlier connection before the
        # chat UI renders. One local saver per database path keeps checkpoints durable
        # without turning routine UI reruns into database setup work.
        connection = sqlite3.connect(
            resolved_path,
            check_same_thread=False,
            timeout=5,
        )
        connection.execute("PRAGMA busy_timeout = 5000")
        saver = SqliteSaver(connection)
        saver.setup()
        _CHECKPOINTERS[resolved_path] = saver
        return saver
