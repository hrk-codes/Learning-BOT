from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


def build_sqlite_checkpointer(path: Path) -> SqliteSaver:
    """Create a local durable checkpointer for this single-process learning app.

    A checkpointer persists graph state after each graph step. SQLite makes an
    approval pause survive a Streamlit rerun or local application restart without
    introducing a distributed workflow database. Production multi-worker systems
    should move this responsibility to a managed Postgres-backed saver.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    saver = SqliteSaver(connection)
    saver.setup()
    return saver
