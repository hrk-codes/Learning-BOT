from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path

from memory.models import (
    MemoryEvent,
    MemoryRecord,
    MemoryScope,
    MemorySource,
    MemoryStatus,
    MemoryType,
)


logger = logging.getLogger(__name__)
_DATABASE_LOCK = threading.RLock()


class MemoryRepositoryError(Exception):
    """Raised when the persistent memory store cannot complete an operation."""


class SQLiteMemoryRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @contextmanager
    def _transaction(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            # sqlite3.Connection.__exit__ controls the transaction but does not
            # close the handle. Explicit closure prevents locked files on Windows
            # and keeps Streamlit reruns from accumulating database descriptors.
            connection.close()

    def _initialize_schema(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS projects (
            project_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (project_id, user_id),
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );

        CREATE TABLE IF NOT EXISTS memories (
            memory_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            project_id TEXT,
            memory_type TEXT NOT NULL,
            scope TEXT NOT NULL,
            memory_key TEXT NOT NULL,
            content TEXT NOT NULL,
            normalized_content TEXT NOT NULL,
            source TEXT NOT NULL,
            confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
            importance REAL NOT NULL CHECK (importance >= 0 AND importance <= 1),
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            valid_from TEXT NOT NULL,
            valid_until TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );

        CREATE INDEX IF NOT EXISTS idx_memories_user_status
            ON memories(user_id, status);
        CREATE INDEX IF NOT EXISTS idx_memories_scope
            ON memories(user_id, scope, project_id, status);
        CREATE INDEX IF NOT EXISTS idx_memories_key
            ON memories(user_id, memory_key, scope, project_id, status);
        CREATE INDEX IF NOT EXISTS idx_memories_normalized
            ON memories(user_id, normalized_content, status);

        CREATE TABLE IF NOT EXISTS memory_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id TEXT,
            user_id TEXT NOT NULL,
            project_id TEXT,
            event_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            details_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE INDEX IF NOT EXISTS idx_memory_events_user
            ON memory_events(user_id, created_at DESC);
        """
        try:
            with _DATABASE_LOCK, self._transaction() as connection:
                connection.executescript(schema)
                connection.execute("PRAGMA user_version = 1")
        except sqlite3.Error as exc:
            raise MemoryRepositoryError(f"Could not initialize memory database: {exc}") from exc

    def find_duplicate(
        self,
        *,
        user_id: str,
        project_id: str | None,
        scope: MemoryScope,
        normalized_content: str,
    ) -> MemoryRecord | None:
        row = self._fetch_one(
            """
            SELECT * FROM memories
            WHERE user_id = ? AND scope = ?
              AND COALESCE(project_id, '') = COALESCE(?, '')
              AND normalized_content = ? AND status = ?
            LIMIT 1
            """,
            (
                user_id,
                scope.value,
                project_id,
                normalized_content,
                MemoryStatus.ACTIVE.value,
            ),
        )
        return self._row_to_record(row) if row else None

    def find_active_by_key(
        self,
        *,
        user_id: str,
        project_id: str | None,
        scope: MemoryScope,
        key: str,
    ) -> list[MemoryRecord]:
        rows = self._fetch_all(
            """
            SELECT * FROM memories
            WHERE user_id = ? AND scope = ?
              AND COALESCE(project_id, '') = COALESCE(?, '')
              AND memory_key = ? AND status = ?
            ORDER BY updated_at DESC
            """,
            (user_id, scope.value, project_id, key, MemoryStatus.ACTIVE.value),
        )
        return self._rows_to_records(rows)

    def store(
        self,
        record: MemoryRecord,
        supersede: Iterable[MemoryRecord] = (),
    ) -> None:
        superseded = list(supersede)
        try:
            with _DATABASE_LOCK, self._transaction() as connection:
                self._ensure_owner(connection, record)
                for previous in superseded:
                    connection.execute(
                        """
                        UPDATE memories
                        SET status = ?, updated_at = ?, valid_until = ?
                        WHERE memory_id = ? AND user_id = ? AND status = ?
                        """,
                        (
                            MemoryStatus.SUPERSEDED.value,
                            record.created_at,
                            record.created_at,
                            previous.memory_id,
                            record.user_id,
                            MemoryStatus.ACTIVE.value,
                        ),
                    )
                    self._insert_event(
                        connection,
                        memory_id=previous.memory_id,
                        user_id=record.user_id,
                        project_id=previous.project_id,
                        event_type="superseded",
                        created_at=record.created_at,
                        details={"replacement_memory_id": record.memory_id},
                    )

                connection.execute(
                    """
                    INSERT INTO memories (
                        memory_id, user_id, project_id, memory_type, scope,
                        memory_key, content, normalized_content, source,
                        confidence, importance, status, created_at, updated_at,
                        valid_from, valid_until, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.memory_id,
                        record.user_id,
                        record.project_id,
                        record.memory_type.value,
                        record.scope.value,
                        record.key,
                        record.content,
                        record.normalized_content,
                        record.source.value,
                        record.confidence,
                        record.importance,
                        record.status.value,
                        record.created_at,
                        record.updated_at,
                        record.valid_from,
                        record.valid_until,
                        json.dumps(record.metadata, ensure_ascii=False),
                    ),
                )
                self._insert_event(
                    connection,
                    memory_id=record.memory_id,
                    user_id=record.user_id,
                    project_id=record.project_id,
                    event_type="created",
                    created_at=record.created_at,
                    details={
                        "type": record.memory_type.value,
                        "scope": record.scope.value,
                        "source": record.source.value,
                        "superseded_count": len(superseded),
                    },
                )
        except sqlite3.Error as exc:
            raise MemoryRepositoryError(f"Could not store memory: {exc}") from exc

    def retrieve_candidates(
        self,
        *,
        user_id: str,
        project_id: str | None,
        memory_types: set[MemoryType] | None = None,
    ) -> list[MemoryRecord]:
        parameters: list[object] = [
            user_id,
            MemoryStatus.ACTIVE.value,
            MemoryScope.USER.value,
        ]
        if project_id:
            scope_clause = "(scope = ? OR (scope = ? AND project_id = ?))"
            parameters.extend([MemoryScope.PROJECT.value, project_id])
        else:
            scope_clause = "scope = ?"

        type_clause = ""
        if memory_types:
            placeholders = ", ".join("?" for _ in memory_types)
            type_clause = f" AND memory_type IN ({placeholders})"
            parameters.extend(memory_type.value for memory_type in memory_types)

        rows = self._fetch_all(
            f"""
            SELECT * FROM memories
            WHERE user_id = ? AND status = ? AND {scope_clause}
              AND (valid_until IS NULL OR valid_until > strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
              {type_clause}
            ORDER BY updated_at DESC
            """,
            tuple(parameters),
        )
        return self._rows_to_records(rows)

    def list_memories(
        self,
        *,
        user_id: str,
        project_id: str | None = None,
        statuses: set[MemoryStatus] | None = None,
    ) -> list[MemoryRecord]:
        parameters: list[object] = [user_id]
        clauses = ["user_id = ?"]
        if project_id is not None:
            clauses.append("project_id = ?")
            parameters.append(project_id)
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            parameters.extend(status.value for status in statuses)
        rows = self._fetch_all(
            f"SELECT * FROM memories WHERE {' AND '.join(clauses)} ORDER BY updated_at DESC",
            tuple(parameters),
        )
        return self._rows_to_records(rows)

    def delete_memories(self, *, user_id: str, memory_ids: Iterable[str], deleted_at: str) -> int:
        requested_ids = tuple(dict.fromkeys(memory_ids))
        if not requested_ids:
            return 0
        placeholders = ", ".join("?" for _ in requested_ids)
        try:
            with _DATABASE_LOCK, self._transaction() as connection:
                rows = connection.execute(
                    f"SELECT memory_id, project_id FROM memories WHERE user_id = ? AND memory_id IN ({placeholders})",
                    (user_id, *requested_ids),
                ).fetchall()
                connection.execute(
                    f"DELETE FROM memories WHERE user_id = ? AND memory_id IN ({placeholders})",
                    (user_id, *requested_ids),
                )
                for row in rows:
                    self._insert_event(
                        connection,
                        memory_id=row["memory_id"],
                        user_id=user_id,
                        project_id=row["project_id"],
                        event_type="deleted",
                        created_at=deleted_at,
                        details={},
                    )
                return len(rows)
        except sqlite3.Error as exc:
            raise MemoryRepositoryError(f"Could not delete memory: {exc}") from exc

    def expire_due(self, *, user_id: str, expired_at: str) -> int:
        try:
            with _DATABASE_LOCK, self._transaction() as connection:
                rows = connection.execute(
                    """
                    SELECT memory_id, project_id FROM memories
                    WHERE user_id = ? AND status = ?
                      AND valid_until IS NOT NULL AND valid_until <= ?
                    """,
                    (user_id, MemoryStatus.ACTIVE.value, expired_at),
                ).fetchall()
                for row in rows:
                    connection.execute(
                        "UPDATE memories SET status = ?, updated_at = ? WHERE memory_id = ?",
                        (MemoryStatus.EXPIRED.value, expired_at, row["memory_id"]),
                    )
                    self._insert_event(
                        connection,
                        memory_id=row["memory_id"],
                        user_id=user_id,
                        project_id=row["project_id"],
                        event_type="expired",
                        created_at=expired_at,
                        details={},
                    )
                return len(rows)
        except sqlite3.Error as exc:
            raise MemoryRepositoryError(f"Could not expire memories: {exc}") from exc

    def list_events(self, *, user_id: str, limit: int = 50) -> list[MemoryEvent]:
        rows = self._fetch_all(
            """
            SELECT * FROM memory_events
            WHERE user_id = ? ORDER BY event_id DESC LIMIT ?
            """,
            (user_id, max(1, min(limit, 500))),
        )
        events: list[MemoryEvent] = []
        for row in rows:
            try:
                events.append(
                    MemoryEvent(
                        event_id=int(row["event_id"]),
                        memory_id=row["memory_id"],
                        user_id=row["user_id"],
                        project_id=row["project_id"],
                        event_type=row["event_type"],
                        created_at=row["created_at"],
                        details=json.loads(row["details_json"] or "{}"),
                    )
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                logger.warning("Skipped malformed memory event event_id=%s error=%s", row["event_id"], exc)
        return events

    def _ensure_owner(self, connection: sqlite3.Connection, record: MemoryRecord) -> None:
        connection.execute(
            "INSERT OR IGNORE INTO users(user_id, created_at) VALUES (?, ?)",
            (record.user_id, record.created_at),
        )
        if record.project_id:
            connection.execute(
                """
                INSERT OR IGNORE INTO projects(project_id, user_id, created_at)
                VALUES (?, ?, ?)
                """,
                (record.project_id, record.user_id, record.created_at),
            )

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection,
        *,
        memory_id: str | None,
        user_id: str,
        project_id: str | None,
        event_type: str,
        created_at: str,
        details: dict,
    ) -> None:
        connection.execute(
            """
            INSERT INTO memory_events(
                memory_id, user_id, project_id, event_type, created_at, details_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                user_id,
                project_id,
                event_type,
                created_at,
                json.dumps(details, ensure_ascii=False),
            ),
        )

    def _fetch_one(self, query: str, parameters: tuple[object, ...]) -> sqlite3.Row | None:
        try:
            with _DATABASE_LOCK, self._transaction() as connection:
                return connection.execute(query, parameters).fetchone()
        except sqlite3.Error as exc:
            raise MemoryRepositoryError(f"Could not read memory database: {exc}") from exc

    def _fetch_all(self, query: str, parameters: tuple[object, ...]) -> list[sqlite3.Row]:
        try:
            with _DATABASE_LOCK, self._transaction() as connection:
                return connection.execute(query, parameters).fetchall()
        except sqlite3.Error as exc:
            raise MemoryRepositoryError(f"Could not read memory database: {exc}") from exc

    def _rows_to_records(self, rows: Iterable[sqlite3.Row]) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        for row in rows:
            try:
                records.append(self._row_to_record(row))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                logger.warning("Skipped malformed memory record memory_id=%s error=%s", row["memory_id"], exc)
        return records

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            memory_id=row["memory_id"],
            user_id=row["user_id"],
            project_id=row["project_id"],
            memory_type=MemoryType(row["memory_type"]),
            scope=MemoryScope(row["scope"]),
            key=row["memory_key"],
            content=row["content"],
            normalized_content=row["normalized_content"],
            source=MemorySource(row["source"]),
            confidence=float(row["confidence"]),
            importance=float(row["importance"]),
            status=MemoryStatus(row["status"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            valid_from=row["valid_from"],
            valid_until=row["valid_until"],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )
