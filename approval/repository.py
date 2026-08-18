from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from approval.models import (
    ActionProposal,
    ApprovalAuditEvent,
    ApprovalRequest,
    ExecutionReceipt,
)


class ApprovalRepositoryError(Exception):
    """Raised when durable approval state cannot be read or written."""


class SQLiteApprovalRepository:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def save_action(self, proposal: ActionProposal) -> None:
        self._execute(
            """
            INSERT OR REPLACE INTO actions
            (action_id, version, plan_id, task_id, user_id, status, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                proposal.action_id,
                proposal.version,
                proposal.plan_id,
                proposal.task_id,
                proposal.user_id,
                proposal.status.value,
                json.dumps(proposal.to_dict(), ensure_ascii=True),
                proposal.updated_at,
            ),
        )

    def get_action(
        self, action_id: str, version: int | None = None
    ) -> ActionProposal | None:
        query = "SELECT payload_json FROM actions WHERE action_id = ?"
        values: tuple = (action_id,)
        if version is not None:
            query += " AND version = ?"
            values = (action_id, version)
        else:
            query += " ORDER BY version DESC LIMIT 1"
        row = self._fetchone(query, values)
        return ActionProposal.from_dict(json.loads(row[0])) if row else None

    def save_approval(self, request: ApprovalRequest) -> None:
        self._execute(
            """
            INSERT OR REPLACE INTO approvals
            (approval_id, action_id, action_version, plan_id, task_id, user_id,
             status, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request.approval_id,
                request.action_id,
                request.action_version,
                request.plan_id,
                request.task_id,
                request.user_id,
                request.status.value,
                json.dumps(request.to_dict(), ensure_ascii=True),
                request.updated_at,
            ),
        )

    def get_approval(self, approval_id: str) -> ApprovalRequest | None:
        row = self._fetchone(
            "SELECT payload_json FROM approvals WHERE approval_id = ?",
            (approval_id,),
        )
        return ApprovalRequest.from_dict(json.loads(row[0])) if row else None

    def get_action_approval(
        self, action_id: str, action_version: int
    ) -> ApprovalRequest | None:
        row = self._fetchone(
            """
            SELECT payload_json FROM approvals
            WHERE action_id = ? AND action_version = ?
            ORDER BY updated_at DESC LIMIT 1
            """,
            (action_id, action_version),
        )
        return ApprovalRequest.from_dict(json.loads(row[0])) if row else None

    def save_receipt(self, receipt: ExecutionReceipt) -> ExecutionReceipt:
        existing = self.get_receipt(receipt.idempotency_key)
        if existing is not None:
            return existing
        self._execute(
            """
            INSERT INTO receipts
            (receipt_id, idempotency_key, action_id, action_version, status, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.receipt_id,
                receipt.idempotency_key,
                receipt.action_id,
                receipt.action_version,
                receipt.status,
                json.dumps(receipt.to_dict(), ensure_ascii=True),
            ),
        )
        return receipt

    def get_receipt(self, idempotency_key: str) -> ExecutionReceipt | None:
        row = self._fetchone(
            "SELECT payload_json FROM receipts WHERE idempotency_key = ?",
            (idempotency_key,),
        )
        return ExecutionReceipt.from_dict(json.loads(row[0])) if row else None

    def append_audit(self, event: ApprovalAuditEvent) -> None:
        self._execute(
            """
            INSERT INTO audit_events
            (event_id, event_type, action_id, action_version, approval_id,
             plan_id, task_id, user_id, created_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.event_type,
                event.action_id,
                event.action_version,
                event.approval_id,
                event.plan_id,
                event.task_id,
                event.user_id,
                event.created_at,
                json.dumps(event.metadata, ensure_ascii=True),
            ),
        )

    def list_audit(self, action_id: str) -> list[ApprovalAuditEvent]:
        rows = self._fetchall(
            """
            SELECT event_id, event_type, action_id, action_version, approval_id,
                   plan_id, task_id, user_id, created_at, metadata_json
            FROM audit_events WHERE action_id = ? ORDER BY created_at
            """,
            (action_id,),
        )
        return [
            ApprovalAuditEvent(
                event_id=row[0],
                event_type=row[1],
                action_id=row[2],
                action_version=row[3],
                approval_id=row[4],
                plan_id=row[5],
                task_id=row[6],
                user_id=row[7],
                created_at=row[8],
                metadata=json.loads(row[9]),
            )
            for row in rows
        ]

    def save_workflow(self, state, *, user_id: str) -> None:
        from planner.serialization import plan_state_to_dict

        self._execute(
            """
            INSERT OR REPLACE INTO workflows
            (plan_id, user_id, status, state_json, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                state.plan_id,
                user_id,
                state.status.value,
                json.dumps(plan_state_to_dict(state), ensure_ascii=True),
                state.updated_at,
            ),
        )

    def load_workflow(self, plan_id: str):
        from planner.serialization import plan_state_from_dict

        row = self._fetchone(
            "SELECT state_json FROM workflows WHERE plan_id = ?", (plan_id,)
        )
        return plan_state_from_dict(json.loads(row[0])) if row else None

    def find_waiting_workflow(self, *, user_id: str):
        from planner.serialization import plan_state_from_dict

        row = self._fetchone(
            """
            SELECT state_json FROM workflows
            WHERE user_id = ? AND status = 'waiting_for_approval'
            ORDER BY updated_at DESC LIMIT 1
            """,
            (user_id,),
        )
        return plan_state_from_dict(json.loads(row[0])) if row else None

    def _initialize(self) -> None:
        statements = [
            """
            CREATE TABLE IF NOT EXISTS actions (
                action_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                plan_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (action_id, version)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS approvals (
                approval_id TEXT PRIMARY KEY,
                action_id TEXT NOT NULL,
                action_version INTEGER NOT NULL,
                plan_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS receipts (
                receipt_id TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL UNIQUE,
                action_id TEXT NOT NULL,
                action_version INTEGER NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                action_id TEXT NOT NULL,
                action_version INTEGER NOT NULL,
                approval_id TEXT,
                plan_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS workflows (
                plan_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                status TEXT NOT NULL,
                state_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
        ]
        try:
            with self._connection() as connection:
                for statement in statements:
                    connection.execute(statement)
        except sqlite3.Error as exc:
            raise ApprovalRepositoryError(f"Approval database initialization failed: {exc}") from exc

    def _execute(self, query: str, values: tuple) -> None:
        try:
            with self._connection() as connection:
                connection.execute(query, values)
        except sqlite3.Error as exc:
            raise ApprovalRepositoryError(f"Approval database write failed: {exc}") from exc

    def _fetchone(self, query: str, values: tuple):
        try:
            with self._connection() as connection:
                return connection.execute(query, values).fetchone()
        except sqlite3.Error as exc:
            raise ApprovalRepositoryError(f"Approval database read failed: {exc}") from exc

    def _fetchall(self, query: str, values: tuple):
        try:
            with self._connection() as connection:
                return connection.execute(query, values).fetchall()
        except sqlite3.Error as exc:
            raise ApprovalRepositoryError(f"Approval database read failed: {exc}") from exc

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
