from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from platform_api.errors import ApiError
from platform_api.models import AgentRun, RunEvent, RunStatus, TERMINAL_RUN_STATUSES, User, Workspace, new_id, utc_now


def owned_workspace(db: Session, user: User, workspace_id: str) -> Workspace:
    workspace = db.scalar(
        select(Workspace).where(Workspace.id == workspace_id, Workspace.owner_id == user.id)
    )
    if workspace is None:
        raise ApiError(404, "workspace_not_found", "Workspace not found.")
    return workspace


def owned_run(db: Session, user_id: str, run_id: str) -> AgentRun:
    run = db.scalar(select(AgentRun).where(AgentRun.id == run_id, AgentRun.user_id == user_id))
    if run is None:
        raise ApiError(404, "run_not_found", "Run not found.")
    return run


def create_run(
    db: Session,
    *,
    user: User,
    workspace_id: str,
    goal: str,
    mode: str,
    idempotency_key: str | None,
    request_id: str,
) -> tuple[AgentRun, bool]:
    owned_workspace(db, user, workspace_id)
    clean_key = idempotency_key.strip()[:160] if idempotency_key else None
    if clean_key:
        existing = db.scalar(
            select(AgentRun).where(
                AgentRun.user_id == user.id,
                AgentRun.idempotency_key == clean_key,
            )
        )
        if existing is not None:
            return existing, False
    run = AgentRun(
        user_id=user.id,
        workspace_id=workspace_id,
        goal=goal.strip(),
        mode=mode,
        status=RunStatus.QUEUED,
        trace_id=f"trc_{new_id('')[-32:]}",
        idempotency_key=clean_key,
    )
    db.add(run)
    db.flush()
    append_event(
        db,
        run,
        "run.queued",
        "Run accepted and queued for execution.",
        status=RunStatus.QUEUED.value,
        public_data={"request_id": request_id, "mode": mode},
    )
    db.commit()
    db.refresh(run)
    return run, True


def append_event(
    db: Session,
    run: AgentRun,
    event_type: str,
    message: str,
    *,
    node: str | None = None,
    status: str | None = None,
    public_data: dict[str, Any] | None = None,
) -> RunEvent:
    event = RunEvent(
        run_id=run.id,
        event_type=event_type,
        message=message[:500],
        node=node,
        status=status,
        public_data=public_data or {},
    )
    db.add(event)
    return event


def claim_next_run(db: Session) -> AgentRun | None:
    query = (
        select(AgentRun)
        .where(AgentRun.status == RunStatus.QUEUED.value)
        .order_by(AgentRun.created_at.asc())
        .limit(1)
    )
    if db.bind is not None and db.bind.dialect.name == "postgresql":
        query = query.with_for_update(skip_locked=True)
    candidate = db.scalar(query)
    if candidate is None:
        return None
    started = utc_now()
    claimed = db.execute(
        update(AgentRun)
        .where(AgentRun.id == candidate.id, AgentRun.status == RunStatus.QUEUED.value)
        .values(
            status=RunStatus.RUNNING.value,
            current_node="manager",
            progress=0.05,
            started_at=started,
            queue_seconds=max(0.0, (started - _aware(candidate.created_at)).total_seconds()),
        )
    )
    if claimed.rowcount != 1:
        db.rollback()
        return None
    db.commit()
    return db.get(AgentRun, candidate.id)


def cancel_run(db: Session, run: AgentRun) -> AgentRun:
    if run.status in {item.value for item in TERMINAL_RUN_STATUSES}:
        return run
    run.cancellation_requested = True
    if run.status == RunStatus.QUEUED.value:
        run.status = RunStatus.CANCELLED.value
        run.progress = 1.0
        run.completed_at = utc_now()
        append_event(db, run, "run.cancelled", "Queued run cancelled.", status=run.status)
    else:
        append_event(
            db,
            run,
            "run.cancellation_requested",
            "Cancellation requested; the worker will stop at the next execution boundary.",
            status=run.status,
        )
    db.commit()
    db.refresh(run)
    return run


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

