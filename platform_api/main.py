from __future__ import annotations

import asyncio
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from platform_api.config import get_platform_settings
from platform_api.database import build_session_factory, get_db, initialize_development_schema
from platform_api.errors import ApiError, install_error_handlers
from platform_api.models import (
    AgentRun,
    Approval,
    AuditEvent,
    Document,
    PlatformMemory,
    RunEvent,
    RunStatus,
    ToolConfig,
    User,
    Workspace,
    utc_now,
)
from platform_api.observability import RequestContextMiddleware, configure_logging, rate_limiter
from platform_api.run_service import cancel_run, create_run, owned_run, owned_workspace
from platform_api.schemas import (
    AnalyticsResponse,
    ApprovalDecision,
    ApprovalResponse,
    DocumentResponse,
    MemoryCreate,
    MemoryResponse,
    RegisterRequest,
    RunCreate,
    RunDetailResponse,
    RunEventResponse,
    RunResponse,
    SessionResponse,
    ToolResponse,
    ToolUpdate,
    UserResponse,
    WorkspaceResponse,
)
from platform_api.security import authenticate_user, create_access_token, get_current_user, hash_password
from platform_api.worker_service import WorkerService, WorkerThread
from tools.factory import build_default_registry
from rag.storage.vector_store import JsonVectorStore


settings = get_platform_settings()
configure_logging(settings.log_level)
session_factory = build_session_factory(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_development_schema(settings)
    worker_thread = None
    if settings.embedded_worker:
        worker_thread = WorkerThread(WorkerService(settings, session_factory))
        worker_thread.start()
    app.state.worker_thread = worker_thread
    yield
    if worker_thread is not None:
        worker_thread.stop()


app = FastAPI(
    title="Learning BOT Agent Platform",
    version="11.0.0",
    description="Production API boundary for the Stage 10 multi-agent runtime.",
    lifespan=lifespan,
    docs_url="/api/docs" if not settings.is_production else None,
    redoc_url=None,
    openapi_url="/api/openapi.json" if not settings.is_production else None,
)
app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["authorization", "content-type", "idempotency-key", "x-request-id"],
)
install_error_handlers(app)


def audit(
    db: Session,
    request: Request,
    action: str,
    resource_type: str,
    *,
    user_id: str | None = None,
    workspace_id: str | None = None,
    resource_id: str | None = None,
    details: dict | None = None,
) -> None:
    db.add(
        AuditEvent(
            user_id=user_id,
            workspace_id=workspace_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            request_id=request.state.request_id,
            details=details or {},
        )
    )


def request_limit(request: Request, scope: str, limit: int, actor: str) -> None:
    rate_limiter.check(f"{scope}:{actor}", limit)


@app.get("/health/live", tags=["health"])
def live() -> dict:
    return {"status": "alive", "service": "agent-platform-api", "version": "11.0.0"}


@app.get("/health/ready", tags=["health"])
def ready(db: Session = Depends(get_db)) -> dict:
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        raise ApiError(503, "database_unavailable", "The database is not ready.") from exc
    return {
        "status": "ready",
        "database": "connected",
        "worker_mode": "embedded" if settings.embedded_worker else "external",
    }


@app.post("/api/v1/auth/register", response_model=SessionResponse, status_code=201, tags=["auth"])
def register(payload: RegisterRequest, request: Request, db: Session = Depends(get_db)) -> SessionResponse:
    request_limit(request, "register", 5, request.client.host if request.client else "unknown")
    user = User(
        email=payload.email,
        display_name=payload.display_name.strip(),
        password_hash=hash_password(payload.password),
    )
    db.add(user)
    try:
        db.flush()
    except IntegrityError as exc:
        db.rollback()
        raise ApiError(409, "email_exists", "An account with this email already exists.") from exc
    workspace = Workspace(owner_id=user.id, name="My AI Workspace", slug="my-ai-workspace")
    db.add(workspace)
    db.flush()
    for tool in build_default_registry().list_tools():
        db.add(
            ToolConfig(
                user_id=user.id,
                workspace_id=workspace.id,
                tool_name=tool.name,
                enabled=tool.permission in {"safe", "read_only_external"},
            )
        )
    audit(db, request, "account.registered", "user", user_id=user.id, workspace_id=workspace.id, resource_id=user.id)
    db.commit()
    token, expires_in = create_access_token(user)
    return SessionResponse(
        access_token=token,
        expires_in=expires_in,
        user=UserResponse.model_validate(user),
        workspace=WorkspaceResponse.model_validate(workspace),
    )


@app.post("/api/v1/auth/token", response_model=SessionResponse, tags=["auth"])
def login(
    request: Request,
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
) -> SessionResponse:
    request_limit(request, "login", 10, request.client.host if request.client else "unknown")
    user = authenticate_user(db, form.username, form.password)
    if user is None:
        raise ApiError(401, "invalid_credentials", "Email or password is incorrect.")
    workspace = db.scalar(select(Workspace).where(Workspace.owner_id == user.id).order_by(Workspace.created_at))
    if workspace is None:
        raise ApiError(500, "workspace_missing", "This account has no workspace.")
    audit(db, request, "session.created", "user", user_id=user.id, workspace_id=workspace.id, resource_id=user.id)
    db.commit()
    token, expires_in = create_access_token(user)
    return SessionResponse(
        access_token=token,
        expires_in=expires_in,
        user=UserResponse.model_validate(user),
        workspace=WorkspaceResponse.model_validate(workspace),
    )


@app.get("/api/v1/session", response_model=SessionResponse, tags=["auth"])
def session(current: User = Depends(get_current_user), db: Session = Depends(get_db)) -> SessionResponse:
    workspace = db.scalar(select(Workspace).where(Workspace.owner_id == current.id).order_by(Workspace.created_at))
    if workspace is None:
        raise ApiError(500, "workspace_missing", "This account has no workspace.")
    token, expires_in = create_access_token(current)
    return SessionResponse(
        access_token=token,
        expires_in=expires_in,
        user=UserResponse.model_validate(current),
        workspace=WorkspaceResponse.model_validate(workspace),
    )


@app.post("/api/v1/runs", response_model=RunResponse, status_code=202, tags=["runs"])
def submit_run(
    payload: RunCreate,
    request: Request,
    response: Response,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AgentRun:
    request_limit(request, "run", settings.run_rate_limit, current.id)
    run, created = create_run(
        db,
        user=current,
        workspace_id=payload.workspace_id,
        goal=payload.goal,
        mode=payload.mode,
        idempotency_key=idempotency_key,
        request_id=request.state.request_id,
    )
    response.status_code = 202 if created else 200
    audit(
        db,
        request,
        "run.created" if created else "run.replayed",
        "run",
        user_id=current.id,
        workspace_id=payload.workspace_id,
        resource_id=run.id,
        details={"mode": payload.mode},
    )
    db.commit()
    return run


@app.get("/api/v1/runs", response_model=list[RunResponse], tags=["runs"])
def list_runs(
    workspace_id: str,
    status: str | None = None,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[AgentRun]:
    owned_workspace(db, current, workspace_id)
    query = select(AgentRun).where(AgentRun.user_id == current.id, AgentRun.workspace_id == workspace_id)
    if status:
        query = query.where(AgentRun.status == status)
    return list(db.scalars(query.order_by(AgentRun.created_at.desc()).limit(100)).all())


@app.get("/api/v1/runs/{run_id}", response_model=RunDetailResponse, tags=["runs"])
def get_run(run_id: str, current: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    run = owned_run(db, current.id, run_id)
    events = db.scalars(select(RunEvent).where(RunEvent.run_id == run.id).order_by(RunEvent.id)).all()
    return {**RunResponse.model_validate(run).model_dump(), "events": events}


@app.post("/api/v1/runs/{run_id}/cancel", response_model=RunResponse, tags=["runs"])
def stop_run(
    run_id: str,
    request: Request,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AgentRun:
    run = owned_run(db, current.id, run_id)
    run = cancel_run(db, run)
    audit(db, request, "run.cancel_requested", "run", user_id=current.id, workspace_id=run.workspace_id, resource_id=run.id)
    db.commit()
    return run


@app.get("/api/v1/runs/{run_id}/events", tags=["runs"])
async def stream_run_events(
    run_id: str,
    after: int = 0,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    owned_run(db, current.id, run_id)
    user_id = current.id

    async def events():
        cursor = max(0, after)
        idle_ticks = 0
        while True:
            with session_factory() as poll_db:
                run = owned_run(poll_db, user_id, run_id)
                rows = poll_db.scalars(
                    select(RunEvent).where(RunEvent.run_id == run_id, RunEvent.id > cursor).order_by(RunEvent.id)
                ).all()
                for row in rows:
                    cursor = row.id
                    payload = RunEventResponse.model_validate(row).model_dump(mode="json")
                    yield f"id: {row.id}\nevent: run_event\ndata: {json.dumps(payload)}\n\n"
                if run.status in {"completed", "failed", "cancelled"} and not rows:
                    yield f"event: done\ndata: {json.dumps({'status': run.status})}\n\n"
                    break
            idle_ticks += 1
            if idle_ticks % 15 == 0:
                yield ": keep-alive\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/v1/tools", response_model=list[ToolResponse], tags=["tools"])
def list_tools(
    workspace_id: str,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[dict]:
    owned_workspace(db, current, workspace_id)
    configs = {
        item.tool_name: item
        for item in db.scalars(select(ToolConfig).where(ToolConfig.workspace_id == workspace_id, ToolConfig.user_id == current.id)).all()
    }
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "permission": tool.permission,
            "side_effect": tool.side_effect.value,
            "requires_confirmation": tool.requires_confirmation,
            "enabled": bool(configs.get(tool.name) and configs[tool.name].enabled),
            "input_schema": tool.input_schema,
        }
        for tool in build_default_registry().list_tools()
    ]


@app.patch("/api/v1/tools/{tool_name}", response_model=ToolResponse, tags=["tools"])
def update_tool(
    tool_name: str,
    workspace_id: str,
    payload: ToolUpdate,
    request: Request,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    owned_workspace(db, current, workspace_id)
    tool = build_default_registry().get_tool(tool_name)
    if tool is None:
        raise ApiError(404, "tool_not_found", "Tool not found.")
    config = db.scalar(select(ToolConfig).where(ToolConfig.workspace_id == workspace_id, ToolConfig.tool_name == tool_name))
    if config is None:
        config = ToolConfig(user_id=current.id, workspace_id=workspace_id, tool_name=tool_name)
        db.add(config)
    config.enabled = payload.enabled
    audit(db, request, "tool.updated", "tool", user_id=current.id, workspace_id=workspace_id, resource_id=tool_name, details={"enabled": payload.enabled})
    db.commit()
    return {
        "name": tool.name,
        "description": tool.description,
        "permission": tool.permission,
        "side_effect": tool.side_effect.value,
        "requires_confirmation": tool.requires_confirmation,
        "enabled": config.enabled,
        "input_schema": tool.input_schema,
    }


@app.get("/api/v1/memories", response_model=list[MemoryResponse], tags=["memory"])
def list_memories(
    workspace_id: str,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[PlatformMemory]:
    owned_workspace(db, current, workspace_id)
    return list(db.scalars(select(PlatformMemory).where(PlatformMemory.user_id == current.id, PlatformMemory.is_active.is_(True), (PlatformMemory.workspace_id == workspace_id) | (PlatformMemory.workspace_id.is_(None))).order_by(PlatformMemory.updated_at.desc())).all())


@app.post("/api/v1/memories", response_model=MemoryResponse, status_code=201, tags=["memory"])
def create_memory_record(
    payload: MemoryCreate,
    request: Request,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PlatformMemory:
    if payload.workspace_id:
        owned_workspace(db, current, payload.workspace_id)
    item = PlatformMemory(user_id=current.id, workspace_id=payload.workspace_id, memory_type=payload.memory_type, content=payload.content.strip())
    db.add(item)
    db.flush()
    audit(db, request, "memory.created", "memory", user_id=current.id, workspace_id=payload.workspace_id, resource_id=item.id, details={"type": item.memory_type})
    db.commit()
    return item


@app.delete("/api/v1/memories/{memory_id}", status_code=204, tags=["memory"])
def delete_memory_record(
    memory_id: str,
    request: Request,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    item = db.scalar(select(PlatformMemory).where(PlatformMemory.id == memory_id, PlatformMemory.user_id == current.id))
    if item is None:
        raise ApiError(404, "memory_not_found", "Memory record not found.")
    item.is_active = False
    audit(db, request, "memory.deleted", "memory", user_id=current.id, workspace_id=item.workspace_id, resource_id=item.id)
    db.commit()
    return Response(status_code=204)


@app.get("/api/v1/documents", response_model=list[DocumentResponse], tags=["knowledge"])
def list_documents(
    workspace_id: str,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[Document]:
    owned_workspace(db, current, workspace_id)
    return list(db.scalars(select(Document).where(Document.user_id == current.id, Document.workspace_id == workspace_id).order_by(Document.created_at.desc())).all())


@app.post("/api/v1/documents", response_model=DocumentResponse, status_code=202, tags=["knowledge"])
async def upload_document(
    request: Request,
    workspace_id: str = Form(),
    version: str = Form("1"),
    file: UploadFile = File(),
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Document:
    owned_workspace(db, current, workspace_id)
    if file.content_type != "application/pdf" or not file.filename or not file.filename.lower().endswith(".pdf"):
        raise ApiError(415, "unsupported_document", "Only PDF documents are supported in Stage 11.")
    max_bytes = get_config_max_upload_bytes()
    content = await file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ApiError(413, "document_too_large", f"PDF exceeds the {max_bytes // (1024 * 1024)} MB upload limit.")
    filename = Path(file.filename).name
    document = Document(
        user_id=current.id,
        workspace_id=workspace_id,
        filename=filename,
        content_type=file.content_type,
        size_bytes=len(content),
        object_key="pending",
        status="queued",
        version=version.strip()[:40] or "1",
    )
    db.add(document)
    db.flush()
    object_key = f"users/{current.id}/workspaces/{workspace_id}/documents/{document.id}/{filename}"
    document.object_key = object_key
    target = settings.object_store_path / object_key
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    audit(db, request, "document.uploaded", "document", user_id=current.id, workspace_id=workspace_id, resource_id=document.id, details={"size_bytes": len(content), "content_type": file.content_type})
    db.commit()
    return document


@app.delete("/api/v1/documents/{document_id}", status_code=204, tags=["knowledge"])
def delete_document(
    document_id: str,
    request: Request,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    item = db.scalar(select(Document).where(Document.id == document_id, Document.user_id == current.id))
    if item is None:
        raise ApiError(404, "document_not_found", "Document not found.")
    target = settings.object_store_path / item.object_key
    if target.is_file():
        target.unlink()
    JsonVectorStore(settings.platform_vector_root / item.workspace_id / "index.json").delete_document(item.id)
    rag_source = settings.object_store_path / "rag" / item.workspace_id / item.id
    if rag_source.is_dir():
        shutil.rmtree(rag_source)
    audit(db, request, "document.deleted", "document", user_id=current.id, workspace_id=item.workspace_id, resource_id=item.id)
    db.delete(item)
    db.commit()
    return Response(status_code=204)


@app.get("/api/v1/approvals", response_model=list[ApprovalResponse], tags=["approvals"])
def list_approvals(
    workspace_id: str,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[Approval]:
    owned_workspace(db, current, workspace_id)
    return list(db.scalars(select(Approval).join(AgentRun, Approval.run_id == AgentRun.id).where(Approval.user_id == current.id, AgentRun.workspace_id == workspace_id).order_by(Approval.created_at.desc())).all())


@app.post("/api/v1/approvals/{approval_id}/decision", response_model=ApprovalResponse, tags=["approvals"])
def decide_approval(
    approval_id: str,
    payload: ApprovalDecision,
    request: Request,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Approval:
    item = db.scalar(select(Approval).where(Approval.id == approval_id, Approval.user_id == current.id))
    if item is None:
        raise ApiError(404, "approval_not_found", "Approval request not found.")
    if item.status != "pending":
        raise ApiError(409, "approval_already_decided", "This approval has already been decided.")
    item.status = payload.decision
    item.decided_at = utc_now()
    audit(db, request, f"approval.{payload.decision}", "approval", user_id=current.id, resource_id=item.id)
    db.commit()
    return item


@app.get("/api/v1/analytics", response_model=AnalyticsResponse, tags=["analytics"])
def analytics(
    workspace_id: str,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AnalyticsResponse:
    owned_workspace(db, current, workspace_id)
    runs = list(db.scalars(select(AgentRun).where(AgentRun.user_id == current.id, AgentRun.workspace_id == workspace_id).order_by(AgentRun.created_at.desc()).limit(1000)).all())
    approvals = list(db.scalars(select(Approval).join(AgentRun, Approval.run_id == AgentRun.id).where(Approval.user_id == current.id, AgentRun.workspace_id == workspace_id)).all())
    statuses = Counter(str(item.status) for item in runs)
    durations = sorted(item.execution_seconds for item in runs if item.execution_seconds > 0)
    daily = defaultdict(int)
    for item in runs:
        daily[item.created_at.date().isoformat()] += 1
    decided = [item for item in approvals if item.status in {"approved", "rejected"}]
    approved = sum(item.status == "approved" for item in decided)
    p95_index = max(0, math.ceil(len(durations) * 0.95) - 1)
    return AnalyticsResponse(
        total_runs=len(runs),
        completed_runs=statuses["completed"],
        failed_runs=statuses["failed"],
        active_runs=statuses["queued"] + statuses["running"] + statuses["waiting_for_approval"],
        approval_rate=round(approved / len(decided), 3) if decided else 0.0,
        average_execution_seconds=round(sum(durations) / len(durations), 3) if durations else 0.0,
        p95_execution_seconds=round(durations[p95_index], 3) if durations else 0.0,
        total_llm_calls=sum(item.llm_calls for item in runs),
        total_tokens=sum(item.input_tokens + item.output_tokens for item in runs),
        total_tool_calls=sum(item.tool_calls for item in runs),
        status_counts=dict(statuses),
        daily_runs=[{"date": day, "runs": count} for day, count in sorted(daily.items())[-14:]],
    )


@app.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
def metrics(db: Session = Depends(get_db)) -> str:
    counts = dict(db.execute(select(AgentRun.status, func.count()).group_by(AgentRun.status)).all())
    lines = ["# HELP agent_platform_runs_total Runs by current lifecycle state.", "# TYPE agent_platform_runs_total gauge"]
    for status, count in counts.items():
        lines.append(f'agent_platform_runs_total{{status="{status}"}} {count}')
    lines.extend(["# HELP agent_platform_build_info Static service build information.", "# TYPE agent_platform_build_info gauge", 'agent_platform_build_info{version="11.0.0"} 1'])
    return "\n".join(lines) + "\n"


def get_config_max_upload_bytes() -> int:
    from config import get_config

    return get_config().rag_max_upload_mb * 1024 * 1024
