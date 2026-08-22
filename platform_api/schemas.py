from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ApiModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class RegisterRequest(ApiModel):
    email: str = Field(min_length=5, max_length=320)
    password: str = Field(min_length=10, max_length=128)
    display_name: str = Field(min_length=2, max_length=120)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        normalized = value.strip().lower()
        if "@" not in normalized or normalized.startswith("@") or normalized.endswith("@"):
            raise ValueError("Enter a valid email address.")
        return normalized


class UserResponse(ApiModel):
    id: str
    email: str
    display_name: str
    role: str
    created_at: datetime


class WorkspaceResponse(ApiModel):
    id: str
    name: str
    slug: str
    created_at: datetime


class SessionResponse(ApiModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int
    user: UserResponse
    workspace: WorkspaceResponse


class RunCreate(ApiModel):
    workspace_id: str
    goal: str = Field(min_length=3, max_length=12_000)
    mode: Literal["auto", "research", "write", "review"] = "auto"


class RunEventResponse(ApiModel):
    id: int
    event_type: str
    message: str
    node: str | None
    status: str | None
    public_data: dict[str, Any]
    created_at: datetime


class RunResponse(ApiModel):
    id: str
    workspace_id: str
    goal: str
    mode: str
    status: str
    progress: float
    current_node: str | None
    result_text: str | None
    error_code: str | None
    error_message: str | None
    trace_id: str
    llm_calls: int
    input_tokens: int
    output_tokens: int
    tool_calls: int
    estimated_cost_usd: float
    queue_seconds: float
    execution_seconds: float
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class RunDetailResponse(RunResponse):
    events: list[RunEventResponse] = Field(default_factory=list)


class ToolResponse(ApiModel):
    name: str
    description: str
    permission: str
    side_effect: str
    requires_confirmation: bool
    enabled: bool
    input_schema: dict[str, Any]


class ToolUpdate(ApiModel):
    enabled: bool


class MemoryCreate(ApiModel):
    workspace_id: str | None = None
    memory_type: Literal["profile", "preference", "project", "constraint"]
    content: str = Field(min_length=2, max_length=4000)


class MemoryResponse(ApiModel):
    id: str
    workspace_id: str | None
    memory_type: str
    content: str
    is_active: bool
    created_at: datetime
    updated_at: datetime


class DocumentResponse(ApiModel):
    id: str
    workspace_id: str
    filename: str
    content_type: str
    size_bytes: int
    status: str
    chunk_count: int
    version: str
    created_at: datetime
    indexed_at: datetime | None


class ApprovalResponse(ApiModel):
    id: str
    run_id: str
    action_type: str
    summary: str
    proposal: dict[str, Any]
    status: str
    created_at: datetime
    decided_at: datetime | None


class ApprovalDecision(ApiModel):
    decision: Literal["approved", "rejected"]


class AnalyticsResponse(ApiModel):
    total_runs: int
    completed_runs: int
    failed_runs: int
    active_runs: int
    approval_rate: float
    average_execution_seconds: float
    p95_execution_seconds: float
    total_llm_calls: int
    total_tokens: int
    total_tool_calls: int
    status_counts: dict[str, int]
    daily_runs: list[dict[str, Any]]

