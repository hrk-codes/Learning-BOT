from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SideEffectType(str, Enum):
    NONE = "none"
    READ_ONLY = "read_only"
    REVERSIBLE_WRITE = "reversible_write"
    IRREVERSIBLE_WRITE = "irreversible_write"
    EXTERNAL_COMMUNICATION = "external_communication"
    FINANCIAL = "financial"
    DESTRUCTIVE = "destructive"


class ActionStatus(str, Enum):
    PROPOSED = "proposed"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    EDITED = "edited"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    EDITED = "edited"


@dataclass(frozen=True)
class RiskAssessment:
    risk_level: RiskLevel
    reason: str
    assessed_at: str = field(default_factory=utc_now_iso)
    latency_seconds: float = 0.0


@dataclass(frozen=True)
class ActionProposal:
    action_id: str
    plan_id: str
    task_id: str
    user_id: str
    tool_name: str
    tool_version: str
    version: int
    arguments: dict[str, Any]
    purpose: str
    side_effect: SideEffectType
    risk_level: RiskLevel
    risk_reason: str
    requires_approval: bool
    preview: dict[str, Any]
    argument_digest: str
    idempotency_key: str
    status: ActionStatus
    created_at: str
    updated_at: str
    expires_at: str | None = None

    @classmethod
    def create(
        cls,
        *,
        plan_id: str,
        task_id: str,
        user_id: str,
        tool_name: str,
        tool_version: str,
        arguments: dict[str, Any],
        purpose: str,
        side_effect: SideEffectType,
        assessment: RiskAssessment,
        requires_approval: bool,
        preview: dict[str, Any],
        expires_at: str | None,
        action_id: str | None = None,
        version: int = 1,
    ) -> "ActionProposal":
        action_id = action_id or f"act_{uuid.uuid4().hex}"
        now = utc_now_iso()
        return cls(
            action_id=action_id,
            plan_id=plan_id,
            task_id=task_id,
            user_id=user_id,
            tool_name=tool_name,
            tool_version=tool_version,
            version=version,
            arguments=arguments,
            purpose=purpose,
            side_effect=side_effect,
            risk_level=assessment.risk_level,
            risk_reason=assessment.reason,
            requires_approval=requires_approval,
            preview=preview,
            argument_digest=calculate_action_digest(
                tool_name, tool_version, version, arguments
            ),
            idempotency_key=f"{action_id}:v{version}",
            status=(
                ActionStatus.PENDING_APPROVAL
                if requires_approval
                else ActionStatus.APPROVED
            ),
            created_at=now,
            updated_at=now,
            expires_at=expires_at,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["side_effect"] = self.side_effect.value
        payload["risk_level"] = self.risk_level.value
        payload["status"] = self.status.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ActionProposal":
        return cls(
            **{
                **payload,
                "side_effect": SideEffectType(payload["side_effect"]),
                "risk_level": RiskLevel(payload["risk_level"]),
                "status": ActionStatus(payload["status"]),
            }
        )


@dataclass(frozen=True)
class ApprovalRequest:
    approval_id: str
    action_id: str
    action_version: int
    plan_id: str
    task_id: str
    user_id: str
    risk_level: RiskLevel
    reason: str
    preview: dict[str, Any]
    status: ApprovalStatus
    created_at: str
    updated_at: str
    expires_at: str
    decided_at: str | None = None

    @classmethod
    def create(cls, proposal: ActionProposal) -> "ApprovalRequest":
        if proposal.expires_at is None:
            raise ValueError("Approval requests require an expiration timestamp.")
        now = utc_now_iso()
        return cls(
            approval_id=f"apr_{uuid.uuid4().hex}",
            action_id=proposal.action_id,
            action_version=proposal.version,
            plan_id=proposal.plan_id,
            task_id=proposal.task_id,
            user_id=proposal.user_id,
            risk_level=proposal.risk_level,
            reason=proposal.risk_reason,
            preview=proposal.preview,
            status=ApprovalStatus.PENDING,
            created_at=now,
            updated_at=now,
            expires_at=proposal.expires_at,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["risk_level"] = self.risk_level.value
        payload["status"] = self.status.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ApprovalRequest":
        return cls(
            **{
                **payload,
                "risk_level": RiskLevel(payload["risk_level"]),
                "status": ApprovalStatus(payload["status"]),
            }
        )


@dataclass(frozen=True)
class ExecutionReceipt:
    receipt_id: str
    action_id: str
    action_version: int
    idempotency_key: str
    tool_name: str
    status: str
    executed_at: str
    external_id: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        proposal: ActionProposal,
        *,
        status: str,
        external_id: str | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ExecutionReceipt":
        return cls(
            receipt_id=f"rcpt_{uuid.uuid4().hex}",
            action_id=proposal.action_id,
            action_version=proposal.version,
            idempotency_key=proposal.idempotency_key,
            tool_name=proposal.tool_name,
            status=status,
            executed_at=utc_now_iso(),
            external_id=external_id,
            error=error,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExecutionReceipt":
        return cls(**payload)


@dataclass(frozen=True)
class ApprovalAuditEvent:
    event_id: str
    event_type: str
    action_id: str
    action_version: int
    approval_id: str | None
    plan_id: str
    task_id: str
    user_id: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


def calculate_action_digest(
    tool_name: str,
    tool_version: str,
    action_version: int,
    arguments: dict[str, Any],
) -> str:
    canonical = json.dumps(
        {
            "tool_name": tool_name,
            "tool_version": tool_version,
            "action_version": action_version,
            "arguments": arguments,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)
