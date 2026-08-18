import uuid
from typing import Any

from approval.models import ActionProposal, ApprovalAuditEvent, ApprovalRequest, utc_now_iso
from approval.repository import SQLiteApprovalRepository


class ApprovalAuditLog:
    def __init__(self, repository: SQLiteApprovalRepository) -> None:
        self.repository = repository

    def record(
        self,
        event_type: str,
        proposal: ActionProposal,
        request: ApprovalRequest | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        # Audit metadata deliberately excludes arguments and preview content. The
        # durable proposal stores what is needed for execution while the event log
        # answers who/what/when without duplicating sensitive payloads.
        self.repository.append_audit(
            ApprovalAuditEvent(
                event_id=f"evt_{uuid.uuid4().hex}",
                event_type=event_type,
                action_id=proposal.action_id,
                action_version=proposal.version,
                approval_id=request.approval_id if request else None,
                plan_id=proposal.plan_id,
                task_id=proposal.task_id,
                user_id=proposal.user_id,
                created_at=utc_now_iso(),
                metadata=metadata or {},
            )
        )
