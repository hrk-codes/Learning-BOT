from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Callable

from approval.audit import ApprovalAuditLog
from approval.models import (
    ActionProposal,
    ActionStatus,
    ApprovalRequest,
    ApprovalStatus,
    ExecutionReceipt,
    calculate_action_digest,
    parse_timestamp,
    utc_now_iso,
)
from approval.policy import ApprovalPolicy
from approval.repository import ApprovalRepositoryError, SQLiteApprovalRepository
from approval.risk_engine import RiskEngine
from tools.base import ToolDefinition


ToolLookup = Callable[[str], ToolDefinition | None]


class ApprovalServiceError(Exception):
    """Fail-closed error for approval, authorization, or persistence uncertainty."""


class ApprovalService:
    def __init__(
        self,
        *,
        repository: SQLiteApprovalRepository,
        tool_lookup: ToolLookup,
        risk_engine: RiskEngine,
        policy: ApprovalPolicy,
    ) -> None:
        self.repository = repository
        self.tool_lookup = tool_lookup
        self.risk_engine = risk_engine
        self.policy = policy
        self.audit = ApprovalAuditLog(repository)

    def assess_tool(self, tool_name: str, arguments: dict):
        tool = self._require_tool(tool_name)
        clean_arguments = tool.validate_arguments(arguments)
        assessment = self.risk_engine.evaluate(tool, clean_arguments)
        return (
            tool,
            clean_arguments,
            assessment,
            self.policy.requires_approval(tool, assessment),
        )

    def request_approval(
        self,
        *,
        plan_id: str,
        task_id: str,
        user_id: str,
        tool_name: str,
        arguments: dict,
        purpose: str,
    ) -> tuple[ActionProposal, ApprovalRequest | None]:
        tool = self._require_tool(tool_name)
        clean_arguments = tool.validate_arguments(arguments)
        assessment = self.risk_engine.evaluate(tool, clean_arguments)
        requires_approval = self.policy.requires_approval(tool, assessment)
        expires_at = None
        if requires_approval:
            expires_at = (
                datetime.now(timezone.utc)
                + timedelta(seconds=self.policy.confirmation_timeout_seconds)
            ).isoformat()
        proposal = ActionProposal.create(
            plan_id=plan_id,
            task_id=task_id,
            user_id=user_id,
            tool_name=tool.name,
            tool_version=tool.version,
            arguments=clean_arguments,
            purpose=purpose,
            side_effect=tool.side_effect,
            assessment=assessment,
            requires_approval=requires_approval,
            preview=tool.build_preview(clean_arguments),
            expires_at=expires_at,
        )
        request = ApprovalRequest.create(proposal) if requires_approval else None
        try:
            self.repository.save_action(proposal)
            if request:
                self.repository.save_approval(request)
            self.audit.record("action_proposed", proposal, request)
            self.audit.record(
                "risk_assessed",
                proposal,
                request,
                metadata={
                    "risk_level": assessment.risk_level.value,
                    "latency_seconds": round(assessment.latency_seconds, 6),
                    "requires_approval": requires_approval,
                },
            )
            if request:
                self.audit.record("approval_requested", proposal, request)
        except ApprovalRepositoryError as exc:
            raise ApprovalServiceError(str(exc)) from exc
        return proposal, request

    def approve(
        self,
        approval_id: str,
        *,
        user_id: str,
        expected_version: int,
    ) -> ActionProposal:
        request, proposal = self._decision_context(
            approval_id, user_id=user_id, expected_version=expected_version
        )
        now = utc_now_iso()
        request = replace(
            request,
            status=ApprovalStatus.APPROVED,
            updated_at=now,
            decided_at=now,
        )
        proposal = replace(proposal, status=ActionStatus.APPROVED, updated_at=now)
        self.repository.save_approval(request)
        self.repository.save_action(proposal)
        self.audit.record("approval_granted", proposal, request)
        return proposal

    def deny(
        self,
        approval_id: str,
        *,
        user_id: str,
        expected_version: int,
    ) -> ActionProposal:
        return self._reject(
            approval_id,
            user_id=user_id,
            expected_version=expected_version,
            approval_status=ApprovalStatus.DENIED,
            action_status=ActionStatus.DENIED,
            event_type="approval_denied",
        )

    def cancel(
        self,
        approval_id: str,
        *,
        user_id: str,
        expected_version: int,
    ) -> ActionProposal:
        return self._reject(
            approval_id,
            user_id=user_id,
            expected_version=expected_version,
            approval_status=ApprovalStatus.CANCELLED,
            action_status=ActionStatus.CANCELLED,
            event_type="action_cancelled",
        )

    def edit(
        self,
        approval_id: str,
        *,
        user_id: str,
        expected_version: int,
        arguments: dict,
    ) -> tuple[ActionProposal, ApprovalRequest]:
        request, proposal = self._decision_context(
            approval_id, user_id=user_id, expected_version=expected_version
        )
        tool = self._require_tool(proposal.tool_name)
        clean_arguments = tool.validate_arguments(arguments)
        assessment = self.risk_engine.evaluate(tool, clean_arguments)
        now = utc_now_iso()

        # Editing changes the action being authorized. The old approval becomes
        # unusable, and a new version must pass risk and approval checks again.
        old_request = replace(
            request,
            status=ApprovalStatus.EDITED,
            updated_at=now,
            decided_at=now,
        )
        old_proposal = replace(proposal, status=ActionStatus.EDITED, updated_at=now)
        expires_at = (
            datetime.now(timezone.utc)
            + timedelta(seconds=self.policy.confirmation_timeout_seconds)
        ).isoformat()
        new_proposal = ActionProposal.create(
            action_id=proposal.action_id,
            version=proposal.version + 1,
            plan_id=proposal.plan_id,
            task_id=proposal.task_id,
            user_id=proposal.user_id,
            tool_name=tool.name,
            tool_version=tool.version,
            arguments=clean_arguments,
            purpose=proposal.purpose,
            side_effect=tool.side_effect,
            assessment=assessment,
            requires_approval=True,
            preview=tool.build_preview(clean_arguments),
            expires_at=expires_at,
        )
        new_request = ApprovalRequest.create(new_proposal)
        self.repository.save_approval(old_request)
        self.repository.save_action(old_proposal)
        self.repository.save_action(new_proposal)
        self.repository.save_approval(new_request)
        self.audit.record(
            "action_edited",
            new_proposal,
            new_request,
            metadata={"previous_version": proposal.version},
        )
        self.audit.record("approval_requested", new_proposal, new_request)
        return new_proposal, new_request

    def get_approval(self, approval_id: str) -> ApprovalRequest:
        request = self.repository.get_approval(approval_id)
        if request is None:
            raise ApprovalServiceError("Approval state is unavailable; execution is blocked.")
        return self._expire_if_needed(request)

    def get_action(self, action_id: str, version: int | None = None) -> ActionProposal:
        proposal = self.repository.get_action(action_id, version)
        if proposal is None:
            raise ApprovalServiceError("Action proposal is unavailable; execution is blocked.")
        return proposal

    def get_action_approval(self, action_id: str, version: int) -> ApprovalRequest:
        request = self.repository.get_action_approval(action_id, version)
        if request is None:
            raise ApprovalServiceError("Approval state is unavailable; execution is blocked.")
        return self._expire_if_needed(request)

    def get_receipt(self, idempotency_key: str) -> ExecutionReceipt | None:
        return self.repository.get_receipt(idempotency_key)

    def verify_approved(
        self,
        *,
        action_id: str,
        action_version: int,
        user_id: str,
    ) -> ActionProposal:
        proposal = self.get_action(action_id, action_version)
        request = self.get_action_approval(action_id, action_version)
        if proposal.user_id != user_id or request.user_id != user_id:
            raise ApprovalServiceError("Approval belongs to a different user session.")
        if request.status != ApprovalStatus.APPROVED:
            raise ApprovalServiceError(
                f"Action is not approved; current approval state is {request.status.value}."
            )
        if proposal.status not in {ActionStatus.APPROVED, ActionStatus.EXECUTING}:
            raise ApprovalServiceError(
                f"Action proposal is not executable; current state is {proposal.status.value}."
            )
        tool = self._require_tool(proposal.tool_name)
        if tool.version != proposal.tool_version:
            raise ApprovalServiceError("Tool contract changed after approval; new approval is required.")
        digest = calculate_action_digest(
            proposal.tool_name,
            proposal.tool_version,
            proposal.version,
            proposal.arguments,
        )
        if digest != proposal.argument_digest:
            raise ApprovalServiceError("Approved action arguments no longer match their digest.")
        return proposal

    def mark_execution_started(self, proposal: ActionProposal) -> ActionProposal:
        proposal = replace(
            proposal, status=ActionStatus.EXECUTING, updated_at=utc_now_iso()
        )
        self.repository.save_action(proposal)
        request = self.repository.get_action_approval(proposal.action_id, proposal.version)
        self.audit.record("action_started", proposal, request)
        return proposal

    def record_execution(
        self,
        proposal: ActionProposal,
        *,
        success: bool,
        external_id: str | None = None,
        error: str | None = None,
        metadata: dict | None = None,
    ) -> ExecutionReceipt:
        existing = self.repository.get_receipt(proposal.idempotency_key)
        if existing is not None:
            return existing
        status = "completed" if success else "failed"
        receipt = ExecutionReceipt.create(
            proposal,
            status=status,
            external_id=external_id,
            error=error,
            metadata=metadata,
        )
        receipt = self.repository.save_receipt(receipt)
        updated = replace(
            proposal,
            status=ActionStatus.COMPLETED if success else ActionStatus.FAILED,
            updated_at=utc_now_iso(),
        )
        self.repository.save_action(updated)
        request = self.repository.get_action_approval(proposal.action_id, proposal.version)
        self.audit.record(
            "action_completed" if success else "action_failed",
            updated,
            request,
            metadata={"receipt_id": receipt.receipt_id, "status": receipt.status},
        )
        return receipt

    def save_workflow(self, state, *, user_id: str) -> None:
        try:
            self.repository.save_workflow(state, user_id=user_id)
        except ApprovalRepositoryError as exc:
            raise ApprovalServiceError(str(exc)) from exc

    def load_workflow(self, plan_id: str):
        return self.repository.load_workflow(plan_id)

    def find_waiting_workflow(self, *, user_id: str):
        try:
            return self.repository.find_waiting_workflow(user_id=user_id)
        except ApprovalRepositoryError as exc:
            raise ApprovalServiceError(str(exc)) from exc

    def list_audit(self, action_id: str):
        try:
            return self.repository.list_audit(action_id)
        except ApprovalRepositoryError as exc:
            raise ApprovalServiceError(str(exc)) from exc

    def _decision_context(
        self, approval_id: str, *, user_id: str, expected_version: int
    ) -> tuple[ApprovalRequest, ActionProposal]:
        request = self.get_approval(approval_id)
        if request.user_id != user_id:
            raise ApprovalServiceError("Approval belongs to a different user session.")
        if request.action_version != expected_version:
            raise ApprovalServiceError("Action version changed; review the latest proposal.")
        if request.status != ApprovalStatus.PENDING:
            raise ApprovalServiceError(
                f"Approval is no longer pending; current state is {request.status.value}."
            )
        proposal = self.get_action(request.action_id, request.action_version)
        return request, proposal

    def _reject(
        self,
        approval_id: str,
        *,
        user_id: str,
        expected_version: int,
        approval_status: ApprovalStatus,
        action_status: ActionStatus,
        event_type: str,
    ) -> ActionProposal:
        request, proposal = self._decision_context(
            approval_id, user_id=user_id, expected_version=expected_version
        )
        now = utc_now_iso()
        request = replace(
            request,
            status=approval_status,
            updated_at=now,
            decided_at=now,
        )
        proposal = replace(proposal, status=action_status, updated_at=now)
        self.repository.save_approval(request)
        self.repository.save_action(proposal)
        self.audit.record(event_type, proposal, request)
        return proposal

    def _expire_if_needed(self, request: ApprovalRequest) -> ApprovalRequest:
        if request.status not in {ApprovalStatus.PENDING, ApprovalStatus.APPROVED}:
            return request
        if datetime.now(timezone.utc) < parse_timestamp(request.expires_at):
            return request
        now = utc_now_iso()
        request = replace(
            request,
            status=ApprovalStatus.EXPIRED,
            updated_at=now,
            decided_at=now,
        )
        proposal = self.get_action(request.action_id, request.action_version)
        proposal = replace(proposal, status=ActionStatus.EXPIRED, updated_at=now)
        self.repository.save_approval(request)
        self.repository.save_action(proposal)
        self.audit.record("approval_expired", proposal, request)
        return request

    def _require_tool(self, name: str) -> ToolDefinition:
        tool = self.tool_lookup(name)
        if tool is None:
            raise ApprovalServiceError(f"Tool {name!r} is unavailable; execution is blocked.")
        return tool
