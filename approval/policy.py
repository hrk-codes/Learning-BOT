from __future__ import annotations

from dataclasses import dataclass

from approval.models import RiskAssessment, RiskLevel, SideEffectType
from tools.base import ToolDefinition


_RISK_ORDER = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}


@dataclass(frozen=True)
class ApprovalPolicy:
    confirmation_timeout_seconds: int = 300
    minimum_approval_risk: RiskLevel = RiskLevel.MEDIUM
    confirmation_mode: str = "per_action"

    def requires_approval(
        self, tool: ToolDefinition, assessment: RiskAssessment
    ) -> bool:
        if tool.requires_confirmation:
            return True
        if tool.permission == "side_effecting":
            return True
        if tool.side_effect not in {SideEffectType.NONE, SideEffectType.READ_ONLY}:
            return True
        return (
            _RISK_ORDER[assessment.risk_level]
            >= _RISK_ORDER[self.minimum_approval_risk]
        )

    def allowed_without_confirmation(
        self, tool: ToolDefinition, assessment: RiskAssessment
    ) -> bool:
        return not self.requires_approval(tool, assessment)
