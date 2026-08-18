from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from approval.models import RiskAssessment, RiskLevel, SideEffectType

if TYPE_CHECKING:
    from tools.base import ToolDefinition


_RISK_ORDER = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}


class RiskEngine:
    """Deterministic V1 rules; model-provided risk labels are never authoritative."""

    def evaluate(
        self, tool: "ToolDefinition", arguments: dict[str, Any]
    ) -> RiskAssessment:
        started = time.perf_counter()
        level = tool.risk_level
        reasons = [
            f"Tool side-effect class is {tool.side_effect.value}",
            f"contract base risk is {tool.risk_level.value}",
        ]

        if tool.side_effect == SideEffectType.EXTERNAL_COMMUNICATION:
            recipients = arguments.get("to") or arguments.get("recipients") or []
            if isinstance(recipients, str):
                recipients = [recipients]
            if isinstance(recipients, list) and len(recipients) > 10:
                level = _max_risk(level, RiskLevel.CRITICAL)
                reasons.append("recipient scope exceeds 10 targets")
            elif isinstance(recipients, list) and len(recipients) > 1:
                level = _max_risk(level, RiskLevel.HIGH)
                reasons.append("action targets multiple recipients")

        if tool.side_effect in {
            SideEffectType.DESTRUCTIVE,
            SideEffectType.IRREVERSIBLE_WRITE,
        }:
            level = _max_risk(level, RiskLevel.HIGH)
            targets = arguments.get("paths") or arguments.get("items") or []
            if isinstance(targets, list) and len(targets) > 20:
                level = RiskLevel.CRITICAL
                reasons.append("destructive scope exceeds 20 targets")

        if tool.side_effect == SideEffectType.FINANCIAL:
            level = RiskLevel.CRITICAL
            reasons.append("financial actions always require critical handling in Stage 8")

        return RiskAssessment(
            risk_level=level,
            reason="; ".join(reasons),
            latency_seconds=time.perf_counter() - started,
        )


def _max_risk(first: RiskLevel, second: RiskLevel) -> RiskLevel:
    return first if _RISK_ORDER[first] >= _RISK_ORDER[second] else second
