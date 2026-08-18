from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from approval.models import ActionProposal


class GateStatus(str, Enum):
    PROCEED = "proceed"
    WAITING = "waiting"
    DENIED = "denied"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    FAILED = "failed"


@dataclass(frozen=True)
class ApprovalGateOutcome:
    status: GateStatus
    message: str = ""
    approved_action: ActionProposal | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
