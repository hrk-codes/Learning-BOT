from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from multi_agent.agents.contracts import AgentName


@dataclass(frozen=True)
class AgentResult:
    """Normalized, serializable specialist result consumed by the manager."""

    agent_name: AgentName
    task_id: str
    status: str
    output: dict[str, Any] | None
    sources: tuple[dict[str, Any], ...] = ()
    error: str | None = None
    duration_seconds: float = 0.0
    retry_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name.value,
            "task_id": self.task_id,
            "status": self.status,
            "output": self.output,
            "sources": list(self.sources),
            "error": self.error,
            "duration_seconds": round(self.duration_seconds, 4),
            "retry_count": self.retry_count,
            "metadata": self.metadata,
        }
