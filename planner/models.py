from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class PlanStatus(str, Enum):
    PLANNING = "planning"
    VALIDATED = "validated"
    RUNNING = "running"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskCapability(str, Enum):
    LLM = "llm"
    TOOL = "tool"
    RAG = "rag"
    MEMORY = "memory"


@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    sources: tuple[dict[str, Any], ...] = ()
    duration_seconds: float = 0.0
    attempt: int = 1
    retryable: bool = False


@dataclass
class PlanTask:
    task_id: str
    description: str
    capability: TaskCapability
    dependencies: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    output_key: str | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, Any] = field(default_factory=dict)
    query: str | None = None
    priority: int = 0
    required: bool = True
    max_retries: int = 0
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    result: TaskResult | None = None
    error: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    started_at: str | None = None
    completed_at: str | None = None

    def public_summary(self) -> dict[str, Any]:
        return {
            "id": self.task_id,
            "description": self.description,
            "status": self.status.value,
            "capability": self.capability.value,
            "tool_name": self.tool_name,
            "dependencies": list(self.dependencies),
            "inputs": list(self.inputs),
            "output_key": self.output_key,
            "priority": self.priority,
            "required": self.required,
            "attempts": self.attempts,
            "max_retries": self.max_retries,
            "duration_seconds": round(
                self.result.duration_seconds if self.result else 0.0, 4
            ),
            "error": self.error,
        }


@dataclass(frozen=True)
class PlanEvent:
    event_type: str
    timestamp: str
    task_id: str | None = None
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanMetrics:
    planning_seconds: float = 0.0
    validation_seconds: float = 0.0
    execution_seconds: float = 0.0
    evaluation_seconds: float = 0.0
    planner_calls: int = 0
    executor_llm_calls: int = 0
    evaluator_calls: int = 0
    replan_calls: int = 0
    tool_calls: int = 0
    rag_retrievals: int = 0
    memory_retrievals: int = 0


@dataclass
class GoalEvaluation:
    goal_satisfied: bool
    reason: str
    final_answer: str = ""
    replan_needed: bool = False
    missing: tuple[str, ...] = ()


@dataclass
class PlanState:
    goal: str
    tasks: list[PlanTask]
    assumptions: tuple[str, ...] = ()
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: int = 1
    revision: int = 0
    status: PlanStatus = PlanStatus.PLANNING
    active_task_id: str | None = None
    outputs: dict[str, Any] = field(default_factory=dict)
    final_answer: str = ""
    evaluation: GoalEvaluation | None = None
    execution_steps: int = 0
    events: list[PlanEvent] = field(default_factory=list)
    metrics: PlanMetrics = field(default_factory=PlanMetrics)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    completed_at: str | None = None

    def get_task(self, task_id: str) -> PlanTask | None:
        return next((task for task in self.tasks if task.task_id == task_id), None)

    def record_event(
        self,
        event_type: str,
        *,
        task_id: str | None = None,
        message: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.updated_at = utc_now_iso()
        self.events.append(
            PlanEvent(
                event_type=event_type,
                timestamp=self.updated_at,
                task_id=task_id,
                message=message,
                metadata=metadata or {},
            )
        )

    def progress(self) -> tuple[int, int]:
        required = [task for task in self.tasks if task.required]
        completed = [task for task in required if task.status == TaskStatus.COMPLETED]
        return len(completed), len(required)

    def public_summary(self) -> dict[str, Any]:
        completed, total = self.progress()
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "status": self.status.value,
            "version": self.version,
            "revision": self.revision,
            "progress": {"completed": completed, "required": total},
            "active_task_id": self.active_task_id,
            "execution_steps": self.execution_steps,
            "assumptions": list(self.assumptions),
            "output_keys": sorted(self.outputs),
            "tasks": [task.public_summary() for task in self.tasks],
            "metrics": {
                "planning_seconds": round(self.metrics.planning_seconds, 4),
                "validation_seconds": round(self.metrics.validation_seconds, 4),
                "execution_seconds": round(self.metrics.execution_seconds, 4),
                "evaluation_seconds": round(self.metrics.evaluation_seconds, 4),
                "planner_calls": self.metrics.planner_calls,
                "executor_llm_calls": self.metrics.executor_llm_calls,
                "evaluator_calls": self.metrics.evaluator_calls,
                "replan_calls": self.metrics.replan_calls,
                "tool_calls": self.metrics.tool_calls,
                "rag_retrievals": self.metrics.rag_retrievals,
                "memory_retrievals": self.metrics.memory_retrievals,
            },
        }

