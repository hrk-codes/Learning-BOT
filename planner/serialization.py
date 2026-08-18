from __future__ import annotations

from typing import Any

from planner.models import (
    GoalEvaluation,
    PlanEvent,
    PlanMetrics,
    PlanState,
    PlanStatus,
    PlanTask,
    TaskCapability,
    TaskResult,
    TaskStatus,
)


def plan_state_to_dict(state: PlanState) -> dict[str, Any]:
    return {
        "goal": state.goal,
        "tasks": [_task_to_dict(task) for task in state.tasks],
        "assumptions": list(state.assumptions),
        "plan_id": state.plan_id,
        "version": state.version,
        "revision": state.revision,
        "status": state.status.value,
        "active_task_id": state.active_task_id,
        "outputs": state.outputs,
        "final_answer": state.final_answer,
        "evaluation": (
            {
                "goal_satisfied": state.evaluation.goal_satisfied,
                "reason": state.evaluation.reason,
                "final_answer": state.evaluation.final_answer,
                "replan_needed": state.evaluation.replan_needed,
                "missing": list(state.evaluation.missing),
            }
            if state.evaluation
            else None
        ),
        "execution_steps": state.execution_steps,
        "events": [
            {
                "event_type": event.event_type,
                "timestamp": event.timestamp,
                "task_id": event.task_id,
                "message": event.message,
                "metadata": event.metadata,
            }
            for event in state.events
        ],
        "metrics": vars(state.metrics),
        "created_at": state.created_at,
        "updated_at": state.updated_at,
        "completed_at": state.completed_at,
    }


def plan_state_from_dict(payload: dict[str, Any]) -> PlanState:
    evaluation_payload = payload.get("evaluation")
    evaluation = None
    if evaluation_payload:
        evaluation = GoalEvaluation(
            goal_satisfied=bool(evaluation_payload["goal_satisfied"]),
            reason=str(evaluation_payload["reason"]),
            final_answer=str(evaluation_payload.get("final_answer", "")),
            replan_needed=bool(evaluation_payload.get("replan_needed", False)),
            missing=tuple(evaluation_payload.get("missing", [])),
        )
    metric_fields = PlanMetrics.__dataclass_fields__
    metrics = PlanMetrics(
        **{
            key: value
            for key, value in payload.get("metrics", {}).items()
            if key in metric_fields
        }
    )
    return PlanState(
        goal=payload["goal"],
        tasks=[_task_from_dict(task) for task in payload.get("tasks", [])],
        assumptions=tuple(payload.get("assumptions", [])),
        plan_id=payload["plan_id"],
        version=int(payload.get("version", 1)),
        revision=int(payload.get("revision", 0)),
        status=PlanStatus(payload["status"]),
        active_task_id=payload.get("active_task_id"),
        outputs=payload.get("outputs", {}),
        final_answer=payload.get("final_answer", ""),
        evaluation=evaluation,
        execution_steps=int(payload.get("execution_steps", 0)),
        events=[PlanEvent(**event) for event in payload.get("events", [])],
        metrics=metrics,
        created_at=payload["created_at"],
        updated_at=payload["updated_at"],
        completed_at=payload.get("completed_at"),
    )


def _task_to_dict(task: PlanTask) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "description": task.description,
        "capability": task.capability.value,
        "dependencies": list(task.dependencies),
        "inputs": list(task.inputs),
        "output_key": task.output_key,
        "tool_name": task.tool_name,
        "tool_arguments": task.tool_arguments,
        "query": task.query,
        "priority": task.priority,
        "required": task.required,
        "max_retries": task.max_retries,
        "status": task.status.value,
        "attempts": task.attempts,
        "result": _result_to_dict(task.result) if task.result else None,
        "error": task.error,
        "action_id": task.action_id,
        "action_version": task.action_version,
        "approval_id": task.approval_id,
        "execution_receipt_id": task.execution_receipt_id,
        "created_at": task.created_at,
        "started_at": task.started_at,
        "completed_at": task.completed_at,
    }


def _task_from_dict(payload: dict[str, Any]) -> PlanTask:
    return PlanTask(
        task_id=payload["task_id"],
        description=payload["description"],
        capability=TaskCapability(payload["capability"]),
        dependencies=tuple(payload.get("dependencies", [])),
        inputs=tuple(payload.get("inputs", [])),
        output_key=payload.get("output_key"),
        tool_name=payload.get("tool_name"),
        tool_arguments=payload.get("tool_arguments", {}),
        query=payload.get("query"),
        priority=int(payload.get("priority", 0)),
        required=bool(payload.get("required", True)),
        max_retries=int(payload.get("max_retries", 0)),
        status=TaskStatus(payload.get("status", TaskStatus.PENDING.value)),
        attempts=int(payload.get("attempts", 0)),
        result=_result_from_dict(payload["result"]) if payload.get("result") else None,
        error=payload.get("error"),
        action_id=payload.get("action_id"),
        action_version=payload.get("action_version"),
        approval_id=payload.get("approval_id"),
        execution_receipt_id=payload.get("execution_receipt_id"),
        created_at=payload["created_at"],
        started_at=payload.get("started_at"),
        completed_at=payload.get("completed_at"),
    )


def _result_to_dict(result: TaskResult) -> dict[str, Any]:
    return {
        "task_id": result.task_id,
        "status": result.status.value,
        "output": result.output,
        "error": result.error,
        "metadata": result.metadata,
        "sources": list(result.sources),
        "duration_seconds": result.duration_seconds,
        "attempt": result.attempt,
        "retryable": result.retryable,
    }


def _result_from_dict(payload: dict[str, Any]) -> TaskResult:
    return TaskResult(
        task_id=payload["task_id"],
        status=TaskStatus(payload["status"]),
        output=payload.get("output"),
        error=payload.get("error"),
        metadata=payload.get("metadata", {}),
        sources=tuple(payload.get("sources", [])),
        duration_seconds=float(payload.get("duration_seconds", 0.0)),
        attempt=int(payload.get("attempt", 1)),
        retryable=bool(payload.get("retryable", False)),
    )
