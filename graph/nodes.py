from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from langgraph.types import interrupt

from approval.models import ApprovalStatus
from approval.service import ApprovalService, ApprovalServiceError
from executor.retry_policy import RetryPolicy
from llm.groq_client import GroqClientError
from planner.models import PlanState, PlanStatus, TaskStatus, utc_now_iso
from planner.planner import PlannerError
from planner.runtime import PlanningRuntime
from planner.serialization import plan_state_from_dict, plan_state_to_dict
from rag.pipeline import RagPipelineError

from graph.state import GraphAgentState


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphDependencies:
    """Runtime services reconstructed outside persistent graph state.

    Database connections, model call functions, and tool implementations are live
    runtime resources. The checkpoint stores only serializable identifiers and data,
    then a fresh app process creates these services again before resuming a thread.
    """

    planning_runtime: PlanningRuntime
    approval_service: ApprovalService | None
    approval_user_id: str
    max_task_retries: int


def build_nodes(dependencies: GraphDependencies) -> dict[str, Any]:
    """Adapt existing Stage 7/8 subsystems into focused graph nodes."""

    def plan(state: GraphAgentState) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            plan_state = dependencies.planning_runtime.planner.create_plan(
                goal=state["goal"],
                conversation_context=state.get("conversation_context", []),
                memory_context=state.get("memory_context"),
                knowledge_base=state.get("knowledge_base", {"available": False}),
                active_tools=dependencies.planning_runtime.active_tools,
            )
        except (GroqClientError, PlannerError) as exc:
            return _workflow_error(state, "planner", started, str(exc))

        plan_state.status = PlanStatus.RUNNING
        plan_state.record_event("GRAPH PLAN STARTED", metadata={"run_id": state["run_id"]})
        return _plan_update(
            state,
            plan_state,
            node="planner",
            started=started,
            status="completed",
            next_node="task_router",
            details={"task_count": len(plan_state.tasks)},
            updates={"status": "running", "error": None},
        )

    def task_router(state: GraphAgentState) -> dict[str, Any]:
        started = time.perf_counter()
        plan_state = _plan(state)
        if plan_state.status in {PlanStatus.COMPLETED, PlanStatus.FAILED, PlanStatus.CANCELLED}:
            return _plan_update(
                state,
                plan_state,
                node="task_router",
                started=started,
                status="completed",
                next_node="finalize",
                details={"plan_status": plan_state.status.value},
                updates={"next_task_id": None},
            )

        dependencies.planning_runtime.scheduler.refresh(
            plan_state, dependencies.planning_runtime.catalog
        )
        ready = dependencies.planning_runtime.scheduler.ready_tasks(plan_state)
        next_task_id = ready[0].task_id if ready else None
        return _plan_update(
            state,
            plan_state,
            node="task_router",
            started=started,
            status="completed",
            next_node="execute_task" if next_task_id else "evaluate",
            details={"ready_task_count": len(ready), "next_task_id": next_task_id},
            updates={"next_task_id": next_task_id},
        )

    def execute_task(state: GraphAgentState) -> dict[str, Any]:
        started = time.perf_counter()
        plan_state = _plan(state)
        task_id = state.get("next_task_id")
        task = plan_state.get_task(task_id) if task_id else None
        if task is None:
            return _workflow_error(
                state, "execute_task", started, "The graph had no schedulable task."
            )
        try:
            result = dependencies.planning_runtime.executor.execute(task, plan_state)
        except (ApprovalServiceError, GroqClientError, RagPipelineError, ValueError) as exc:
            task.status = TaskStatus.FAILED
            task.error = str(exc)
            task.completed_at = utc_now_iso()
            plan_state.record_event("TASK FAILED", task_id=task.task_id, message=str(exc))
            result_payload = {
                "task_id": task.task_id,
                "status": TaskStatus.FAILED.value,
                "retryable": False,
                "error": str(exc),
            }
            return _plan_update(
                state,
                plan_state,
                node="execute_task",
                started=started,
                status="failed",
                next_node="task_router",
                details={"task_id": task.task_id, "error_type": type(exc).__name__},
                updates={"last_result": result_payload, "next_task_id": task.task_id},
            )

        result_payload = {
            "task_id": task.task_id,
            "status": result.status.value,
            "retryable": result.retryable,
            "error": result.error,
            "attempt": result.attempt,
        }
        next_node = "approval" if result.status == TaskStatus.WAITING_FOR_APPROVAL else "task_router"
        if result.status == TaskStatus.WAITING_FOR_APPROVAL:
            # The task runner has prepared a version-locked proposal but has not
            # invoked the side effect. Persist this lifecycle state before the next
            # node interrupts so a restart can render the exact same review.
            plan_state.status = PlanStatus.WAITING_FOR_APPROVAL
            plan_state.final_answer = (
                "Execution is paused at a human approval gate. Review the frozen "
                "action proposal before continuing."
            )
            plan_state.record_event("GRAPH PAUSED FOR APPROVAL", task_id=task.task_id)
        return _plan_update(
            state,
            plan_state,
            node="execute_task",
            started=started,
            status="completed" if result.status == TaskStatus.COMPLETED else result.status.value,
            next_node=next_node,
            details={"task_id": task.task_id, "task_status": task.status.value, "attempts": task.attempts},
            updates={"last_result": result_payload, "next_task_id": task.task_id},
        )

    def retry_task(state: GraphAgentState) -> dict[str, Any]:
        started = time.perf_counter()
        plan_state = _plan(state)
        task = plan_state.get_task(state.get("next_task_id"))
        if task is None:
            return _workflow_error(state, "retry_task", started, "Retry target is unavailable.")
        decision = RetryPolicy(dependencies.max_task_retries).should_retry(
            task.result,
            attempt=task.attempts,
            task_max_retries=task.max_retries,
        ) if task.result else False
        if not decision:
            return _plan_update(
                state,
                plan_state,
                node="retry_task",
                started=started,
                status="skipped",
                next_node="task_router",
                details={"task_id": task.task_id, "reason": "retry budget exhausted"},
            )
        task.status = TaskStatus.READY
        task.error = None
        plan_state.record_event(
            "GRAPH RETRY SCHEDULED",
            task_id=task.task_id,
            metadata={"next_attempt": task.attempts + 1},
        )
        return _plan_update(
            state,
            plan_state,
            node="retry_task",
            started=started,
            status="completed",
            next_node="execute_task",
            details={"task_id": task.task_id, "next_attempt": task.attempts + 1},
            updates={"graph_retry_count": int(state.get("graph_retry_count", 0)) + 1},
        )

    def approval(state: GraphAgentState) -> dict[str, Any]:
        started = time.perf_counter()
        plan_state = _plan(state)
        task = next(
            (
                item
                for item in plan_state.tasks
                if item.status == TaskStatus.WAITING_FOR_APPROVAL
            ),
            None,
        )
        if task is None or not task.action_id or not task.action_version or not task.approval_id:
            return _workflow_error(
                state, "approval", started, "The approval-bound task is incomplete."
            )
        if dependencies.approval_service is None:
            return _workflow_error(
                state, "approval", started, "Approval storage is unavailable; execution is blocked."
            )
        try:
            request = dependencies.approval_service.get_approval(task.approval_id)
            proposal = dependencies.approval_service.get_action(
                task.action_id, task.action_version
            )
        except ApprovalServiceError as exc:
            return _workflow_error(state, "approval", started, str(exc))

        approval_payload = {
            "kind": "approval_required",
            "thread_id": state["thread_id"],
            "approval_id": request.approval_id,
            "action_id": proposal.action_id,
            "action_version": proposal.version,
            "task_id": task.task_id,
            "tool_name": proposal.tool_name,
            "risk_level": proposal.risk_level.value,
            "preview": proposal.preview,
            "expires_at": request.expires_at,
        }
        if request.status == ApprovalStatus.PENDING:
            # `interrupt` persists graph state before handing control to Streamlit.
            # This node restarts on resume, so the existing approval service remains
            # the source of truth for the human decision and action version.
            interrupt(approval_payload)

        plan_state.status = PlanStatus.RUNNING
        plan_state.record_event(
            "GRAPH APPROVAL RESUMED",
            task_id=task.task_id,
            metadata={"approval_status": request.status.value},
        )
        return _plan_update(
            state,
            plan_state,
            node="approval",
            started=started,
            status="resumed",
            next_node="execute_task",
            details={"task_id": task.task_id, "approval_status": request.status.value},
            updates={"approval": approval_payload, "status": "running"},
        )

    def evaluate(state: GraphAgentState) -> dict[str, Any]:
        started = time.perf_counter()
        plan_state = _plan(state)
        plan_state.status = PlanStatus.EVALUATING
        try:
            plan_state.evaluation = dependencies.planning_runtime.evaluator.evaluate(plan_state)
        except GroqClientError as exc:
            return _workflow_error(state, "evaluate", started, str(exc), plan_state)
        evaluation = plan_state.evaluation
        plan_state.record_event(
            "GOAL EVALUATED",
            message=evaluation.reason,
            metadata={
                "satisfied": evaluation.goal_satisfied,
                "replan_needed": evaluation.replan_needed,
            },
        )
        next_node = "finalize" if evaluation.goal_satisfied else "replan"
        return _plan_update(
            state,
            plan_state,
            node="evaluate",
            started=started,
            status="completed",
            next_node=next_node,
            details={"goal_satisfied": evaluation.goal_satisfied, "replan_needed": evaluation.replan_needed},
        )

    def replan(state: GraphAgentState) -> dict[str, Any]:
        started = time.perf_counter()
        plan_state = _plan(state)
        evaluation = plan_state.evaluation
        if evaluation is None:
            return _workflow_error(state, "replan", started, "No evaluation was available for replanning.")
        try:
            revised = dependencies.planning_runtime.replanner.replan(
                plan_state,
                reason=evaluation.reason,
                active_tools=dependencies.planning_runtime.active_tools,
            )
        except PlannerError as exc:
            plan_state.status = PlanStatus.FAILED
            plan_state.final_answer = f"The plan stopped safely because replanning failed: {exc}"
            plan_state.completed_at = utc_now_iso()
            plan_state.record_event("PLAN FAILED", message=str(exc))
            return _plan_update(
                state,
                plan_state,
                node="replan",
                started=started,
                status="failed",
                next_node="finalize",
                details={"error_type": type(exc).__name__},
                updates={"status": "failed", "error": str(exc)},
            )
        revised.status = PlanStatus.RUNNING
        return _plan_update(
            state,
            revised,
            node="replan",
            started=started,
            status="completed",
            next_node="task_router",
            details={"revision": revised.revision, "task_count": len(revised.tasks)},
        )

    def finalize(state: GraphAgentState) -> dict[str, Any]:
        started = time.perf_counter()
        plan_state = _plan(state)
        if plan_state.status != PlanStatus.FAILED:
            evaluation = plan_state.evaluation
            if evaluation and evaluation.goal_satisfied:
                plan_state.status = PlanStatus.COMPLETED
                plan_state.final_answer = _append_sources(evaluation.final_answer, plan_state)
                plan_state.record_event("PLAN COMPLETED")
            elif not plan_state.final_answer:
                plan_state.status = PlanStatus.FAILED
                plan_state.final_answer = (
                    "The plan finished its available work, but the original goal was not satisfied."
                )
                plan_state.record_event("PLAN FAILED", message="goal evaluation incomplete")
        plan_state.completed_at = plan_state.completed_at or utc_now_iso()
        status = plan_state.status.value
        return _plan_update(
            state,
            plan_state,
            node="finalize",
            started=started,
            status=status,
            next_node=None,
            details={"plan_status": status},
            updates={
                "final_answer": plan_state.final_answer,
                "status": status,
                "completed_at": plan_state.completed_at,
                "next_task_id": None,
            },
        )

    return {
        "planner": plan,
        "task_router": task_router,
        "execute_task": execute_task,
        "retry_task": retry_task,
        "approval": approval,
        "evaluate": evaluate,
        "replan": replan,
        "finalize": finalize,
    }


def _plan(state: GraphAgentState) -> PlanState:
    return plan_state_from_dict(state["plan_state"])


def _workflow_error(
    state: GraphAgentState,
    node: str,
    started: float,
    message: str,
    plan_state: PlanState | None = None,
) -> dict[str, Any]:
    plan = plan_state or _plan(state)
    plan.status = PlanStatus.FAILED
    plan.final_answer = f"The graph stopped safely at {node}: {message}"
    plan.completed_at = utc_now_iso()
    plan.record_event("GRAPH NODE FAILED", message=message, metadata={"node": node})
    logger.error("GRAPH NODE FAILED node=%s error=%s", node, message)
    return _plan_update(
        state,
        plan,
        node=node,
        started=started,
        status="failed",
        next_node="finalize",
        details={"error_type": "workflow_error"},
        updates={"status": "failed", "error": message, "final_answer": plan.final_answer},
    )


def _plan_update(
    state: GraphAgentState,
    plan_state: PlanState,
    *,
    node: str,
    started: float,
    status: str,
    next_node: str | None,
    details: dict[str, Any],
    updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = utc_now_iso()
    payload: dict[str, Any] = {
        "plan_state": plan_state_to_dict(plan_state),
        "node_trace": [
            {
                "node": node,
                "status": status,
                "next_node": next_node,
                "started_at": now,
                "duration_seconds": round(time.perf_counter() - started, 4),
                "details": details,
            }
        ],
    }
    if updates:
        payload.update(updates)
    return payload


def _append_sources(answer: str, state: PlanState) -> str:
    if not answer or "\nSources:\n" in answer:
        return answer
    references: list[str] = []
    seen: set[tuple[object, object]] = set()
    for task in state.tasks:
        if not task.result:
            continue
        for source in task.result.sources:
            identity = (source.get("filename"), source.get("page_number"))
            if identity in seen:
                continue
            seen.add(identity)
            filename = source.get("filename") or "Unknown source"
            page = source.get("page_number")
            references.append(f"- {filename}" + (f" - page {page}" if page else ""))
    return answer.rstrip() + ("\n\nSources:\n" + "\n".join(references) if references else "")
