from __future__ import annotations

from graph.state import GraphAgentState
from planner.models import PlanStatus, TaskStatus
from planner.serialization import plan_state_from_dict


def route_after_task_router(state: GraphAgentState) -> str:
    """Choose the next graph edge from explicit planner state, not hidden loops."""

    plan = plan_state_from_dict(state["plan_state"])
    if plan.status in {PlanStatus.COMPLETED, PlanStatus.FAILED, PlanStatus.CANCELLED}:
        return "finalize"
    if state.get("next_task_id"):
        return "execute_task"
    return "evaluate"


def route_after_execution(state: GraphAgentState) -> str:
    """Route task outcomes into approval, a bounded retry loop, or more scheduling."""

    plan = plan_state_from_dict(state["plan_state"])
    task_id = state.get("next_task_id")
    task = plan.get_task(task_id) if task_id else None
    if task and task.status == TaskStatus.WAITING_FOR_APPROVAL:
        return "approval"
    if _can_retry(state, task):
        return "retry_task"
    return "task_router"


def route_after_evaluation(state: GraphAgentState) -> str:
    """Make completion, replanning, and safe failure visible as graph branches."""

    plan = plan_state_from_dict(state["plan_state"])
    evaluation = plan.evaluation
    if evaluation and evaluation.goal_satisfied:
        return "finalize"
    if (
        evaluation
        and evaluation.replan_needed
        and plan.revision < int(state.get("max_plan_revisions", 0))
    ):
        return "replan"
    return "finalize"


def _can_retry(state: GraphAgentState, task) -> bool:
    result = state.get("last_result") or {}
    if task is None or not result.get("retryable"):
        return False
    # A future registry entry could be consequential even when its name is not
    # known here. Keep graph-level retries away from every tool; Stage 8 remains
    # the authority for exact-action idempotency and retry policy at that boundary.
    if task.capability.value == "tool":
        return False
    return task.attempts <= task.max_retries
