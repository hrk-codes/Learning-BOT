from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from executor.executor import TaskExecutor
from planner.goal_evaluator import GoalEvaluator
from planner.models import PlanState, PlanStatus, TaskStatus, utc_now_iso
from planner.plan_validator import CapabilityCatalog
from planner.planner import Planner, PlannerError
from planner.replanner import Replanner
from planner.scheduler import TaskScheduler


logger = logging.getLogger(__name__)
StatusCallback = Callable[[PlanState], None]
CancellationCheck = Callable[[], bool]


class PlanningRuntime:
    def __init__(
        self,
        *,
        planner: Planner,
        scheduler: TaskScheduler,
        executor: TaskExecutor,
        evaluator: GoalEvaluator,
        replanner: Replanner,
        catalog: CapabilityCatalog,
        max_plan_revisions: int,
        active_tools: list[dict[str, Any]],
    ) -> None:
        self.planner = planner
        self.scheduler = scheduler
        self.executor = executor
        self.evaluator = evaluator
        self.replanner = replanner
        self.catalog = catalog
        self.max_plan_revisions = max(0, max_plan_revisions)
        self.active_tools = active_tools

    def run(
        self,
        *,
        goal: str,
        conversation_context: list[dict[str, str]],
        memory_context: dict[str, Any] | None,
        knowledge_base: dict[str, Any],
        status_callback: StatusCallback | None = None,
        cancellation_check: CancellationCheck | None = None,
    ) -> PlanState:
        state = self.planner.create_plan(
            goal=goal,
            conversation_context=conversation_context,
            memory_context=memory_context,
            knowledge_base=knowledge_base,
            active_tools=self.active_tools,
        )
        state.status = PlanStatus.RUNNING
        state.record_event("PLAN STARTED", metadata={"revision": state.revision})
        _notify(status_callback, state)

        while state.status == PlanStatus.RUNNING:
            if cancellation_check is not None and cancellation_check():
                self.scheduler.cancel_remaining(state)
                state.status = PlanStatus.CANCELLED
                state.final_answer = "Plan execution was cancelled. No new tasks were started."
                state.completed_at = utc_now_iso()
                state.record_event("PLAN CANCELLED")
                _notify(status_callback, state)
                break

            self.scheduler.refresh(state, self.catalog)
            ready = self.scheduler.ready_tasks(state)
            _notify(status_callback, state)
            if ready:
                # V1 executes one ready task at a time. Keeping READY distinct from
                # RUNNING preserves a clean path to safe parallelism later.
                state.active_task_id = ready[0].task_id
                _notify(status_callback, state)
                self.executor.execute(ready[0], state)
                _notify(status_callback, state)
                continue

            state.status = PlanStatus.EVALUATING
            state.record_event("GOAL EVALUATION STARTED")
            evaluation = self.evaluator.evaluate(state)
            state.evaluation = evaluation
            state.record_event(
                "GOAL EVALUATED",
                message=evaluation.reason,
                metadata={
                    "satisfied": evaluation.goal_satisfied,
                    "replan_needed": evaluation.replan_needed,
                },
            )

            if evaluation.goal_satisfied:
                state.status = PlanStatus.COMPLETED
                state.final_answer = _append_sources(evaluation.final_answer, state)
                state.completed_at = utc_now_iso()
                state.record_event("PLAN COMPLETED")
                _notify(status_callback, state)
                break

            if evaluation.replan_needed and state.revision < self.max_plan_revisions:
                try:
                    state = self.replanner.replan(
                        state,
                        reason=evaluation.reason,
                        active_tools=self.active_tools,
                    )
                except PlannerError as exc:
                    state.status = PlanStatus.FAILED
                    state.final_answer = f"The plan stopped safely because replanning failed: {exc}"
                    state.completed_at = utc_now_iso()
                    state.record_event("PLAN FAILED", message=str(exc))
                    _notify(status_callback, state)
                    break
                state.status = PlanStatus.RUNNING
                _notify(status_callback, state)
                continue

            state.status = PlanStatus.FAILED
            missing = ", ".join(evaluation.missing)
            state.final_answer = evaluation.final_answer or (
                "The plan finished its available work, but the original goal was not "
                "satisfied."
                + (f" Missing: {missing}." if missing else "")
            )
            state.completed_at = utc_now_iso()
            state.record_event("PLAN FAILED", message=evaluation.reason)
            _notify(status_callback, state)
            break

        logger.info(
            "PLAN END status=%s revision=%s steps=%s",
            state.status.value,
            state.revision,
            state.execution_steps,
        )
        return state


def _notify(callback: StatusCallback | None, state: PlanState) -> None:
    if callback is not None:
        callback(state)


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
    if not references:
        return answer
    return answer.rstrip() + "\n\nSources:\n" + "\n".join(references)
