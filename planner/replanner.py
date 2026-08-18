from __future__ import annotations

import copy
import json
import logging
from collections.abc import Callable
from typing import Any

from planner.models import PlanState, PlanStatus, TaskStatus
from planner.plan_validator import CapabilityCatalog, PlanValidationError, PlanValidator
from planner.planner import PlannerError, parse_plan_payload
from prompts.replanner_prompt import REPLANNER_SYSTEM_PROMPT


logger = logging.getLogger(__name__)
LLMFn = Callable[[list[dict[str, str]]], str]


class Replanner:
    def __init__(
        self,
        *,
        llm_fn: LLMFn,
        validator: PlanValidator,
        catalog: CapabilityCatalog,
        max_repair_attempts: int,
        default_task_retries: int,
    ) -> None:
        self.llm_fn = llm_fn
        self.validator = validator
        self.catalog = catalog
        self.max_repair_attempts = max(0, max_repair_attempts)
        self.default_task_retries = max(0, default_task_retries)

    def replan(
        self,
        state: PlanState,
        *,
        reason: str,
        active_tools: list[dict[str, Any]],
    ) -> PlanState:
        messages = self._messages(state, reason, active_tools)
        last_error = "Replanner did not return replacement tasks."
        for attempt in range(self.max_repair_attempts + 1):
            raw = self.llm_fn(messages)
            candidate = copy.deepcopy(state)
            candidate.metrics.replan_calls += 1
            try:
                additions = parse_plan_payload(
                    raw,
                    goal=state.goal,
                    default_task_retries=self.default_task_retries,
                )
                self._retire_unfinished(candidate)
                candidate.tasks.extend(additions.tasks)
                candidate.assumptions = tuple(
                    dict.fromkeys((*candidate.assumptions, *additions.assumptions))
                )
                candidate.revision += 1
                candidate.status = PlanStatus.VALIDATED
                self.validator.validate(candidate, self.catalog)
                candidate.record_event(
                    "PLAN REVISED",
                    message=reason,
                    metadata={
                        "revision": candidate.revision,
                        "new_task_count": len(additions.tasks),
                    },
                )
                logger.info(
                    "PLAN REVISED revision=%s new_task_count=%s",
                    candidate.revision,
                    len(additions.tasks),
                )
                return candidate
            except (PlannerError, PlanValidationError) as exc:
                last_error = str(exc)
                if attempt >= self.max_repair_attempts:
                    break
                messages = [
                    {"role": "system", "content": REPLANNER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "goal": state.goal,
                                "repair_issues": (
                                    exc.issues
                                    if isinstance(exc, PlanValidationError)
                                    else [str(exc)]
                                ),
                                "invalid_revision": raw[:12000],
                                "existing_task_ids": [
                                    task.task_id for task in state.tasks
                                ],
                            },
                            ensure_ascii=True,
                        ),
                    },
                ]
        raise PlannerError(f"Replanning failed safely: {last_error}")

    def _messages(
        self,
        state: PlanState,
        reason: str,
        active_tools: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        payload = {
            "goal": state.goal,
            "revision_reason": reason,
            "current_revision": state.revision,
            "tasks": [task.public_summary() for task in state.tasks],
            "completed_output_keys": sorted(state.outputs),
            "available_capabilities": {
                "llm": True,
                "memory": self.catalog.memory_available,
                "rag": self.catalog.rag_available,
                "tools": active_tools,
            },
            "constraints": {
                "max_tasks": self.validator.max_tasks,
                "max_task_retries": self.validator.max_task_retries,
            },
        }
        return [
            {"role": "system", "content": REPLANNER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=True, default=str)[:24000],
            },
        ]

    @staticmethod
    def _retire_unfinished(state: PlanState) -> None:
        for task in state.tasks:
            if task.status != TaskStatus.COMPLETED:
                task.required = False
                if task.status in {TaskStatus.PENDING, TaskStatus.READY}:
                    task.status = TaskStatus.CANCELLED
                    task.error = "Superseded by a revised plan."

