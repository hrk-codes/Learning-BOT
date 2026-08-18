from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

from planner.models import PlanState, PlanStatus, PlanTask, TaskCapability
from planner.plan_validator import CapabilityCatalog, PlanValidationError, PlanValidator
from prompts.planner_prompt import PLANNER_SYSTEM_PROMPT, PLAN_REPAIR_SYSTEM_PROMPT


logger = logging.getLogger(__name__)
LLMFn = Callable[[list[dict[str, str]]], str]


class PlannerError(Exception):
    """Raised when a valid bounded plan cannot be produced."""


class Planner:
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

    def create_plan(
        self,
        *,
        goal: str,
        conversation_context: list[dict[str, str]],
        memory_context: dict[str, Any] | None,
        knowledge_base: dict[str, Any],
        active_tools: list[dict[str, Any]],
    ) -> PlanState:
        started = time.perf_counter()
        calls = 0
        messages = self._build_messages(
            goal=goal,
            conversation_context=conversation_context,
            memory_context=memory_context,
            knowledge_base=knowledge_base,
            active_tools=active_tools,
        )
        last_error = "Planner did not return a plan."

        for attempt in range(self.max_repair_attempts + 1):
            calls += 1
            logger.info("PLAN CREATED attempt=%s", attempt + 1)
            raw = self.llm_fn(messages)
            try:
                state = parse_plan_payload(
                    raw,
                    goal=goal,
                    default_task_retries=self.default_task_retries,
                )
                self.validator.validate(state, self.catalog)
                state.status = PlanStatus.VALIDATED
                state.metrics.planner_calls = calls
                state.metrics.planning_seconds = time.perf_counter() - started
                state.record_event(
                    "PLAN_VALIDATED",
                    metadata={"task_count": len(state.tasks), "attempt": attempt + 1},
                )
                logger.info("PLAN VALIDATED task_count=%s", len(state.tasks))
                return state
            except (PlannerError, PlanValidationError) as exc:
                last_error = str(exc)
                logger.warning("PLAN REJECTED attempt=%s reason=%s", attempt + 1, last_error)
                if attempt >= self.max_repair_attempts:
                    break
                messages = [
                    {"role": "system", "content": PLAN_REPAIR_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "goal": goal,
                                "available_capabilities": _capability_payload(
                                    self.catalog, active_tools
                                ),
                                "validator_issues": _issues(exc),
                                "invalid_plan": raw[:12000],
                            },
                            ensure_ascii=True,
                        ),
                    },
                ]

        raise PlannerError(
            f"The planner could not produce a valid plan after {calls} attempt(s): {last_error}"
        )

    def _build_messages(
        self,
        *,
        goal: str,
        conversation_context: list[dict[str, str]],
        memory_context: dict[str, Any] | None,
        knowledge_base: dict[str, Any],
        active_tools: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        request = {
            "goal": goal,
            "recent_conversation": conversation_context[-6:],
            "relevant_long_term_memory": memory_context
            or {"available": False, "records": []},
            "knowledge_base": knowledge_base,
            "available_capabilities": _capability_payload(self.catalog, active_tools),
            "constraints": {
                "max_tasks": self.validator.max_tasks,
                "max_task_retries": self.validator.max_task_retries,
            },
        }
        return [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Create the execution plan for this runtime state:\n"
                + json.dumps(request, ensure_ascii=True, indent=2),
            },
        ]


def parse_plan_payload(
    raw: str,
    *,
    goal: str,
    default_task_retries: int,
) -> PlanState:
    payload = _parse_json_object(raw)
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list):
        raise PlannerError("Plan field 'tasks' must be a list.")
    assumptions = payload.get("assumptions", [])
    if not isinstance(assumptions, list) or not all(
        isinstance(value, str) for value in assumptions
    ):
        raise PlannerError("Plan field 'assumptions' must be a list of strings.")

    tasks = [
        _parse_task(item, default_task_retries=default_task_retries)
        for item in raw_tasks
    ]
    return PlanState(goal=goal, tasks=tasks, assumptions=tuple(assumptions))


def _parse_task(raw: object, *, default_task_retries: int) -> PlanTask:
    if not isinstance(raw, dict):
        raise PlannerError("Every task must be a JSON object.")
    task_id = raw.get("id")
    description = raw.get("description")
    capability_value = raw.get("capability")
    if not isinstance(task_id, str) or not isinstance(description, str):
        raise PlannerError("Every task requires string id and description fields.")
    try:
        capability = TaskCapability(capability_value)
    except ValueError as exc:
        raise PlannerError(f"Task {task_id!r} has unknown capability {capability_value!r}.") from exc

    dependencies = _string_tuple(raw.get("dependencies", []), "dependencies", task_id)
    inputs = _string_tuple(raw.get("inputs", []), "inputs", task_id)
    output_key = raw.get("output_key")
    if output_key is not None and not isinstance(output_key, str):
        raise PlannerError(f"Task {task_id!r} output_key must be a string or null.")
    tool_name = raw.get("tool_name")
    if tool_name is not None and not isinstance(tool_name, str):
        raise PlannerError(f"Task {task_id!r} tool_name must be a string or null.")
    tool_arguments = raw.get("tool_arguments", {})
    if not isinstance(tool_arguments, dict):
        raise PlannerError(f"Task {task_id!r} tool_arguments must be an object.")
    query = raw.get("query")
    if query is not None and not isinstance(query, str):
        raise PlannerError(f"Task {task_id!r} query must be a string or null.")

    return PlanTask(
        task_id=task_id,
        description=description,
        capability=capability,
        dependencies=dependencies,
        inputs=inputs,
        output_key=output_key,
        tool_name=tool_name,
        tool_arguments=tool_arguments,
        query=query,
        priority=_integer(raw.get("priority", 0), "priority", task_id),
        required=_boolean(raw.get("required", True), "required", task_id),
        max_retries=_integer(
            raw.get("max_retries", default_task_retries), "max_retries", task_id
        ),
    )


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise PlannerError("Planner output was not valid JSON.") from exc
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError as nested_exc:
            raise PlannerError("Planner output was not valid JSON.") from nested_exc
    if not isinstance(payload, dict):
        raise PlannerError("Planner output must be one JSON object.")
    return payload


def _string_tuple(value: object, field: str, task_id: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise PlannerError(f"Task {task_id!r} field {field!r} must be a list of strings.")
    return tuple(value)


def _integer(value: object, field: str, task_id: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlannerError(f"Task {task_id!r} field {field!r} must be an integer.")
    return value


def _boolean(value: object, field: str, task_id: str) -> bool:
    if not isinstance(value, bool):
        raise PlannerError(f"Task {task_id!r} field {field!r} must be a boolean.")
    return value


def _capability_payload(
    catalog: CapabilityCatalog, active_tools: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "llm": {"available": True},
        "memory": {"available": catalog.memory_available},
        "rag": {"available": catalog.rag_available},
        "tools": active_tools,
    }


def _issues(exc: Exception) -> list[str]:
    if isinstance(exc, PlanValidationError):
        return exc.issues
    return [str(exc)]
