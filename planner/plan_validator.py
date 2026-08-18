from __future__ import annotations

import re
import time
from dataclasses import dataclass

from planner.dependency_graph import find_cycle
from planner.models import PlanState, TaskCapability, TaskStatus


TASK_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
BUILT_IN_INPUTS = {"goal", "memory_context", "conversation_context"}


@dataclass(frozen=True)
class CapabilityCatalog:
    tools: frozenset[str] = frozenset()
    rag_available: bool = False
    memory_available: bool = False

    def supports(self, capability: TaskCapability, tool_name: str | None = None) -> bool:
        if capability == TaskCapability.LLM:
            return True
        if capability == TaskCapability.TOOL:
            return bool(tool_name and tool_name in self.tools)
        if capability == TaskCapability.RAG:
            return self.rag_available
        if capability == TaskCapability.MEMORY:
            return self.memory_available
        return False


class PlanValidationError(Exception):
    def __init__(self, issues: list[str]) -> None:
        self.issues = issues
        super().__init__("; ".join(issues))


class PlanValidator:
    def __init__(self, *, max_tasks: int, max_task_retries: int) -> None:
        self.max_tasks = max(1, max_tasks)
        self.max_task_retries = max(0, max_task_retries)

    def validate(self, state: PlanState, catalog: CapabilityCatalog) -> None:
        started = time.perf_counter()
        issues: list[str] = []
        active_tasks = [task for task in state.tasks if task.required]
        if not active_tasks:
            issues.append("The plan must contain at least one required task.")
        if len(active_tasks) > self.max_tasks:
            issues.append(
                f"The plan has {len(active_tasks)} required tasks; maximum is {self.max_tasks}."
            )

        ids = [task.task_id for task in state.tasks]
        known_ids = set(ids)
        if len(ids) != len(known_ids):
            issues.append("Task IDs must be unique.")

        output_producers: dict[str, str] = {}
        for task in state.tasks:
            if not TASK_ID_PATTERN.fullmatch(task.task_id):
                issues.append(
                    f"Task ID {task.task_id!r} must use lowercase letters, numbers, and underscores."
                )
            if not task.description.strip():
                issues.append(f"Task {task.task_id!r} needs a description.")
            if task.max_retries < 0 or task.max_retries > self.max_task_retries:
                issues.append(
                    f"Task {task.task_id!r} max_retries must be between 0 and "
                    f"{self.max_task_retries}."
                )
            if task.required and not catalog.supports(task.capability, task.tool_name):
                detail = (
                    f"tool {task.tool_name!r}" if task.capability == TaskCapability.TOOL
                    else task.capability.value
                )
                issues.append(f"Task {task.task_id!r} requests unavailable capability {detail}.")
            if task.capability == TaskCapability.TOOL and not task.tool_name:
                issues.append(f"Task {task.task_id!r} must name a tool.")
            if task.task_id in task.dependencies:
                issues.append(f"Task {task.task_id!r} cannot depend on itself.")
            for dependency in task.dependencies:
                if dependency not in known_ids:
                    issues.append(
                        f"Task {task.task_id!r} depends on unknown task {dependency!r}."
                    )
                dependency_task = state.get_task(dependency)
                if (
                    task.required
                    and dependency_task is not None
                    and not dependency_task.required
                    and dependency_task.status != TaskStatus.COMPLETED
                ):
                    issues.append(
                        f"Task {task.task_id!r} depends on retired incomplete task "
                        f"{dependency!r}."
                    )
            if task.output_key:
                if task.output_key in output_producers:
                    issues.append(
                        f"Output key {task.output_key!r} is produced by more than one task."
                    )
                output_producers[task.output_key] = task.task_id
            elif task.required:
                issues.append(f"Task {task.task_id!r} needs a unique output_key.")

        for task in state.tasks:
            dependency_outputs = {
                dependency.output_key
                for dependency_id in task.dependencies
                if (dependency := state.get_task(dependency_id)) is not None
                and dependency.output_key
            }
            for input_key in task.inputs:
                if input_key not in BUILT_IN_INPUTS and input_key not in dependency_outputs:
                    issues.append(
                        f"Task {task.task_id!r} input {input_key!r} is not produced by a dependency."
                    )

        cycle = find_cycle(state.tasks)
        if cycle:
            issues.append(f"Dependency cycle detected among: {', '.join(cycle)}.")

        state.metrics.validation_seconds += time.perf_counter() - started
        if issues:
            raise PlanValidationError(issues)
