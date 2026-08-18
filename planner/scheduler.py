from planner.models import PlanState, TaskStatus, utc_now_iso
from planner.plan_validator import BUILT_IN_INPUTS, CapabilityCatalog


class TaskScheduler:
    """Select ready work from a validated DAG; execution remains sequential in V1."""

    def refresh(self, state: PlanState, catalog: CapabilityCatalog) -> None:
        for task in state.tasks:
            if task.status not in {TaskStatus.PENDING, TaskStatus.READY}:
                continue
            dependencies = [state.get_task(task_id) for task_id in task.dependencies]
            if any(
                dependency is None
                or dependency.status
                in {TaskStatus.FAILED, TaskStatus.BLOCKED, TaskStatus.CANCELLED}
                for dependency in dependencies
            ):
                task.status = TaskStatus.BLOCKED
                task.error = "A required dependency did not complete successfully."
                task.completed_at = utc_now_iso()
                state.record_event(
                    "TASK_BLOCKED",
                    task_id=task.task_id,
                    message="dependency failure",
                )
                continue

            dependencies_complete = all(
                dependency is not None and dependency.status == TaskStatus.COMPLETED
                for dependency in dependencies
            )
            inputs_available = all(
                input_key in BUILT_IN_INPUTS or input_key in state.outputs
                for input_key in task.inputs
            )
            capability_available = catalog.supports(task.capability, task.tool_name)
            if dependencies_complete and inputs_available and capability_available:
                if task.status != TaskStatus.READY:
                    task.status = TaskStatus.READY
                    state.record_event("TASK_READY", task_id=task.task_id)
            else:
                task.status = TaskStatus.PENDING

    def ready_tasks(self, state: PlanState) -> list:
        indexed = {task.task_id: index for index, task in enumerate(state.tasks)}
        return sorted(
            (task for task in state.tasks if task.status == TaskStatus.READY),
            key=lambda task: (-task.priority, indexed[task.task_id]),
        )

    def cancel_remaining(self, state: PlanState) -> None:
        for task in state.tasks:
            if task.status in {TaskStatus.PENDING, TaskStatus.READY}:
                task.status = TaskStatus.CANCELLED
                task.error = "Plan execution was cancelled before this task started."
                task.completed_at = utc_now_iso()
                state.record_event("TASK_CANCELLED", task_id=task.task_id)
