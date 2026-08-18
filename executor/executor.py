import logging
import time

from executor.retry_policy import RetryPolicy
from executor.task_runner import TaskRunner
from planner.models import PlanState, PlanTask, TaskResult, TaskStatus, utc_now_iso


logger = logging.getLogger(__name__)


class TaskExecutor:
    def __init__(
        self,
        *,
        runner: TaskRunner,
        retry_policy: RetryPolicy,
        max_execution_steps: int,
    ) -> None:
        self.runner = runner
        self.retry_policy = retry_policy
        self.max_execution_steps = max(1, max_execution_steps)

    def execute(self, task: PlanTask, state: PlanState) -> TaskResult:
        if task.status != TaskStatus.READY:
            raise ValueError(f"Task {task.task_id!r} is not ready for execution.")

        started = time.perf_counter()
        task.status = TaskStatus.RUNNING
        task.started_at = utc_now_iso()
        state.active_task_id = task.task_id
        state.record_event("TASK STARTED", task_id=task.task_id)
        logger.info(
            "TASK STARTED task_id=%s capability=%s",
            task.task_id,
            task.capability.value,
        )

        while True:
            if state.execution_steps >= self.max_execution_steps:
                result = TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error=(
                        f"Plan reached the maximum of {self.max_execution_steps} "
                        "execution steps."
                    ),
                    attempt=task.attempts,
                )
                break

            task.attempts += 1
            state.execution_steps += 1
            result = self.runner.run(task, state)
            state.record_event(
                "TASK_ATTEMPT",
                task_id=task.task_id,
                metadata={
                    "attempt": task.attempts,
                    "status": result.status.value,
                    "retryable": result.retryable,
                },
            )
            if result.status == TaskStatus.COMPLETED:
                break
            if not self.retry_policy.should_retry(
                result,
                attempt=task.attempts,
                task_max_retries=task.max_retries,
            ):
                break
            state.record_event(
                "TASK RETRIED",
                task_id=task.task_id,
                metadata={"next_attempt": task.attempts + 1},
            )
            logger.warning("TASK RETRIED task_id=%s attempt=%s", task.task_id, task.attempts)

        task.result = result
        task.error = result.error
        task.completed_at = utc_now_iso()
        state.active_task_id = None
        state.metrics.execution_seconds += time.perf_counter() - started
        if result.status == TaskStatus.COMPLETED:
            task.status = TaskStatus.COMPLETED
            if task.output_key:
                state.outputs[task.output_key] = result.output
            state.record_event(
                "TASK COMPLETED",
                task_id=task.task_id,
                metadata={"attempts": task.attempts, "output_key": task.output_key},
            )
            logger.info("TASK COMPLETED task_id=%s attempts=%s", task.task_id, task.attempts)
        else:
            task.status = TaskStatus.FAILED
            state.record_event(
                "TASK FAILED",
                task_id=task.task_id,
                message=result.error or "task failed",
                metadata={"attempts": task.attempts},
            )
            logger.error("TASK FAILED task_id=%s attempts=%s", task.task_id, task.attempts)
        return result

