import logging
import time

from executor.retry_policy import RetryPolicy
from executor.task_runner import TaskRunner
from approval.gate import GateStatus
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
        if task.status not in {
            TaskStatus.READY,
            TaskStatus.WAITING_FOR_APPROVAL,
            TaskStatus.APPROVED,
        }:
            raise ValueError(f"Task {task.task_id!r} is not ready for execution.")

        started = time.perf_counter()
        gate = self.runner.check_approval(task, state)
        if gate.status != GateStatus.PROCEED:
            return self._handle_gate_stop(task, state, gate, started)

        task.status = (
            TaskStatus.APPROVED if gate.approved_action else TaskStatus.RUNNING
        )
        task.started_at = task.started_at or utc_now_iso()
        state.active_task_id = task.task_id
        state.record_event(
            "TASK APPROVED" if gate.approved_action else "TASK STARTED",
            task_id=task.task_id,
            metadata={
                "action_id": task.action_id,
                "action_version": task.action_version,
            },
        )
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
            if gate.approved_action:
                task.status = TaskStatus.EXECUTING
                state.record_event(
                    "ACTION EXECUTING",
                    task_id=task.task_id,
                    metadata={"action_id": task.action_id},
                )
            result = self.runner.run(
                task,
                state,
                approved_action=gate.approved_action,
            )
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

    def _handle_gate_stop(self, task, state, gate, started) -> TaskResult:
        status_map = {
            GateStatus.WAITING: TaskStatus.WAITING_FOR_APPROVAL,
            GateStatus.DENIED: TaskStatus.DENIED,
            GateStatus.CANCELLED: TaskStatus.CANCELLED,
            GateStatus.EXPIRED: TaskStatus.EXPIRED,
            GateStatus.FAILED: TaskStatus.FAILED,
        }
        task_status = status_map[gate.status]
        task.status = task_status
        task.error = gate.message
        state.active_task_id = None
        event_name = {
            GateStatus.WAITING: "APPROVAL REQUESTED",
            GateStatus.DENIED: "ACTION DENIED",
            GateStatus.CANCELLED: "ACTION CANCELLED",
            GateStatus.EXPIRED: "APPROVAL EXPIRED",
            GateStatus.FAILED: "APPROVAL FAILED CLOSED",
        }[gate.status]
        state.record_event(
            event_name,
            task_id=task.task_id,
            message=gate.message,
            metadata=gate.metadata,
        )
        result = TaskResult(
            task_id=task.task_id,
            status=task_status,
            error=gate.message,
            metadata=gate.metadata,
            duration_seconds=time.perf_counter() - started,
            attempt=task.attempts,
        )
        if gate.status != GateStatus.WAITING:
            task.result = result
            task.completed_at = utc_now_iso()
        logger.info(
            "APPROVAL GATE task_id=%s status=%s",
            task.task_id,
            gate.status.value,
        )
        return result
