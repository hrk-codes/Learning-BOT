from planner.models import TaskResult, TaskStatus


class RetryPolicy:
    def __init__(self, max_task_retries: int) -> None:
        self.max_task_retries = max(0, max_task_retries)

    def should_retry(
        self,
        result: TaskResult,
        *,
        attempt: int,
        task_max_retries: int,
    ) -> bool:
        retry_budget = min(max(0, task_max_retries), self.max_task_retries)
        return (
            result.status == TaskStatus.FAILED
            and result.retryable
            and attempt <= retry_budget
        )

