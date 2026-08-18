from collections import deque

from planner.models import PlanTask


def find_cycle(tasks: list[PlanTask]) -> list[str]:
    """Return one dependency cycle, or an empty list for a valid DAG."""
    dependencies = {task.task_id: set(task.dependencies) for task in tasks}
    indegree = {task_id: len(values) for task_id, values in dependencies.items()}
    dependents: dict[str, list[str]] = {task_id: [] for task_id in dependencies}
    for task_id, task_dependencies in dependencies.items():
        for dependency in task_dependencies:
            if dependency in dependents:
                dependents[dependency].append(task_id)

    ready = deque(task_id for task_id, degree in indegree.items() if degree == 0)
    visited: list[str] = []
    while ready:
        task_id = ready.popleft()
        visited.append(task_id)
        for dependent in dependents[task_id]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)

    if len(visited) == len(tasks):
        return []
    return [task_id for task_id, degree in indegree.items() if degree > 0]


def downstream_task_ids(tasks: list[PlanTask], failed_task_id: str) -> set[str]:
    dependents: dict[str, set[str]] = {task.task_id: set() for task in tasks}
    for task in tasks:
        for dependency in task.dependencies:
            dependents.setdefault(dependency, set()).add(task.task_id)

    downstream: set[str] = set()
    queue = deque(dependents.get(failed_task_id, set()))
    while queue:
        task_id = queue.popleft()
        if task_id in downstream:
            continue
        downstream.add(task_id)
        queue.extend(dependents.get(task_id, set()))
    return downstream

