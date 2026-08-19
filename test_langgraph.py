from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from approval.models import ApprovalStatus
from graph.graph import build_agent_graph
from graph.nodes import GraphDependencies
from planner.models import (
    GoalEvaluation,
    PlanState,
    PlanStatus,
    PlanTask,
    TaskCapability,
    TaskResult,
    TaskStatus,
)
from planner.serialization import plan_state_to_dict


def test_linear_graph_moves_serializable_state_between_nodes(tmp_path: Path) -> None:
    graph = build_test_graph(tmp_path, executor=CompletingExecutor())
    result = graph.invoke(initial_state("linear"), graph_config("linear"))

    assert result["status"] == "completed"
    assert result["final_answer"] == "Completed safely."
    assert node_names(result) == ["planner", "task_router", "execute_task", "task_router", "evaluate", "finalize"]


def test_retryable_read_only_task_uses_explicit_graph_loop(tmp_path: Path) -> None:
    executor = RetryOnceExecutor()
    graph = build_test_graph(tmp_path, executor=executor)
    result = graph.invoke(initial_state("retry"), graph_config("retry"))

    assert result["status"] == "completed"
    assert result["graph_retry_count"] == 1
    assert executor.calls == 2
    assert "retry_task" in node_names(result)


def test_permanent_failure_does_not_retry_or_loop_forever(tmp_path: Path) -> None:
    executor = PermanentFailureExecutor()
    graph = build_test_graph(tmp_path, executor=executor, satisfied=False)
    result = graph.invoke(initial_state("permanent"), graph_config("permanent"))

    assert result["status"] == "failed"
    assert result["graph_retry_count"] == 0
    assert executor.calls == 1
    assert node_names(result)[-1] == "finalize"


def test_approval_interrupt_survives_new_graph_instance_and_resumes(tmp_path: Path) -> None:
    approval = FakeApprovalService()
    checkpoint_path = tmp_path / "graph-checkpoints.db"
    graph = build_test_graph(
        tmp_path,
        executor=ApprovalExecutor(approval),
        approval_service=approval,
        checkpoint_path=checkpoint_path,
    )
    config = graph_config("approval-thread")

    paused = graph.invoke(initial_state("approval"), config)
    assert "__interrupt__" in paused
    assert graph.get_state(config).next == ("approval",)

    # A restart creates a new graph/checkpointer object but points to the same
    # durable SQLite file. The same thread ID restores the paused state.
    approval.request.status = ApprovalStatus.APPROVED
    restarted_graph = build_test_graph(
        tmp_path,
        executor=ApprovalExecutor(approval),
        approval_service=approval,
        checkpoint_path=checkpoint_path,
    )
    resumed = restarted_graph.invoke(Command(resume={"continue": True}), config)

    assert resumed["status"] == "completed"
    assert restarted_graph.get_state(config).next == ()
    assert "approval" in node_names(resumed)


def test_checkpoint_update_keeps_a_revised_action_paused_for_a_new_decision(
    tmp_path: Path,
) -> None:
    approval = FakeApprovalService()
    graph = build_test_graph(
        tmp_path, executor=ApprovalExecutor(approval), approval_service=approval
    )
    config = graph_config("edited-approval")

    graph.invoke(initial_state("edited-approval"), config)
    paused = graph.get_state(config)
    # Editing a Stage 8 proposal updates serializable plan data. It must not make
    # the graph silently pass the LangGraph interrupt before a new decision.
    graph.update_state(config, {"plan_state": paused.values["plan_state"]})

    assert graph.get_state(config).next == ("approval",)


def test_threads_keep_their_state_isolated(tmp_path: Path) -> None:
    graph = build_test_graph(tmp_path, executor=CompletingExecutor())
    first = graph.invoke(initial_state("one"), graph_config("thread-one"))
    second = graph.invoke(initial_state("two"), graph_config("thread-two"))

    assert first["thread_id"] == "thread-one"
    assert second["thread_id"] == "thread-two"
    assert graph.get_state(graph_config("thread-one")).values["goal"] == "goal-one"
    assert graph.get_state(graph_config("thread-two")).values["goal"] == "goal-two"


def build_test_graph(
    tmp_path: Path,
    *,
    executor,
    satisfied: bool = True,
    approval_service=None,
    checkpoint_path: Path | None = None,
):
    path = checkpoint_path or tmp_path / "checkpoints.db"
    saver = SqliteSaver(sqlite3.connect(path, check_same_thread=False))
    runtime = SimpleNamespace(
        planner=FakePlanner(),
        scheduler=FakeScheduler(),
        executor=executor,
        evaluator=FakeEvaluator(satisfied=satisfied),
        replanner=FakeReplanner(),
        active_tools=[],
        catalog=SimpleNamespace(),
    )
    return build_agent_graph(
        GraphDependencies(
            planning_runtime=runtime,
            approval_service=approval_service,
            approval_user_id="user-1",
            max_task_retries=1,
        ),
        saver,
    )


def initial_state(label: str) -> dict:
    state = PlanState(goal=f"goal-{label}", tasks=[])
    return {
        "run_id": f"run-{label}",
        "thread_id": f"thread-{label}",
        "user_id": "user-1",
        "goal": f"goal-{label}",
        "conversation_context": [],
        "memory_context": {"records": []},
        "knowledge_base": {"available": False, "documents": []},
        "plan_state": plan_state_to_dict(state),
        "next_task_id": None,
        "last_result": None,
        "final_answer": "",
        "status": "running",
        "error": None,
        "graph_retry_count": 0,
        "max_plan_revisions": 0,
        "approval": None,
        "started_at": "2026-08-20T00:00:00+00:00",
        "completed_at": None,
        "node_trace": [],
    }


def graph_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}


def node_names(result: dict) -> list[str]:
    return [event["node"] for event in result["node_trace"]]


class FakePlanner:
    def create_plan(self, *, goal: str, **_kwargs) -> PlanState:
        return PlanState(
            goal=goal,
            tasks=[
                PlanTask(
                    task_id="work",
                    description="Complete one deterministic task",
                    capability=TaskCapability.LLM,
                    output_key="result",
                    max_retries=1,
                )
            ],
            status=PlanStatus.VALIDATED,
        )


class FakeScheduler:
    def refresh(self, state: PlanState, _catalog) -> None:
        for task in state.tasks:
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.READY

    def ready_tasks(self, state: PlanState) -> list[PlanTask]:
        return [task for task in state.tasks if task.status == TaskStatus.READY]


class FakeEvaluator:
    def __init__(self, *, satisfied: bool) -> None:
        self.satisfied = satisfied

    def evaluate(self, _state: PlanState) -> GoalEvaluation:
        if self.satisfied:
            return GoalEvaluation(
                goal_satisfied=True,
                reason="All required outputs are present.",
                final_answer="Completed safely.",
            )
        return GoalEvaluation(
            goal_satisfied=False,
            reason="Permanent provider failure.",
            final_answer="The requested work could not complete.",
            replan_needed=False,
            missing=("result",),
        )


class FakeReplanner:
    def replan(self, *_args, **_kwargs):
        raise AssertionError("This test graph must not replan.")


class CompletingExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, task: PlanTask, state: PlanState) -> TaskResult:
        self.calls += 1
        task.attempts += 1
        task.status = TaskStatus.COMPLETED
        result = TaskResult(task.task_id, TaskStatus.COMPLETED, output="done")
        task.result = result
        state.outputs[task.output_key] = result.output
        return result


class RetryOnceExecutor(CompletingExecutor):
    def execute(self, task: PlanTask, state: PlanState) -> TaskResult:
        self.calls += 1
        task.attempts += 1
        if self.calls == 1:
            task.status = TaskStatus.FAILED
            result = TaskResult(
                task.task_id,
                TaskStatus.FAILED,
                error="temporary timeout",
                retryable=True,
            )
            task.result = result
            return result
        task.status = TaskStatus.COMPLETED
        result = TaskResult(task.task_id, TaskStatus.COMPLETED, output="recovered")
        task.result = result
        state.outputs[task.output_key] = result.output
        return result


class PermanentFailureExecutor(CompletingExecutor):
    def execute(self, task: PlanTask, _state: PlanState) -> TaskResult:
        self.calls += 1
        task.attempts += 1
        task.status = TaskStatus.FAILED
        result = TaskResult(
            task.task_id,
            TaskStatus.FAILED,
            error="authentication failed",
            retryable=False,
        )
        task.result = result
        return result


class FakeApprovalService:
    def __init__(self) -> None:
        self.request = SimpleNamespace(
            approval_id="approval-1",
            status=ApprovalStatus.PENDING,
            expires_at="2099-01-01T00:00:00+00:00",
        )
        self.proposal = SimpleNamespace(
            action_id="action-1",
            version=1,
            tool_name="email.send_mock",
            risk_level=SimpleNamespace(value="high"),
            preview={"title": "Mock email", "fields": {"To": "user@example.com"}},
        )

    def get_approval(self, _approval_id):
        return self.request

    def get_action(self, _action_id, _version):
        return self.proposal


class ApprovalExecutor(CompletingExecutor):
    def __init__(self, approval: FakeApprovalService) -> None:
        super().__init__()
        self.approval = approval

    def execute(self, task: PlanTask, state: PlanState) -> TaskResult:
        self.calls += 1
        if task.status == TaskStatus.READY:
            task.status = TaskStatus.WAITING_FOR_APPROVAL
            task.action_id = "action-1"
            task.action_version = 1
            task.approval_id = "approval-1"
            result = TaskResult(task.task_id, TaskStatus.WAITING_FOR_APPROVAL)
            task.result = result
            return result
        assert self.approval.request.status == ApprovalStatus.APPROVED
        task.status = TaskStatus.COMPLETED
        result = TaskResult(task.task_id, TaskStatus.COMPLETED, output="approved work")
        task.result = result
        state.outputs[task.output_key] = result.output
        return result
