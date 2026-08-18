import json

import pytest

from executor.executor import TaskExecutor
from executor.retry_policy import RetryPolicy
from executor.task_runner import TaskRunner
from planner.goal_evaluator import GoalEvaluator
from planner.models import (
    PlanState,
    PlanStatus,
    PlanTask,
    TaskCapability,
    TaskResult,
    TaskStatus,
)
from planner.plan_validator import (
    CapabilityCatalog,
    PlanValidationError,
    PlanValidator,
)
from planner.planner import Planner, PlannerError
from planner.planning_need import PlanningNeedDetector
from planner.replanner import Replanner
from planner.runtime import PlanningRuntime
from planner.scheduler import TaskScheduler
from rag.models import RetrievedChunk, RetrievalResult
from tools.base import ToolDefinition, ToolResult
from tools.factory import build_default_registry
from tools.manager import ToolManager
from tools.registry import ToolRegistry


def make_task(
    task_id: str,
    *,
    capability: TaskCapability = TaskCapability.LLM,
    dependencies: tuple[str, ...] = (),
    inputs: tuple[str, ...] = ("goal",),
    output_key: str | None = None,
    tool_name: str | None = None,
    tool_arguments: dict | None = None,
    query: str | None = None,
    max_retries: int = 0,
) -> PlanTask:
    return PlanTask(
        task_id=task_id,
        description=f"Complete {task_id}",
        capability=capability,
        dependencies=dependencies,
        inputs=inputs,
        output_key=output_key or f"{task_id}_output",
        tool_name=tool_name,
        tool_arguments=tool_arguments or {},
        query=query,
        max_retries=max_retries,
    )


def llm_catalog(**kwargs) -> CapabilityCatalog:
    return CapabilityCatalog(**kwargs)


def validator(max_tasks: int = 8, max_retries: int = 1) -> PlanValidator:
    return PlanValidator(max_tasks=max_tasks, max_task_retries=max_retries)


def empty_manager() -> ToolManager:
    return ToolManager(ToolRegistry(), set())


def test_simple_request_skips_planning_and_complex_goal_enters_planner() -> None:
    detector = PlanningNeedDetector()

    assert detector.detect("What is an HTTP header?").needs_planning is False
    decision = detector.detect(
        "Research PostgreSQL and MySQL, compare their tradeoffs, and recommend one."
    )
    assert decision.needs_planning is True
    assert decision.score >= 3


def test_valid_plan_exposes_dependency_only_after_parent_completes() -> None:
    research = make_task("research_db")
    compare = make_task(
        "compare_db",
        dependencies=("research_db",),
        inputs=("research_db_output",),
    )
    state = PlanState(goal="Compare databases", tasks=[research, compare])
    catalog = llm_catalog()
    validator().validate(state, catalog)

    scheduler = TaskScheduler()
    scheduler.refresh(state, catalog)
    assert [task.task_id for task in scheduler.ready_tasks(state)] == ["research_db"]
    assert compare.status == TaskStatus.PENDING

    research.status = TaskStatus.COMPLETED
    state.outputs[research.output_key] = "PostgreSQL notes"
    scheduler.refresh(state, catalog)
    assert compare.status == TaskStatus.READY


def test_cycle_is_rejected_before_execution() -> None:
    first = make_task("first_task", dependencies=("second_task",))
    second = make_task("second_task", dependencies=("first_task",))
    state = PlanState(goal="Invalid cycle", tasks=[first, second])

    with pytest.raises(PlanValidationError, match="cycle"):
        validator().validate(state, llm_catalog())


def test_missing_capability_and_oversized_plan_are_rejected() -> None:
    state = PlanState(
        goal="Unavailable work",
        tasks=[
            make_task("private_docs", capability=TaskCapability.RAG),
            make_task("extra_task"),
        ],
    )

    with pytest.raises(PlanValidationError) as exc_info:
        validator(max_tasks=1).validate(state, llm_catalog(rag_available=False))

    assert "maximum is 1" in str(exc_info.value)
    assert "unavailable capability rag" in str(exc_info.value)


def test_independent_tasks_are_ready_together_but_v1_can_select_one() -> None:
    state = PlanState(
        goal="Independent research",
        tasks=[make_task("research_one"), make_task("research_two")],
    )
    scheduler = TaskScheduler()
    scheduler.refresh(state, llm_catalog())

    assert [task.status for task in state.tasks] == [TaskStatus.READY, TaskStatus.READY]
    assert scheduler.ready_tasks(state)[0].task_id == "research_one"


def test_planner_repairs_invalid_capability_once() -> None:
    responses = iter(
        [
            json.dumps(
                {
                    "assumptions": [],
                    "tasks": [
                        {
                            "id": "bad_tool",
                            "description": "Use an unavailable tool",
                            "capability": "tool",
                            "dependencies": [],
                            "inputs": ["goal"],
                            "output_key": "tool_output",
                            "tool_name": "missing.tool",
                            "tool_arguments": {},
                            "priority": 0,
                            "required": True,
                            "max_retries": 0,
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "assumptions": [],
                    "tasks": [
                        {
                            "id": "answer_goal",
                            "description": "Answer using the LLM",
                            "capability": "llm",
                            "dependencies": [],
                            "inputs": ["goal"],
                            "output_key": "answer",
                            "tool_name": None,
                            "tool_arguments": {},
                            "query": None,
                            "priority": 0,
                            "required": True,
                            "max_retries": 0,
                        }
                    ],
                }
            ),
        ]
    )
    planner = Planner(
        llm_fn=lambda _messages: next(responses),
        validator=validator(),
        catalog=llm_catalog(),
        max_repair_attempts=1,
        default_task_retries=0,
    )

    state = planner.create_plan(
        goal="Answer safely",
        conversation_context=[],
        memory_context=None,
        knowledge_base={"available": False, "documents": []},
        active_tools=[],
    )

    assert state.status == PlanStatus.VALIDATED
    assert state.metrics.planner_calls == 2
    assert state.tasks[0].task_id == "answer_goal"


def test_planner_receives_selected_memory_not_the_memory_database() -> None:
    captured = {}

    def fake_llm(messages):
        captured["prompt"] = messages[-1]["content"]
        return json.dumps(
            {
                "assumptions": [],
                "tasks": [
                    {
                        "id": "use_preference",
                        "description": "Use the selected language preference",
                        "capability": "llm",
                        "dependencies": [],
                        "inputs": ["memory_context"],
                        "output_key": "recommendation",
                        "tool_name": None,
                        "tool_arguments": {},
                        "query": None,
                        "priority": 0,
                        "required": True,
                        "max_retries": 0,
                    }
                ],
            }
        )

    planner = Planner(
        llm_fn=fake_llm,
        validator=validator(),
        catalog=llm_catalog(memory_available=True),
        max_repair_attempts=0,
        default_task_retries=0,
    )
    planner.create_plan(
        goal="Recommend a backend language",
        conversation_context=[],
        memory_context={"records": [{"content": "User prefers Python"}]},
        knowledge_base={"available": False, "documents": []},
        active_tools=[],
    )

    assert "User prefers Python" in captured["prompt"]


def test_transient_tool_failure_retries_once_then_completes() -> None:
    calls = {"count": 0}

    def flaky_tool(_arguments):
        calls["count"] += 1
        if calls["count"] == 1:
            return ToolResult(
                success=False,
                error="temporarily unavailable",
                metadata={"retryable": True},
            )
        return ToolResult(success=True, result={"value": 42})

    registry = ToolRegistry()
    registry.register_tool(
        ToolDefinition(
            name="demo.flaky",
            description="Transient test tool",
            input_schema={"type": "object", "properties": {}, "required": []},
            output_schema={"type": "object"},
            permission="safe",
            timeout_seconds=1,
            version="test",
            execute=flaky_tool,
        )
    )
    manager = ToolManager(registry, {"demo.flaky"})
    task = make_task(
        "call_flaky",
        capability=TaskCapability.TOOL,
        tool_name="demo.flaky",
        max_retries=1,
    )
    task.status = TaskStatus.READY
    state = PlanState(goal="Use flaky tool", tasks=[task])
    runner = TaskRunner(
        llm_fn=lambda _messages: "unused",
        tool_manager=manager,
        rag_pipeline=None,
        memory_search_fn=None,
        conversation_context=[],
        memory_context=None,
        rag_top_k=4,
        rag_min_score=0.25,
    )
    executor = TaskExecutor(
        runner=runner,
        retry_policy=RetryPolicy(1),
        max_execution_steps=3,
    )

    executor.execute(task, state)

    assert task.status == TaskStatus.COMPLETED
    assert task.attempts == 2
    assert state.outputs[task.output_key] == {"value": 42}


def test_side_effecting_tool_stays_blocked_without_confirmation() -> None:
    registry = ToolRegistry()
    registry.register_tool(
        ToolDefinition(
            name="demo.write",
            description="Side effect test tool",
            input_schema={"type": "object", "properties": {}, "required": []},
            output_schema={"type": "object"},
            permission="side_effecting",
            timeout_seconds=1,
            version="test",
            execute=lambda _arguments: ToolResult(success=True, result="should not run"),
        )
    )
    task = make_task(
        "write_data",
        capability=TaskCapability.TOOL,
        tool_name="demo.write",
    )
    task.status = TaskStatus.READY
    state = PlanState(goal="Write data", tasks=[task])
    executor = TaskExecutor(
        runner=TaskRunner(
            llm_fn=lambda _messages: "unused",
            tool_manager=ToolManager(registry, {"demo.write"}),
            rag_pipeline=None,
            memory_search_fn=None,
            conversation_context=[],
            memory_context=None,
            rag_top_k=4,
            rag_min_score=0.25,
        ),
        retry_policy=RetryPolicy(1),
        max_execution_steps=2,
    )

    executor.execute(task, state)

    assert task.status == TaskStatus.FAILED
    assert task.attempts == 0
    assert "require user confirmation" in task.error


def test_failed_dependency_blocks_downstream_without_running_it() -> None:
    failed = make_task("failed_work")
    failed.status = TaskStatus.FAILED
    downstream = make_task(
        "dependent_work",
        dependencies=("failed_work",),
        inputs=("failed_work_output",),
    )
    state = PlanState(goal="Handle failure", tasks=[failed, downstream])

    TaskScheduler().refresh(state, llm_catalog())

    assert downstream.status == TaskStatus.BLOCKED
    assert downstream.attempts == 0


def test_goal_evaluator_cannot_complete_with_required_pending_task() -> None:
    state = PlanState(goal="Do required work", tasks=[make_task("required_work")])
    evaluator = GoalEvaluator(
        lambda _messages: json.dumps(
            {
                "goal_satisfied": True,
                "reason": "Looks done",
                "final_answer": "Done",
                "replan_needed": False,
                "missing": [],
            }
        )
    )

    result = evaluator.evaluate(state)

    assert result.goal_satisfied is False
    assert result.replan_needed is True
    assert "required_work" in result.missing


def test_goal_evaluator_cannot_accept_empty_required_rag_evidence() -> None:
    task = make_task("retrieve_policy", capability=TaskCapability.RAG)
    task.status = TaskStatus.COMPLETED
    task.result = TaskResult(
        task_id=task.task_id,
        status=TaskStatus.COMPLETED,
        output={"evidence_found": False, "chunks": []},
    )
    state = PlanState(goal="Answer from policy", tasks=[task])
    evaluator = GoalEvaluator(
        lambda _messages: json.dumps(
            {
                "goal_satisfied": True,
                "reason": "Done",
                "final_answer": "Unsupported answer",
                "replan_needed": False,
                "missing": [],
            }
        )
    )

    result = evaluator.evaluate(state)

    assert result.goal_satisfied is False
    assert result.replan_needed is True
    assert task.task_id in result.missing


def test_replanner_preserves_completed_work_and_retires_failed_work() -> None:
    completed = make_task("completed_work")
    completed.status = TaskStatus.COMPLETED
    failed = make_task("failed_work")
    failed.status = TaskStatus.FAILED
    state = PlanState(
        goal="Recover plan",
        tasks=[completed, failed],
        outputs={completed.output_key: "useful result"},
    )
    response = json.dumps(
        {
            "assumptions": ["Use completed evidence"],
            "tasks": [
                {
                    "id": "replacement_work",
                    "description": "Replace the failed work",
                    "capability": "llm",
                    "dependencies": ["completed_work"],
                    "inputs": ["completed_work_output"],
                    "output_key": "replacement_output",
                    "tool_name": None,
                    "tool_arguments": {},
                    "query": None,
                    "priority": 0,
                    "required": True,
                    "max_retries": 0,
                }
            ],
        }
    )
    replanner = Replanner(
        llm_fn=lambda _messages: response,
        validator=validator(),
        catalog=llm_catalog(),
        max_repair_attempts=0,
        default_task_retries=0,
    )

    revised = replanner.replan(state, reason="failed task", active_tools=[])

    assert revised.revision == 1
    assert revised.get_task("completed_work").status == TaskStatus.COMPLETED
    assert revised.get_task("failed_work").required is False
    assert revised.get_task("replacement_work").status == TaskStatus.PENDING


def test_runtime_cancellation_stops_before_starting_new_work() -> None:
    state = PlanState(goal="Cancel me", tasks=[make_task("waiting_work")])

    class StubPlanner:
        def create_plan(self, **_kwargs):
            state.status = PlanStatus.VALIDATED
            return state

    class MustNotExecute:
        def execute(self, _task, _state):
            raise AssertionError("Executor must not run after cancellation")

    runtime = PlanningRuntime(
        planner=StubPlanner(),
        scheduler=TaskScheduler(),
        executor=MustNotExecute(),
        evaluator=None,
        replanner=None,
        catalog=llm_catalog(),
        max_plan_revisions=0,
        active_tools=[],
    )

    result = runtime.run(
        goal=state.goal,
        conversation_context=[],
        memory_context=None,
        knowledge_base={"available": False},
        cancellation_check=lambda: True,
    )

    assert result.status == PlanStatus.CANCELLED
    assert result.tasks[0].status == TaskStatus.CANCELLED
    assert result.tasks[0].attempts == 0


def test_replanner_cannot_repeat_completed_approval_bound_action() -> None:
    completed = make_task(
        "send_update",
        capability=TaskCapability.TOOL,
        output_key="send_receipt",
        tool_name="email.send_mock",
        tool_arguments={"to": "jane@example.com"},
    )
    completed.status = TaskStatus.COMPLETED
    completed.action_id = "act_completed"
    completed.action_version = 2
    state = PlanState(goal="Send the edited update", tasks=[completed])
    response = json.dumps(
        {
            "assumptions": [],
            "tasks": [
                {
                    "id": "send_update_again",
                    "description": "Repeat the completed send",
                    "capability": "tool",
                    "dependencies": [],
                    "inputs": ["goal"],
                    "output_key": "second_receipt",
                    "tool_name": "email.send_mock",
                    "tool_arguments": {"to": "jane@example.com"},
                    "query": None,
                    "priority": 0,
                    "required": True,
                    "max_retries": 0,
                }
            ],
        }
    )
    replanner = Replanner(
        llm_fn=lambda _messages: response,
        validator=PlanValidator(max_tasks=8, max_task_retries=1),
        catalog=CapabilityCatalog(tools=frozenset({"email.send_mock"})),
        max_repair_attempts=0,
        default_task_retries=0,
    )

    with pytest.raises(PlannerError, match="completed approval-bound side effect"):
        replanner.replan(
            state,
            reason="evaluator requested duplicate work",
            active_tools=[],
        )


class FakeRagPipeline:
    max_context_chars = 4000

    def retrieve(self, query, top_k, min_score):
        return RetrievalResult(
            query=query,
            chunks=[
                RetrievedChunk(
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    text="The handbook allows 10 carry-over leave days.",
                    score=0.9,
                    metadata={"filename": "handbook.pdf", "page_number": 1},
                )
            ],
            query_embedding_seconds=0.01,
            retrieval_seconds=0.01,
            top_k=top_k,
            min_score=min_score,
        )


def test_combined_memory_rag_tool_and_llm_plan_execution() -> None:
    memory_task = make_task(
        "read_memory",
        capability=TaskCapability.MEMORY,
        inputs=("goal",),
        output_key="preference",
        query="preferred leave month",
    )
    rag_task = make_task(
        "read_policy",
        capability=TaskCapability.RAG,
        inputs=("goal",),
        output_key="policy",
        query="carry-over leave allowance",
    )
    tool_task = make_task(
        "calculate_total",
        capability=TaskCapability.TOOL,
        inputs=("goal",),
        output_key="calculation",
        tool_name="calculator.evaluate",
        tool_arguments={"expression": "10 + 5"},
    )
    synthesis = make_task(
        "synthesize_answer",
        dependencies=("read_memory", "read_policy", "calculate_total"),
        inputs=("preference", "policy", "calculation"),
        output_key="final_draft",
    )
    state = PlanState(
        goal="Use memory, policy, and exact math",
        tasks=[memory_task, rag_task, tool_task, synthesis],
    )
    manager = ToolManager(build_default_registry(), {"calculator.evaluate"})
    runner = TaskRunner(
        llm_fn=lambda messages: (
            "Grounded synthesis"
            if all(key in messages[-1]["content"] for key in ("preference", "policy", "calculation"))
            else "missing inputs"
        ),
        tool_manager=manager,
        rag_pipeline=FakeRagPipeline(),
        memory_search_fn=lambda _query: {"records": ["User prefers December leave"]},
        conversation_context=[],
        memory_context=None,
        rag_top_k=4,
        rag_min_score=0.25,
    )
    executor = TaskExecutor(
        runner=runner,
        retry_policy=RetryPolicy(1),
        max_execution_steps=8,
    )
    catalog = llm_catalog(
        tools=frozenset({"calculator.evaluate"}),
        rag_available=True,
        memory_available=True,
    )
    validator().validate(state, catalog)
    scheduler = TaskScheduler()

    while any(task.status != TaskStatus.COMPLETED for task in state.tasks):
        scheduler.refresh(state, catalog)
        executor.execute(scheduler.ready_tasks(state)[0], state)

    assert state.outputs["calculation"]["result"] == 15
    assert state.outputs["final_draft"] == "Grounded synthesis"
    assert state.metrics.memory_retrievals == 1
    assert state.metrics.rag_retrievals == 1
    assert state.metrics.tool_calls == 1
    assert state.metrics.executor_llm_calls == 1
