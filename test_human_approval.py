import json
from dataclasses import replace

import pytest

from approval.models import (
    ActionStatus,
    ApprovalStatus,
    RiskLevel,
    SideEffectType,
)
from approval.policy import ApprovalPolicy
from approval.repository import SQLiteApprovalRepository
from approval.risk_engine import RiskEngine
from approval.service import ApprovalService, ApprovalServiceError
from executor.executor import TaskExecutor
from executor.retry_policy import RetryPolicy
from executor.task_runner import TaskRunner
from planner.goal_evaluator import GoalEvaluator
from planner.models import (
    PlanState,
    PlanStatus,
    PlanTask,
    TaskCapability,
    TaskStatus,
)
from planner.plan_validator import CapabilityCatalog
from planner.planning_need import PlanningNeedDetector
from planner.runtime import PlanningRuntime
from planner.scheduler import TaskScheduler
from tools.base import ToolDefinition, ToolResult
from tools.factory import build_default_registry
from tools.manager import ToolManager
from tools.registry import ToolRegistry


def build_service(tmp_path, registry, timeout_seconds=300):
    return ApprovalService(
        repository=SQLiteApprovalRepository(tmp_path / "approvals.db"),
        tool_lookup=registry.get_tool,
        risk_engine=RiskEngine(),
        policy=ApprovalPolicy(confirmation_timeout_seconds=timeout_seconds),
    )


def build_manager(registry, *, side_effect_permission=True):
    permissions = {"safe", "read_only_external"}
    if side_effect_permission:
        permissions.add("side_effecting")
    return ToolManager(
        registry,
        {tool.name for tool in registry.list_tools()},
        authorized_permissions=permissions,
    )


def build_runner(registry, service, *, side_effect_permission=True, llm_fn=None):
    return TaskRunner(
        llm_fn=llm_fn or (lambda _messages: "Draft project update"),
        tool_manager=build_manager(
            registry, side_effect_permission=side_effect_permission
        ),
        rag_pipeline=None,
        memory_search_fn=None,
        conversation_context=[],
        memory_context=None,
        rag_top_k=4,
        rag_min_score=0.25,
        approval_service=service,
        approval_user_id="user-1",
    )


def build_executor(runner):
    return TaskExecutor(
        runner=runner,
        retry_policy=RetryPolicy(1),
        max_execution_steps=12,
    )


def email_task(task_id="send_email", dependencies=(), body="Project update"):
    return PlanTask(
        task_id=task_id,
        description="Send the reviewed project update",
        capability=TaskCapability.TOOL,
        dependencies=dependencies,
        inputs=(),
        output_key=f"{task_id}_receipt",
        tool_name="email.send_mock",
        tool_arguments={
            "to": "john@example.com",
            "subject": "Project Update",
            "body": body,
        },
        max_retries=1,
    )


def pause_task(executor, task, state):
    task.status = TaskStatus.READY
    result = executor.execute(task, state)
    assert result.status == TaskStatus.WAITING_FOR_APPROVAL
    assert task.status == TaskStatus.WAITING_FOR_APPROVAL
    assert task.attempts == 0
    return result


def approve_task(service, task):
    return service.approve(
        task.approval_id,
        user_id="user-1",
        expected_version=task.action_version,
    )


def test_low_risk_calculator_executes_without_approval(tmp_path) -> None:
    registry = build_default_registry()
    service = build_service(tmp_path, registry)
    task = PlanTask(
        task_id="calculate_total",
        description="Calculate 25 times 17",
        capability=TaskCapability.TOOL,
        output_key="calculation",
        tool_name="calculator.evaluate",
        tool_arguments={"expression": "25 * 17"},
    )
    task.status = TaskStatus.READY
    state = PlanState(goal="Calculate 25 * 17", tasks=[task])

    build_executor(build_runner(registry, service)).execute(task, state)

    assert task.status == TaskStatus.COMPLETED
    assert state.outputs["calculation"]["result"] == 425
    assert task.action_id is None
    assert state.metrics.approval_requests == 0


def test_consequential_send_language_routes_to_approval_aware_planner() -> None:
    decision = PlanningNeedDetector().detect(
        "Send a simulated project update to john@example.com with subject Weekly Update."
    )

    assert decision.needs_planning is True
    assert "consequential action" in decision.reasons[0]


def test_draft_email_is_content_generation_not_a_side_effect(tmp_path) -> None:
    registry = build_default_registry()
    service = build_service(tmp_path, registry)
    task = PlanTask(
        task_id="draft_email",
        description="Draft an email to John about the project update",
        capability=TaskCapability.LLM,
        output_key="draft",
    )
    task.status = TaskStatus.READY
    state = PlanState(goal="Draft an email", tasks=[task])

    build_executor(build_runner(registry, service)).execute(task, state)

    assert task.status == TaskStatus.COMPLETED
    assert state.outputs["draft"] == "Draft project update"
    assert task.approval_id is None


def test_structured_llm_output_resolves_dotted_approval_preview(tmp_path) -> None:
    registry = build_default_registry()
    service = build_service(tmp_path, registry)
    runner = build_runner(
        registry,
        service,
        llm_fn=lambda _messages: json.dumps(
            {
                "email_content": {
                    "to": "john@example.com",
                    "subject": "Weekly Update",
                    "body": "The build is ready.",
                }
            }
        ),
    )
    executor = build_executor(runner)
    draft = PlanTask(
        task_id="prepare_email",
        description="Prepare structured email content",
        capability=TaskCapability.LLM,
        output_key="email_content",
        status=TaskStatus.READY,
    )
    send = email_task(dependencies=("prepare_email",))
    send.tool_arguments = {
        "to": "{{email_content.to}}",
        "subject": "{{email_content.subject}}",
        "body": "{{email_content.body}}",
    }
    state = PlanState(goal="Prepare and send update", tasks=[draft, send])

    executor.execute(draft, state)
    pause_task(executor, send, state)

    assert state.outputs["email_content"]["to"] == "john@example.com"
    proposal = service.get_action(send.action_id, send.action_version)
    assert proposal.preview["fields"] == {
        "To": "john@example.com",
        "Subject": "Weekly Update",
        "Body": "The build is ready.",
    }


def test_send_email_pauses_before_tool_execution(tmp_path) -> None:
    registry = build_default_registry()
    service = build_service(tmp_path, registry)
    task = email_task()
    state = PlanState(goal="Send project update", tasks=[task])

    pause_task(build_executor(build_runner(registry, service)), task, state)

    proposal = service.get_action(task.action_id, task.action_version)
    request = service.get_approval(task.approval_id)
    assert proposal.status == ActionStatus.PENDING_APPROVAL
    assert proposal.preview["fields"]["To"] == "john@example.com"
    assert request.status == ApprovalStatus.PENDING
    assert service.repository.get_receipt(proposal.idempotency_key) is None


def test_approval_executes_exact_action_and_creates_receipt(tmp_path) -> None:
    registry = build_default_registry()
    service = build_service(tmp_path, registry)
    executor = build_executor(build_runner(registry, service))
    task = email_task()
    state = PlanState(goal="Send project update", tasks=[task])
    pause_task(executor, task, state)

    approve_task(service, task)
    executor.execute(task, state)

    assert task.status == TaskStatus.COMPLETED
    assert task.attempts == 1
    assert task.execution_receipt_id
    receipt = service.repository.get_receipt(
        service.get_action(task.action_id, task.action_version).idempotency_key
    )
    assert receipt.status == "completed"
    assert receipt.external_id.startswith("mock_msg_")


def test_denial_marks_task_denied_and_never_executes(tmp_path) -> None:
    registry = build_default_registry()
    service = build_service(tmp_path, registry)
    executor = build_executor(build_runner(registry, service))
    task = email_task()
    state = PlanState(goal="Send project update", tasks=[task])
    pause_task(executor, task, state)

    service.deny(
        task.approval_id,
        user_id="user-1",
        expected_version=task.action_version,
    )
    executor.execute(task, state)

    assert task.status == TaskStatus.DENIED
    assert task.attempts == 0
    assert task.execution_receipt_id is None


def test_cancellation_marks_task_cancelled_and_never_executes(tmp_path) -> None:
    registry = build_default_registry()
    service = build_service(tmp_path, registry)
    executor = build_executor(build_runner(registry, service))
    task = email_task()
    state = PlanState(goal="Send project update", tasks=[task])
    pause_task(executor, task, state)

    service.cancel(
        task.approval_id,
        user_id="user-1",
        expected_version=task.action_version,
    )
    executor.execute(task, state)

    assert task.status == TaskStatus.CANCELLED
    assert task.attempts == 0
    assert task.execution_receipt_id is None


def test_approval_is_bound_to_the_original_user_session(tmp_path) -> None:
    registry = build_default_registry()
    service = build_service(tmp_path, registry)
    task = email_task()
    state = PlanState(goal="Send project update", tasks=[task])
    pause_task(build_executor(build_runner(registry, service)), task, state)

    with pytest.raises(ApprovalServiceError, match="different user session"):
        service.approve(
            task.approval_id,
            user_id="user-2",
            expected_version=task.action_version,
        )

    assert service.get_approval(task.approval_id).status == ApprovalStatus.PENDING


def test_edit_increments_version_and_invalidates_old_approval(tmp_path) -> None:
    registry = build_default_registry()
    service = build_service(tmp_path, registry)
    executor = build_executor(build_runner(registry, service))
    task = email_task()
    state = PlanState(goal="Send project update", tasks=[task])
    pause_task(executor, task, state)
    old_approval_id = task.approval_id

    revised, request = service.edit(
        old_approval_id,
        user_id="user-1",
        expected_version=1,
        arguments={
            "to": "jane@example.com",
            "subject": "Revised Update",
            "body": "Edited body",
        },
    )
    task.action_version = revised.version
    task.approval_id = request.approval_id

    assert revised.version == 2
    assert revised.argument_digest != service.get_action(task.action_id, 1).argument_digest
    assert service.get_approval(old_approval_id).status == ApprovalStatus.EDITED
    with pytest.raises(ApprovalServiceError):
        service.approve(old_approval_id, user_id="user-1", expected_version=1)

    approve_task(service, task)
    executor.execute(task, state)
    assert state.outputs[task.output_key]["to"] == "jane@example.com"


def test_approval_expiry_fails_closed(tmp_path) -> None:
    registry = build_default_registry()
    service = build_service(tmp_path, registry, timeout_seconds=0)
    executor = build_executor(build_runner(registry, service))
    task = email_task()
    state = PlanState(goal="Send project update", tasks=[task])
    pause_task(executor, task, state)

    request = service.get_approval(task.approval_id)
    executor.execute(task, state)

    assert request.status == ApprovalStatus.EXPIRED
    assert task.status == TaskStatus.EXPIRED
    assert task.attempts == 0


def test_approval_does_not_override_revoked_permission(tmp_path) -> None:
    registry = build_default_registry()
    service = build_service(tmp_path, registry)
    task = email_task()
    state = PlanState(goal="Send project update", tasks=[task])
    pause_task(build_executor(build_runner(registry, service)), task, state)
    approve_task(service, task)

    revoked_executor = build_executor(
        build_runner(registry, service, side_effect_permission=False)
    )
    revoked_executor.execute(task, state)

    assert task.status == TaskStatus.FAILED
    assert task.attempts == 0
    assert "not authorized" in task.error


def test_tool_contract_change_invalidates_approval(tmp_path) -> None:
    registry = build_default_registry()
    service = build_service(tmp_path, registry)
    executor = build_executor(build_runner(registry, service))
    task = email_task()
    state = PlanState(goal="Send project update", tasks=[task])
    pause_task(executor, task, state)
    approve_task(service, task)

    email_tool = registry.get_tool("email.send_mock")
    registry.remove_tool("email.send_mock")
    registry.register_tool(replace(email_tool, version="2.0"))
    executor.execute(task, state)

    assert task.status == TaskStatus.FAILED
    assert task.attempts == 0
    assert "Tool contract changed" in task.error


def test_duplicate_execution_uses_existing_receipt(tmp_path) -> None:
    calls = {"count": 0}
    registry = build_default_registry()
    email_tool = registry.get_tool("email.send_mock")

    def counted_execute(arguments):
        calls["count"] += 1
        return ToolResult(
            success=True,
            result={"external_id": "mock_once", "to": arguments["to"]},
            metadata={"external_id": "mock_once"},
        )

    registry.remove_tool("email.send_mock")
    registry.register_tool(replace(email_tool, execute=counted_execute))
    service = build_service(tmp_path, registry)
    runner = build_runner(registry, service)
    executor = build_executor(runner)
    task = email_task()
    state = PlanState(goal="Send project update", tasks=[task])
    pause_task(executor, task, state)
    proposal = approve_task(service, task)

    first = runner.run(task, state, approved_action=proposal)
    second = runner.run(task, state, approved_action=proposal)

    assert first.status == TaskStatus.COMPLETED
    assert second.status == TaskStatus.COMPLETED
    assert second.output["idempotent_replay"] is True
    assert calls["count"] == 1


def test_tool_failure_after_approval_records_one_failed_receipt(tmp_path) -> None:
    calls = {"count": 0}
    registry = ToolRegistry()

    def fail_tool(_arguments):
        calls["count"] += 1
        return ToolResult(success=False, error="simulated provider failure")

    registry.register_tool(
        ToolDefinition(
            name="demo.side_effect",
            description="Failing side-effect test",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
            output_schema={"type": "object"},
            permission="side_effecting",
            timeout_seconds=1,
            version="1.0",
            execute=fail_tool,
            risk_level=RiskLevel.HIGH,
            side_effect=SideEffectType.REVERSIBLE_WRITE,
            requires_confirmation=True,
        )
    )
    service = build_service(tmp_path, registry)
    executor = build_executor(build_runner(registry, service))
    task = PlanTask(
        task_id="failing_action",
        description="Run failing side effect",
        capability=TaskCapability.TOOL,
        output_key="result",
        tool_name="demo.side_effect",
        tool_arguments={"value": "x"},
        max_retries=1,
    )
    state = PlanState(goal="Failure test", tasks=[task])
    pause_task(executor, task, state)
    approve_task(service, task)

    executor.execute(task, state)

    proposal = service.get_action(task.action_id, task.action_version)
    receipt = service.repository.get_receipt(proposal.idempotency_key)
    assert task.status == TaskStatus.FAILED
    assert task.attempts == 1
    assert calls["count"] == 1
    assert receipt.status == "failed"
    assert len(
        [event for event in service.list_audit(task.action_id) if event.event_type == "approval_requested"]
    ) == 1


def test_destructive_mock_has_high_risk_preview_and_waits(tmp_path) -> None:
    registry = build_default_registry()
    service = build_service(tmp_path, registry)
    executor = build_executor(build_runner(registry, service))
    task = PlanTask(
        task_id="delete_files",
        description="Delete two obsolete reports",
        capability=TaskCapability.TOOL,
        output_key="deletion",
        tool_name="files.delete_mock",
        tool_arguments={"paths": ["report1.pdf", "report2.pdf"]},
    )
    state = PlanState(goal="Delete obsolete reports", tasks=[task])

    pause_task(executor, task, state)

    proposal = service.get_action(task.action_id, task.action_version)
    assert proposal.risk_level == RiskLevel.HIGH
    assert proposal.preview["fields"]["Files"] == ["report1.pdf", "report2.pdf"]
    assert proposal.side_effect == SideEffectType.DESTRUCTIVE


def make_runtime(state, registry, service):
    class StubPlanner:
        def create_plan(self, **_kwargs):
            state.status = PlanStatus.VALIDATED
            return state

    runner = build_runner(registry, service)
    return PlanningRuntime(
        planner=StubPlanner(),
        scheduler=TaskScheduler(),
        executor=build_executor(runner),
        evaluator=GoalEvaluator(
            lambda _messages: json.dumps(
                {
                    "goal_satisfied": True,
                    "reason": "All required outputs are present",
                    "final_answer": "Workflow completed safely.",
                    "replan_needed": False,
                    "missing": [],
                }
            )
        ),
        replanner=None,
        catalog=CapabilityCatalog(
            tools=frozenset(tool.name for tool in registry.list_tools())
        ),
        max_plan_revisions=0,
        active_tools=[tool.to_model_description() for tool in registry.list_tools()],
        workflow_persist=lambda current: service.save_workflow(
            current, user_id="user-1"
        ),
    )


def test_plan_persists_pause_and_resumes_after_approval(tmp_path) -> None:
    registry = build_default_registry()
    service = build_service(tmp_path, registry)
    draft = PlanTask(
        task_id="draft_update",
        description="Draft project update",
        capability=TaskCapability.LLM,
        output_key="draft",
    )
    send = email_task(
        dependencies=("draft_update",),
        body="{{draft}}",
    )
    send.inputs = ("draft",)
    state = PlanState(goal="Draft and send project update", tasks=[draft, send])
    runtime = make_runtime(state, registry, service)

    paused = runtime.run(
        goal=state.goal,
        conversation_context=[],
        memory_context=None,
        knowledge_base={"available": False},
    )

    assert paused.status == PlanStatus.WAITING_FOR_APPROVAL
    assert draft.status == TaskStatus.COMPLETED
    assert send.status == TaskStatus.WAITING_FOR_APPROVAL
    loaded = service.find_waiting_workflow(user_id="user-1")
    assert loaded.plan_id == paused.plan_id
    loaded_send = loaded.get_task("send_email")
    approve_task(service, loaded_send)

    completed = runtime.resume(loaded)

    assert completed.status == PlanStatus.COMPLETED
    assert completed.get_task("send_email").status == TaskStatus.COMPLETED
    assert completed.final_answer == "Workflow completed safely."


def test_resume_failure_does_not_leave_workflow_running(tmp_path) -> None:
    registry = build_default_registry()
    service = build_service(tmp_path, registry)
    task = email_task()
    state = PlanState(goal="Send project update", tasks=[task])
    runtime = make_runtime(state, registry, service)
    paused = runtime.run(
        goal=state.goal,
        conversation_context=[],
        memory_context=None,
        knowledge_base={"available": False},
    )
    service.cancel(
        task.approval_id,
        user_id="user-1",
        expected_version=task.action_version,
    )

    class FailingEvaluator:
        def evaluate(self, _state):
            raise RuntimeError("simulated evaluator outage")

    runtime.evaluator = FailingEvaluator()
    with pytest.raises(RuntimeError, match="simulated evaluator outage"):
        runtime.resume(paused)

    persisted = service.load_workflow(paused.plan_id)
    assert paused.status == PlanStatus.FAILED
    assert persisted.status == PlanStatus.FAILED
    assert persisted.events[-1].event_type == "PLAN RESUME FAILED"


def test_two_side_effects_require_two_distinct_approvals(tmp_path) -> None:
    registry = build_default_registry()
    service = build_service(tmp_path, registry)
    first = email_task("send_first")
    second = email_task("send_second", dependencies=("send_first",))
    state = PlanState(goal="Send two separate updates", tasks=[first, second])
    runtime = make_runtime(state, registry, service)

    first_pause = runtime.run(
        goal=state.goal,
        conversation_context=[],
        memory_context=None,
        knowledge_base={"available": False},
    )
    approve_task(service, first_pause.get_task("send_first"))
    second_pause = runtime.resume(first_pause)

    assert second_pause.status == PlanStatus.WAITING_FOR_APPROVAL
    first_task = second_pause.get_task("send_first")
    second_task = second_pause.get_task("send_second")
    assert first_task.status == TaskStatus.COMPLETED
    assert second_task.status == TaskStatus.WAITING_FOR_APPROVAL
    assert first_task.action_id != second_task.action_id
    assert first_task.approval_id != second_task.approval_id

    approve_task(service, second_task)
    completed = runtime.resume(second_pause)
    assert completed.status == PlanStatus.COMPLETED


def test_risk_engine_escalates_large_recipient_scope() -> None:
    tool = ToolDefinition(
        name="email.bulk_mock",
        description="Bulk email test",
        input_schema={
            "type": "object",
            "properties": {
                "recipients": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                }
            },
            "required": ["recipients"],
        },
        output_schema={"type": "object"},
        permission="side_effecting",
        timeout_seconds=1,
        version="1.0",
        execute=lambda _arguments: ToolResult(success=True),
        risk_level=RiskLevel.MEDIUM,
        side_effect=SideEffectType.EXTERNAL_COMMUNICATION,
    )

    assessment = RiskEngine().evaluate(
        tool, {"recipients": [f"user{i}@example.com" for i in range(11)]}
    )

    assert assessment.risk_level == RiskLevel.CRITICAL


def test_fail_closed_when_approval_service_is_unavailable(tmp_path) -> None:
    registry = build_default_registry()
    runner = build_runner(registry, None, side_effect_permission=True)
    executor = build_executor(runner)
    task = email_task()
    state = PlanState(goal="Send without approval service", tasks=[task])
    task.status = TaskStatus.READY

    executor.execute(task, state)

    assert task.status == TaskStatus.FAILED
    assert task.attempts == 0
    assert "unavailable" in task.error
