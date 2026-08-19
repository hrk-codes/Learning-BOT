from __future__ import annotations

from collections.abc import Callable
from typing import Any

from approval.service import ApprovalService
from config import AppConfig
from executor.executor import TaskExecutor
from executor.retry_policy import RetryPolicy
from executor.task_runner import MemorySearchFn, TaskRunner
from llm.groq_client import complete_chat_completion
from planner.goal_evaluator import GoalEvaluator
from planner.models import PlanState
from planner.plan_validator import CapabilityCatalog, PlanValidator
from planner.planner import Planner
from planner.replanner import Replanner
from planner.runtime import CancellationCheck, PlanningRuntime, StatusCallback
from planner.scheduler import TaskScheduler
from rag.pipeline import RagPipeline
from tools.manager import ToolManager


def build_planning_runtime(
    *,
    config: AppConfig,
    conversation_context: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    tool_manager: ToolManager,
    rag_pipeline: RagPipeline | None,
    rag_top_k: int,
    rag_min_score: float,
    long_term_memory_context: dict[str, Any] | None,
    memory_search_fn: MemorySearchFn | None,
    approval_service: ApprovalService | None = None,
    approval_user_id: str = "local-user",
    graph_managed_task_retries: bool = False,
) -> PlanningRuntime:
    """Build the Stage 7/8 services without deciding how they are orchestrated.

    The custom runtime still owns its original procedural loop. Stage 9 asks LangGraph
    to own the transitions instead, so it uses this same factory with task retries
    surfaced to graph routing rather than hidden inside ``TaskExecutor``.
    """

    def planning_llm(messages: list[dict[str, str]]) -> str:
        return complete_chat_completion(
            config=config,
            messages=messages,
            model=model,
            temperature=config.planner_temperature,
            max_tokens=max(max_tokens, config.planner_min_output_tokens),
        )

    def execution_llm(messages: list[dict[str, str]]) -> str:
        return complete_chat_completion(
            config=config,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    active_tools = tool_manager.get_active_tool_descriptions()
    knowledge_base = (
        rag_pipeline.describe_for_agent()
        if rag_pipeline is not None
        else {"available": False, "documents": []}
    )
    catalog = CapabilityCatalog(
        tools=frozenset(tool["name"] for tool in active_tools),
        rag_available=bool(knowledge_base.get("available")),
        memory_available=memory_search_fn is not None,
    )
    validator = PlanValidator(
        max_tasks=config.planner_max_tasks,
        max_task_retries=config.planner_max_task_retries,
    )
    planner = Planner(
        llm_fn=planning_llm,
        validator=validator,
        catalog=catalog,
        max_repair_attempts=config.planner_max_repair_attempts,
        default_task_retries=config.planner_max_task_retries,
    )
    runner = TaskRunner(
        llm_fn=execution_llm,
        tool_manager=tool_manager,
        rag_pipeline=rag_pipeline,
        memory_search_fn=memory_search_fn,
        conversation_context=conversation_context,
        memory_context=long_term_memory_context,
        rag_top_k=rag_top_k,
        rag_min_score=rag_min_score,
        approval_service=approval_service,
        approval_user_id=approval_user_id,
    )
    executor = TaskExecutor(
        runner=runner,
        # Stage 9 uses an explicit retry node. The custom runtime deliberately
        # retains its original in-executor retry loop for comparison.
        retry_policy=RetryPolicy(
            0 if graph_managed_task_retries else config.planner_max_task_retries
        ),
        max_execution_steps=config.planner_max_execution_steps,
    )
    replanner = Replanner(
        llm_fn=planning_llm,
        validator=validator,
        catalog=catalog,
        max_repair_attempts=config.planner_max_repair_attempts,
        default_task_retries=config.planner_max_task_retries,
    )
    return PlanningRuntime(
        planner=planner,
        scheduler=TaskScheduler(),
        executor=executor,
        evaluator=GoalEvaluator(planning_llm),
        replanner=replanner,
        catalog=catalog,
        max_plan_revisions=config.planner_max_revisions,
        active_tools=active_tools,
        workflow_persist=(
            lambda state: approval_service.save_workflow(state, user_id=approval_user_id)
            if approval_service is not None
            else None
        ),
    )


def run_planned_agent(
    *,
    config: AppConfig,
    goal: str,
    conversation_context: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    tool_manager: ToolManager,
    rag_pipeline: RagPipeline | None,
    rag_top_k: int,
    rag_min_score: float,
    long_term_memory_context: dict[str, Any] | None,
    memory_search_fn: MemorySearchFn | None,
    status_callback: StatusCallback | None = None,
    cancellation_check: CancellationCheck | None = None,
    approval_service: ApprovalService | None = None,
    approval_user_id: str = "local-user",
    existing_state: PlanState | None = None,
) -> PlanState:
    runtime = build_planning_runtime(
        config=config,
        conversation_context=conversation_context,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        tool_manager=tool_manager,
        rag_pipeline=rag_pipeline,
        rag_top_k=rag_top_k,
        rag_min_score=rag_min_score,
        long_term_memory_context=long_term_memory_context,
        memory_search_fn=memory_search_fn,
        approval_service=approval_service,
        approval_user_id=approval_user_id,
    )
    if existing_state is not None:
        return runtime.resume(
            existing_state,
            status_callback=status_callback,
            cancellation_check=cancellation_check,
        )
    return runtime.run(
        goal=goal,
        conversation_context=conversation_context,
        memory_context=long_term_memory_context,
        knowledge_base=knowledge_base,
        status_callback=status_callback,
        cancellation_check=cancellation_check,
    )
