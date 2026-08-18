from __future__ import annotations

from collections.abc import Callable
from typing import Any

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
) -> PlanState:
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
    )
    executor = TaskExecutor(
        runner=runner,
        retry_policy=RetryPolicy(config.planner_max_task_retries),
        max_execution_steps=config.planner_max_execution_steps,
    )
    replanner = Replanner(
        llm_fn=planning_llm,
        validator=validator,
        catalog=catalog,
        max_repair_attempts=config.planner_max_repair_attempts,
        default_task_retries=config.planner_max_task_retries,
    )
    runtime = PlanningRuntime(
        planner=planner,
        scheduler=TaskScheduler(),
        executor=executor,
        evaluator=GoalEvaluator(planning_llm),
        replanner=replanner,
        catalog=catalog,
        max_plan_revisions=config.planner_max_revisions,
        active_tools=active_tools,
    )
    return runtime.run(
        goal=goal,
        conversation_context=conversation_context,
        memory_context=long_term_memory_context,
        knowledge_base=knowledge_base,
        status_callback=status_callback,
        cancellation_check=cancellation_check,
    )

