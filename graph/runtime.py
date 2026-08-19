from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.types import Command

from agent.planned_agent import build_planning_runtime
from approval.service import ApprovalService
from config import AppConfig
from graph.checkpoints import build_sqlite_checkpointer
from graph.graph import build_agent_graph
from graph.nodes import GraphDependencies
from graph.state import GraphAgentState
from planner.models import PlanState
from planner.serialization import plan_state_from_dict, plan_state_to_dict
from rag.pipeline import RagPipeline
from tools.manager import ToolManager


@dataclass(frozen=True)
class GraphRunResult:
    thread_id: str
    run_id: str
    plan_state: PlanState
    values: dict[str, Any]
    next_nodes: tuple[str, ...]
    interrupted: bool


class LangGraphPlannedAgent:
    """Stage 9 adapter that turns existing planning services into a durable graph run."""

    def __init__(
        self,
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
        memory_search_fn,
        approval_service: ApprovalService | None,
        approval_user_id: str,
    ) -> None:
        self.config = config
        self.approval_user_id = approval_user_id
        planning_runtime = build_planning_runtime(
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
            graph_managed_task_retries=True,
        )
        self.checkpointer = build_sqlite_checkpointer(config.langgraph_checkpoint_db_path)
        self.graph = build_agent_graph(
            GraphDependencies(
                planning_runtime=planning_runtime,
                approval_service=approval_service,
                approval_user_id=approval_user_id,
                max_task_retries=config.planner_max_task_retries,
            ),
            self.checkpointer,
        )

    def start(
        self,
        *,
        goal: str,
        conversation_context: list[dict[str, str]],
        memory_context: dict[str, Any] | None,
        knowledge_base: dict[str, Any],
    ) -> GraphRunResult:
        run_id = f"run_{uuid.uuid4().hex}"
        thread_id = f"stage9_{uuid.uuid4().hex}"
        initial: GraphAgentState = {
            "run_id": run_id,
            "thread_id": thread_id,
            "user_id": self.approval_user_id,
            "goal": goal,
            "conversation_context": conversation_context[-10:],
            "memory_context": memory_context,
            "knowledge_base": knowledge_base,
            "plan_state": _empty_plan_payload(goal),
            "next_task_id": None,
            "last_result": None,
            "final_answer": "",
            "status": "running",
            "error": None,
            "graph_retry_count": 0,
            "approval": None,
            "started_at": _utc_now(),
            "completed_at": None,
            "node_trace": [],
            "max_plan_revisions": self.config.planner_max_revisions,
        }
        self.graph.invoke(initial, self._graph_config(thread_id))
        return self.get_run(thread_id)

    def resume(self, thread_id: str) -> GraphRunResult:
        self.graph.invoke(Command(resume={"continue": True}), self._graph_config(thread_id))
        return self.get_run(thread_id)

    def get_run(self, thread_id: str) -> GraphRunResult:
        snapshot = self.graph.get_state(self._graph_config(thread_id))
        values = dict(snapshot.values)
        payload = values.get("plan_state")
        if not isinstance(payload, dict):
            raise ValueError("The graph checkpoint does not contain a valid plan state.")
        return GraphRunResult(
            thread_id=thread_id,
            run_id=str(values.get("run_id", thread_id)),
            plan_state=plan_state_from_dict(payload),
            values=values,
            next_nodes=tuple(snapshot.next),
            interrupted="approval" in snapshot.next,
        )

    def update_plan_state(self, thread_id: str, plan_state: PlanState) -> None:
        """Synchronize a reviewed Stage 8 edit before the paused graph resumes."""

        self.graph.update_state(
            self._graph_config(thread_id),
            {"plan_state": plan_state_to_dict(plan_state)},
        )

    def find_waiting_run(self, *, user_id: str) -> GraphRunResult | None:
        """Find the newest paused approval run for one local user identity."""

        seen_threads: set[str] = set()
        for checkpoint in self.checkpointer.list(None):
            configurable = checkpoint.config.get("configurable", {})
            thread_id = configurable.get("thread_id")
            if not isinstance(thread_id, str) or thread_id in seen_threads:
                continue
            seen_threads.add(thread_id)
            values = checkpoint.checkpoint.get("channel_values", {})
            if values.get("user_id") != user_id:
                continue
            try:
                run = self.get_run(thread_id)
            except (ValueError, KeyError):
                continue
            if run.interrupted:
                return run
        return None

    def _graph_config(self, thread_id: str) -> dict[str, Any]:
        # The thread ID is the durable pointer to one graph execution. A generous
        # recursion bound covers task, retry, evaluation, and replan cycles while
        # the planner/executor still enforce their smaller business limits.
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": max(100, self.config.planner_max_execution_steps * 8),
        }


def _empty_plan_payload(goal: str) -> dict[str, Any]:
    from planner.models import PlanState

    return plan_state_to_dict(PlanState(goal=goal, tasks=[]))


def _utc_now() -> str:
    from planner.models import utc_now_iso

    return utc_now_iso()
