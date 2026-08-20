from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from multi_agent.agents.manager import ManagerAgent, classify_goal
from multi_agent.agents.contracts import AgentName, ManagerAction
from multi_agent.agents.researcher import ResearcherAgent
from multi_agent.agents.reviewer import ReviewerAgent
from multi_agent.agents.writer import WriterAgent
from multi_agent.state import MultiAgentState
from planner.models import utc_now_iso
from rag.pipeline import RagPipeline
from tools.manager import ToolManager


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MultiAgentDependencies:
    manager: ManagerAgent
    researcher: ResearcherAgent
    writer: WriterAgent
    reviewer: ReviewerAgent
    rag_pipeline: RagPipeline | None
    rag_top_k: int
    rag_min_score: float
    tool_manager: ToolManager


def build_nodes(dependencies: MultiAgentDependencies) -> dict[str, Any]:
    def manager(state: MultiAgentState) -> dict[str, Any]:
        started = time.perf_counter()
        updates: dict[str, Any] = {}
        if "needs_research" not in state:
            updates.update(classify_goal(state["goal"], state.get("knowledge_base", {})))
        decision = dependencies.manager.decide({**state, **updates})
        task = decision.task.to_dict() if decision.task else None
        next_node = _next_node(decision.action)
        updates.update(
            {
                "current_agent": AgentName.MANAGER.value,
                "manager_decision": decision.to_dict(),
                "current_task": task,
                "status": "running" if decision.action != ManagerAction.FINISH else "synthesizing",
            }
        )
        return _node_update(
            node="manager",
            agent_name=AgentName.MANAGER.value,
            started=started,
            status="completed",
            next_node=next_node,
            task_id=task.get("task_id") if task else None,
            details={"action": decision.action.value, "delegation_count": state.get("delegation_count", 0)},
            updates=updates,
        )

    def researcher(state: MultiAgentState) -> dict[str, Any]:
        started = time.perf_counter()
        task = _task_for(state, AgentName.RESEARCHER)
        attempts = _increment_attempts(state, AgentName.RESEARCHER)
        result = dependencies.researcher.run(
            task,
            rag_pipeline=dependencies.rag_pipeline,
            rag_top_k=dependencies.rag_top_k,
            rag_min_score=dependencies.rag_min_score,
            tool_manager=dependencies.tool_manager,
            retry_count=attempts - 1,
        )
        # New evidence invalidates a prior review and draft. The manager must send
        # the revised artifact through the writer/reviewer sequence again.
        return _specialist_update(
            state,
            node="researcher",
            result=result.to_dict(),
            result_key="research_result",
            started=started,
            attempts=attempts,
            extra_updates={"draft_result": None, "review_result": None},
        )

    def writer(state: MultiAgentState) -> dict[str, Any]:
        started = time.perf_counter()
        task = _task_for(state, AgentName.WRITER)
        attempts = _increment_attempts(state, AgentName.WRITER)
        result = dependencies.writer.run(
            task,
            research=_output(state.get("research_result")),
            style_memory=_writer_memory(state.get("memory_context")),
            revision_feedback=_output(state.get("review_result")),
            retry_count=attempts - 1,
        )
        was_revision = (state.get("manager_decision") or {}).get("action") == ManagerAction.REVISE.value
        return _specialist_update(
            state,
            node="writer",
            result=result.to_dict(),
            result_key="draft_result",
            started=started,
            attempts=attempts,
            extra_updates={
                "review_result": None,
                "revision_count": int(state.get("revision_count", 0)) + int(was_revision),
            },
        )

    def reviewer(state: MultiAgentState) -> dict[str, Any]:
        started = time.perf_counter()
        task = _task_for(state, AgentName.REVIEWER)
        attempts = _increment_attempts(state, AgentName.REVIEWER)
        draft = _output(state.get("draft_result"))
        if not draft:
            return _node_update(
                node="reviewer",
                agent_name=AgentName.REVIEWER.value,
                started=started,
                status="failed",
                next_node="manager",
                task_id=task.task_id,
                details={"error_type": "missing_draft"},
                updates={"error": "Reviewer could not run because no draft was available."},
            )
        result = dependencies.reviewer.run(
            task,
            draft=draft,
            research=_output(state.get("research_result")),
            retry_count=attempts - 1,
        )
        return _specialist_update(
            state,
            node="reviewer",
            result=result.to_dict(),
            result_key="review_result",
            started=started,
            attempts=attempts,
        )

    def finalize(state: MultiAgentState) -> dict[str, Any]:
        started = time.perf_counter()
        final_answer = dependencies.manager.synthesize(state)
        return _node_update(
            node="finalize",
            agent_name=AgentName.MANAGER.value,
            started=started,
            status="completed",
            next_node=None,
            task_id=None,
            details={"agent_result_count": len(state.get("agent_results", []))},
            updates={"final_answer": final_answer, "status": "completed", "completed_at": utc_now_iso()},
        )

    return {
        "manager": manager,
        "researcher": researcher,
        "writer": writer,
        "reviewer": reviewer,
        "finalize": finalize,
    }


def _next_node(action: ManagerAction) -> str:
    return {
        ManagerAction.DELEGATE_RESEARCH: "researcher",
        ManagerAction.DELEGATE_WRITING: "writer",
        ManagerAction.DELEGATE_REVIEW: "reviewer",
        ManagerAction.REVISE: "writer",
        ManagerAction.FINISH: "finalize",
    }[action]


def _task_for(state: MultiAgentState, assigned: AgentName):
    from multi_agent.agents.contracts import DelegatedTask

    raw_task = state.get("current_task")
    if not isinstance(raw_task, dict) or raw_task.get("assigned_agent") != assigned.value:
        raise ValueError(f"The manager did not provide a valid task for {assigned.value}.")
    return DelegatedTask(
        task_id=str(raw_task["task_id"]),
        assigned_agent=assigned,
        goal=str(raw_task["goal"]),
        expected_output=str(raw_task["expected_output"]),
        constraints=tuple(str(item) for item in raw_task.get("constraints", [])),
        allowed_tools=tuple(str(item) for item in raw_task.get("allowed_tools", [])),
        use_rag=bool(raw_task.get("use_rag")),
    )


def _increment_attempts(state: MultiAgentState, agent: AgentName) -> int:
    return int((state.get("agent_attempts") or {}).get(agent.value, 0)) + 1


def _specialist_update(
    state: MultiAgentState,
    *,
    node: str,
    result: dict[str, Any],
    result_key: str,
    started: float,
    attempts: int,
    extra_updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    agent_name = str(result["agent_name"])
    attempts_by_agent = dict(state.get("agent_attempts") or {})
    attempts_by_agent[agent_name] = attempts
    updates = {
        result_key: result,
        "agent_attempts": attempts_by_agent,
        "delegation_count": int(state.get("delegation_count", 0)) + 1,
        "current_agent": agent_name,
    }
    if extra_updates:
        updates.update(extra_updates)
    return _node_update(
        node=node,
        agent_name=agent_name,
        started=started,
        status=result["status"],
        next_node="manager",
        task_id=str(result["task_id"]),
        details={
            "retry_count": result["retry_count"],
            "rag_used": bool(result.get("metadata", {}).get("rag_used")),
            "tools_used": result.get("metadata", {}).get("tools_used", []),
            "error_type": type(result.get("error")).__name__ if result.get("error") else None,
        },
        updates={**updates, "agent_results": [result]},
    )


def _node_update(
    *,
    node: str,
    agent_name: str,
    started: float,
    status: str,
    next_node: str | None,
    task_id: str | None,
    details: dict[str, Any],
    updates: dict[str, Any],
) -> dict[str, Any]:
    return {
        **updates,
        "node_trace": [
            {
                "node": node,
                "agent_name": agent_name,
                "status": status,
                "next_node": next_node,
                "task_id": task_id,
                "duration_seconds": round(time.perf_counter() - started, 4),
                "details": details,
            }
        ],
    }


def _output(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result and result.get("status") == "completed" and isinstance(result.get("output"), dict):
        return result["output"]
    return None


def _writer_memory(context: dict[str, Any] | None) -> list[dict[str, Any]]:
    records = context.get("records", []) if isinstance(context, dict) else []
    return [
        {"type": item.get("type"), "fact": item.get("fact")}
        for item in records
        if isinstance(item, dict) and item.get("type") in {"profile", "procedural"}
    ]
