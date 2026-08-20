from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict


class MultiAgentTraceEvent(TypedDict):
    """Safe execution metadata; never prompts, hidden reasoning, or source text."""

    node: str
    agent_name: str
    status: str
    next_node: str | None
    task_id: str | None
    duration_seconds: float
    details: dict[str, Any]


class MultiAgentState(TypedDict, total=False):
    """Compact shared workflow artifacts for the Stage 10 manager-led team."""

    run_id: str
    thread_id: str
    user_id: str
    goal: str
    conversation_context: list[dict[str, str]]
    memory_context: dict[str, Any] | None
    knowledge_base: dict[str, Any]
    researcher_tools: list[str]
    needs_research: bool
    needs_writing: bool
    needs_review: bool
    current_agent: str | None
    manager_decision: dict[str, Any] | None
    current_task: dict[str, Any] | None
    research_result: dict[str, Any] | None
    draft_result: dict[str, Any] | None
    review_result: dict[str, Any] | None
    agent_attempts: dict[str, int]
    delegation_count: int
    revision_count: int
    max_delegations: int
    max_agent_retries: int
    max_review_revisions: int
    final_answer: str
    status: str
    error: str | None
    started_at: str
    completed_at: str | None
    agent_results: Annotated[list[dict[str, Any]], add]
    node_trace: Annotated[list[MultiAgentTraceEvent], add]
