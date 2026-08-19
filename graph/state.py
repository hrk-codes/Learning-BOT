from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict


class GraphTraceEvent(TypedDict):
    """Safe execution metadata for the developer-facing graph view.

    The trace records node lifecycle data, never hidden model reasoning, prompts,
    API credentials, or full retrieved document text.
    """

    node: str
    status: str
    next_node: str | None
    started_at: str
    duration_seconds: float
    details: dict[str, Any]


class GraphAgentState(TypedDict, total=False):
    """Serializable information shared by Stage 9 graph nodes.

    Graph state describes one workflow run. Long-term memories, documents, and tool
    implementations remain in their existing services; state carries only the
    selected context, identifiers, outputs, and routing facts needed by the graph.
    """

    run_id: str
    thread_id: str
    user_id: str
    goal: str
    conversation_context: list[dict[str, str]]
    memory_context: dict[str, Any] | None
    knowledge_base: dict[str, Any]
    plan_state: dict[str, Any]
    next_task_id: str | None
    last_result: dict[str, Any] | None
    final_answer: str
    status: str
    error: str | None
    graph_retry_count: int
    max_plan_revisions: int
    approval: dict[str, Any] | None
    started_at: str
    completed_at: str | None
    node_trace: Annotated[list[GraphTraceEvent], add]
