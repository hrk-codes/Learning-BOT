from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from graph.nodes import GraphDependencies, build_nodes
from graph.routing import (
    route_after_evaluation,
    route_after_execution,
    route_after_task_router,
)
from graph.state import GraphAgentState


def build_agent_graph(dependencies: GraphDependencies, checkpointer):
    """Define the Stage 9 workflow blueprint separately from any one execution.

    Nodes are the meaningful units of work. Edges reveal the orchestration topology:
    scheduling, task execution, retry loops, human interruption, evaluation, and
    bounded replanning. The compiled graph plus a thread ID becomes one graph run.
    """

    nodes = build_nodes(dependencies)
    workflow = StateGraph(GraphAgentState)
    for name, node in nodes.items():
        workflow.add_node(name, node)

    workflow.add_edge(START, "planner")
    # The planner creates a valid DAG; the router decides which currently-ready
    # task may run instead of embedding scheduler control flow inside the planner.
    workflow.add_edge("planner", "task_router")
    workflow.add_conditional_edges(
        "task_router",
        route_after_task_router,
        {
            "execute_task": "execute_task",
            "evaluate": "evaluate",
            "finalize": "finalize",
        },
    )
    # A task outcome may pause at approval, enter a bounded retry cycle, or return
    # to scheduling so dependency state remains explicit and inspectable.
    workflow.add_conditional_edges(
        "execute_task",
        route_after_execution,
        {
            "approval": "approval",
            "retry_task": "retry_task",
            "task_router": "task_router",
        },
    )
    workflow.add_edge("retry_task", "execute_task")
    # The approval node calls interrupt() while the request is pending. A resume
    # re-enters the exact task so the existing approval/idempotency checks run again.
    workflow.add_edge("approval", "execute_task")
    workflow.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
        {"replan": "replan", "finalize": "finalize"},
    )
    workflow.add_edge("replan", "task_router")
    workflow.add_edge("finalize", END)
    return workflow.compile(checkpointer=checkpointer)
