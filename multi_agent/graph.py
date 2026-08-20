from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from multi_agent.nodes import MultiAgentDependencies, build_nodes
from multi_agent.routing import route_after_manager
from multi_agent.state import MultiAgentState


def build_multi_agent_graph(dependencies: MultiAgentDependencies, checkpointer):
    """Build one graph with specialist nodes, not nested agent graphs.

    The manager remains the only routing authority. Specialists always return to it
    through structured artifacts, which keeps responsibility and recovery explicit.
    """

    workflow = StateGraph(MultiAgentState)
    for name, node in build_nodes(dependencies).items():
        workflow.add_node(name, node)
    workflow.add_edge(START, "manager")
    workflow.add_conditional_edges(
        "manager",
        route_after_manager,
        {
            "researcher": "researcher",
            "writer": "writer",
            "reviewer": "reviewer",
            "finalize": "finalize",
        },
    )
    workflow.add_edge("researcher", "manager")
    workflow.add_edge("writer", "manager")
    workflow.add_edge("reviewer", "manager")
    workflow.add_edge("finalize", END)
    return workflow.compile(checkpointer=checkpointer)
