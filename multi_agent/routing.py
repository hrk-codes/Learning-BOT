from __future__ import annotations

from multi_agent.state import MultiAgentState


def route_after_manager(state: MultiAgentState) -> str:
    """Use the manager's validated, limited action rather than arbitrary model text."""

    decision = state.get("manager_decision") or {}
    action = decision.get("action")
    return {
        "delegate_research": "researcher",
        "delegate_writing": "writer",
        "delegate_review": "reviewer",
        "revise": "writer",
        "finish": "finalize",
    }.get(action, "finalize")
