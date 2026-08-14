import json

from agent.agent_state import AgentState, AgentTraceStep
from agent.decision_schema import AgentDecision
from tools.manager import ToolManager


def route_action(state: AgentState, decision: AgentDecision, tool_manager: ToolManager) -> None:
    if decision.action == "TOOL_CALL":
        _execute_tool_call(state, decision, tool_manager)
        return

    _execute_internal_action(state, decision)


def _execute_internal_action(state: AgentState, decision: AgentDecision) -> None:
    state.record_action_result(decision.content)
    state.record_trace(
        AgentTraceStep(
            iteration=state.iteration_count,
            observation=state.observations[-1],
            action=decision.action,
            status=decision.status,
            content=decision.content,
        )
    )

    if decision.finished:
        state.status = "completed"
        state.final_answer = decision.content


def _execute_tool_call(state: AgentState, decision: AgentDecision, tool_manager: ToolManager) -> None:
    assert decision.tool_name is not None
    assert decision.tool_arguments is not None

    # Tool results become observations, not final answers. The model must see
    # this structured result on the next iteration and decide how to respond.
    tool_result = tool_manager.execute_tool(decision.tool_name, decision.tool_arguments)
    state.tool_call_count += 1
    state.total_tool_latency_seconds += tool_result.elapsed_seconds

    observation_payload = {
        "tool": decision.tool_name,
        "arguments": decision.tool_arguments,
        "success": tool_result.success,
        "result": tool_result.result,
        "error": tool_result.error,
        "metadata": tool_result.metadata,
        "elapsed_seconds": round(tool_result.elapsed_seconds, 4),
    }
    observation = json.dumps(observation_payload, ensure_ascii=False)
    state.record_action_result(f"Tool observation: {observation}")
    state.record_trace(
        AgentTraceStep(
            iteration=state.iteration_count,
            observation=state.observations[-1],
            action=decision.action,
            status=decision.status,
            content=decision.content,
            tool_name=decision.tool_name,
            tool_arguments=decision.tool_arguments,
            tool_result=observation_payload,
            elapsed_seconds=tool_result.elapsed_seconds,
        )
    )
