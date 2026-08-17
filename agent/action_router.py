import json
from typing import TYPE_CHECKING

from agent.agent_state import AgentState, AgentTraceStep
from agent.decision_schema import AgentDecision
from tools.manager import ToolManager

if TYPE_CHECKING:
    from rag.pipeline import RagPipeline


def route_action(
    state: AgentState,
    decision: AgentDecision,
    tool_manager: ToolManager,
    rag_pipeline: "RagPipeline | None" = None,
    rag_top_k: int = 4,
    rag_min_score: float = 0.25,
) -> None:
    if decision.action == "RETRIEVE_KNOWLEDGE":
        _execute_knowledge_retrieval(
            state,
            decision,
            rag_pipeline,
            rag_top_k,
            rag_min_score,
        )
        return
    if decision.action == "TOOL_CALL":
        _execute_tool_call(state, decision, tool_manager)
        return

    _execute_internal_action(state, decision)


def _execute_knowledge_retrieval(
    state: AgentState,
    decision: AgentDecision,
    rag_pipeline: "RagPipeline | None",
    top_k: int,
    min_score: float,
) -> None:
    from rag.context.context_builder import build_knowledge_observation
    from rag.pipeline import RagPipelineError

    assert decision.retrieval_query is not None

    if rag_pipeline is None:
        observation_payload = {
            "success": False,
            "query": decision.retrieval_query,
            "evidence_found": False,
            "error": "Knowledge retrieval is unavailable.",
            "chunks": [],
        }
        latency = 0.0
        raw_chunks: list[dict] = []
    else:
        try:
            result = rag_pipeline.retrieve(
                decision.retrieval_query,
                top_k=top_k,
                min_score=min_score,
            )
            observation_payload = build_knowledge_observation(
                result,
                max_context_chars=rag_pipeline.max_context_chars,
            )
            latency = result.total_seconds
            raw_chunks = [chunk.to_dict() for chunk in result.chunks]
        except RagPipelineError as exc:
            observation_payload = {
                "success": False,
                "query": decision.retrieval_query,
                "evidence_found": False,
                "error": str(exc),
                "chunks": [],
            }
            latency = 0.0
            raw_chunks = []

    # Retrieval is an observation, just like a tool result. It returns evidence
    # to the loop but never bypasses the model/runtime boundary to become an answer.
    state.rag_retrieval_count += 1
    state.total_rag_latency_seconds += latency
    state.retrieved_chunks.extend(raw_chunks)
    observation = json.dumps(observation_payload, ensure_ascii=False)
    state.record_action_result(f"Knowledge observation: {observation}")
    state.record_trace(
        AgentTraceStep(
            iteration=state.iteration_count,
            observation=state.observations[-1],
            action=decision.action,
            status=decision.status,
            content=decision.content,
            retrieval_query=decision.retrieval_query,
            retrieved_chunks=raw_chunks,
            retrieval_latency_seconds=latency,
        )
    )


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
