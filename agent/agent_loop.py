import logging
import json
from collections.abc import Callable

from agent.action_router import route_action
from agent.agent_state import AgentState, AgentTraceStep
from agent.decision_schema import DecisionParseError, parse_agent_decision
from prompts.agent_prompt import AGENT_SYSTEM_PROMPT
from tools.manager import ToolManager


logger = logging.getLogger(__name__)

LLMDecisionFn = Callable[[list[dict[str, str]]], str]


class AgentRuntimeError(Exception):
    """Raised when the agent loop cannot safely continue."""


def run_agent_loop(
    goal: str,
    conversation_context: list[dict[str, str]],
    max_iterations: int,
    llm_decision_fn: LLMDecisionFn,
    tool_manager: ToolManager,
) -> AgentState:
    state = AgentState(goal=goal, max_iterations=max_iterations)
    logger.info("AGENT START goal=%s max_iterations=%s", goal, max_iterations)

    while state.status == "running":
        if state.iteration_count >= state.max_iterations:
            state.status = "max_iterations"
            state.final_answer = _build_max_iteration_answer(state)
            logger.warning("AGENT STOP max_iterations=%s", state.max_iterations)
            break

        # Track the runtime cycle count. The loop is orchestration, not
        # intelligence; this counter prevents the orchestration from running
        # forever if the model keeps choosing non-final actions.
        state.iteration_count += 1
        observation = observe(state)
        state.record_observation(observation)

        try:
            decision = decide(
                state=state,
                conversation_context=conversation_context,
                observation=observation,
                llm_decision_fn=llm_decision_fn,
                tool_manager=tool_manager,
            )
            route_action(state, decision, tool_manager)
        except (DecisionParseError, AgentRuntimeError) as exc:
            state.status = "error"
            state.final_answer = f"The agent stopped safely because its decision step failed: {exc}"
            state.record_trace(
                AgentTraceStep(
                    iteration=state.iteration_count,
                    observation=observation,
                    action="ERROR",
                    status="decision failure",
                    content=str(exc),
                )
            )
            logger.error("AGENT ERROR %s", exc)
            break

    logger.info("AGENT END status=%s iterations=%s", state.status, state.iteration_count)
    return state


def observe(state: AgentState) -> str:
    if not state.action_results:
        return f"Starting goal: {state.goal}"
    return f"Previous step completed. Latest result: {state.action_results[-1]}"


def decide(
    state: AgentState,
    conversation_context: list[dict[str, str]],
    observation: str,
    llm_decision_fn: LLMDecisionFn,
    tool_manager: ToolManager,
):
    context = build_agent_context(state, conversation_context, observation, tool_manager)
    state.llm_call_count += 1
    raw_decision = llm_decision_fn(context)
    return parse_agent_decision(raw_decision)


def build_agent_context(
    state: AgentState,
    conversation_context: list[dict[str, str]],
    observation: str,
    tool_manager: ToolManager,
) -> list[dict[str, str]]:
    visible_trace = [
        {
            "iteration": step.iteration,
            "action": step.action,
            "status": step.status,
            "content": step.content,
            "tool_name": step.tool_name,
            "tool_result": step.tool_result,
        }
        for step in state.trace
    ]

    # The LLM receives selected state, not the entire Python object. This keeps
    # context construction explicit and prepares the same place where Stage 4
    # will later add tool results.
    agent_state_summary = {
        "goal": state.goal,
        "iteration_count": state.iteration_count,
        "max_iterations": state.max_iterations,
        "current_observation": observation,
        "previous_steps": visible_trace,
        "active_tools": tool_manager.get_active_tool_descriptions(),
    }

    return [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        *conversation_context,
        {
            "role": "user",
            "content": (
                "Make the next agent decision from this state. "
                "Return only JSON.\n\n"
                "If a tool is needed, use action=TOOL_CALL with tool_name and tool_arguments. "
                "Only request tools listed in active_tools.\n\n"
                f"Agent state:\n{json.dumps(agent_state_summary, indent=2)}"
            ),
        },
    ]


def _build_max_iteration_answer(state: AgentState) -> str:
    if state.action_results:
        partial = "\n\n".join(state.action_results)
        return (
            "The agent reached its maximum iteration limit before a FINISH decision. "
            "Here is the useful work completed so far:\n\n"
            f"{partial}"
        )
    return "The agent reached its maximum iteration limit before producing a final answer."
