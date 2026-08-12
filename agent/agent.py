from config import AppConfig
from agent.agent_loop import run_agent_loop
from agent.agent_state import AgentState
from llm.groq_client import complete_chat_completion


def run_agent(
    config: AppConfig,
    goal: str,
    conversation_context: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> AgentState:
    def ask_llm(messages: list[dict[str, str]]) -> str:
        return complete_chat_completion(
            config=config,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return run_agent_loop(
        goal=goal,
        conversation_context=conversation_context,
        max_iterations=config.max_agent_iterations,
        llm_decision_fn=ask_llm,
    )
