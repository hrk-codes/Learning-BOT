from config import AppConfig
from agent.agent_loop import run_agent_loop
from agent.agent_state import AgentState
from llm.groq_client import complete_chat_completion
from tools.manager import ToolManager
from rag.pipeline import RagPipeline


def run_agent(
    config: AppConfig,
    goal: str,
    conversation_context: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    tool_manager: ToolManager,
    rag_pipeline: RagPipeline | None = None,
    rag_top_k: int = 4,
    rag_min_score: float = 0.25,
    long_term_memory_context: dict | None = None,
    memory_metrics: dict | None = None,
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
        tool_manager=tool_manager,
        rag_pipeline=rag_pipeline,
        rag_top_k=rag_top_k,
        rag_min_score=rag_min_score,
        long_term_memory_context=long_term_memory_context,
        memory_metrics=memory_metrics,
    )
