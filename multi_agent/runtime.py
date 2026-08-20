from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from typing import Any, Callable

from config import AppConfig
from graph.checkpoints import build_sqlite_checkpointer
from llm.groq_client import complete_chat_completion
from multi_agent.agents.base import AgentConfig, LLMCall
from multi_agent.agents.manager import ManagerAgent
from multi_agent.agents.researcher import ResearcherAgent
from multi_agent.agents.reviewer import ReviewerAgent
from multi_agent.agents.writer import WriterAgent
from multi_agent.graph import build_multi_agent_graph
from multi_agent.nodes import MultiAgentDependencies
from multi_agent.state import MultiAgentState
from planner.models import utc_now_iso
from prompts.manager_prompt import MANAGER_SYSTEM_PROMPT
from prompts.researcher_prompt import RESEARCHER_SYSTEM_PROMPT
from prompts.reviewer_prompt import REVIEWER_SYSTEM_PROMPT
from prompts.writer_prompt import WRITER_SYSTEM_PROMPT
from rag.pipeline import RagPipeline
from tools.manager import ToolManager


@dataclass(frozen=True)
class MultiAgentRunResult:
    thread_id: str
    run_id: str
    values: dict[str, Any]

    @property
    def final_answer(self) -> str:
        return str(self.values.get("final_answer", ""))


class MultiAgentRuntime:
    """Build one Stage 10 workflow around live services and durable checkpoints."""

    def __init__(
        self,
        *,
        config: AppConfig,
        model: str,
        temperature: float,
        max_tokens: int,
        tool_manager: ToolManager,
        rag_pipeline: RagPipeline | None,
        rag_top_k: int,
        rag_min_score: float,
        llm_call: LLMCall | None = None,
    ) -> None:
        self.config = config
        self.tool_manager = tool_manager
        call = llm_call or self._build_llm_call(model, temperature, max_tokens)
        timeout = config.multi_agent_timeout_seconds
        retries = config.multi_agent_output_repair_attempts
        self.manager = ManagerAgent(
            AgentConfig("manager", MANAGER_SYSTEM_PROMPT, model, 0.1, max_tokens, timeout, retries), call
        )
        self.researcher = ResearcherAgent(
            AgentConfig(
                "researcher", RESEARCHER_SYSTEM_PROMPT, model, 0.1, max_tokens, timeout, retries,
                allowed_tools=("search.web",), allow_rag=True,
            ),
            call,
        )
        self.writer = WriterAgent(
            AgentConfig("writer", WRITER_SYSTEM_PROMPT, model, temperature, max_tokens, timeout, retries, allow_memory=True),
            call,
        )
        self.reviewer = ReviewerAgent(
            AgentConfig("reviewer", REVIEWER_SYSTEM_PROMPT, model, 0.0, max_tokens, timeout, retries), call
        )
        self.checkpointer = build_sqlite_checkpointer(config.multi_agent_checkpoint_db_path)
        self.graph = build_multi_agent_graph(
            MultiAgentDependencies(
                manager=self.manager,
                researcher=self.researcher,
                writer=self.writer,
                reviewer=self.reviewer,
                rag_pipeline=rag_pipeline,
                rag_top_k=rag_top_k,
                rag_min_score=rag_min_score,
                tool_manager=tool_manager,
            ),
            self.checkpointer,
        )

    def start(
        self,
        *,
        goal: str,
        user_id: str,
        conversation_context: list[dict[str, str]],
        memory_context: dict[str, Any] | None,
        knowledge_base: dict[str, Any],
    ) -> MultiAgentRunResult:
        run_id = f"run_{uuid.uuid4().hex}"
        thread_id = f"stage10_{uuid.uuid4().hex}"
        researcher_tools = [
            tool.name for tool in self.tool_manager.list_active_tools()
            if tool.name == "search.web"
        ]
        initial: MultiAgentState = {
            "run_id": run_id,
            "thread_id": thread_id,
            "user_id": user_id,
            "goal": goal,
            "conversation_context": conversation_context[-10:],
            "memory_context": memory_context,
            "knowledge_base": knowledge_base,
            "researcher_tools": researcher_tools,
            "current_agent": None,
            "manager_decision": None,
            "current_task": None,
            "research_result": None,
            "draft_result": None,
            "review_result": None,
            "agent_attempts": {},
            "delegation_count": 0,
            "revision_count": 0,
            "max_delegations": self.config.multi_agent_max_delegations,
            "max_agent_retries": self.config.multi_agent_max_agent_retries,
            "max_review_revisions": self.config.multi_agent_max_review_revisions,
            "final_answer": "",
            "status": "running",
            "error": None,
            "started_at": utc_now_iso(),
            "completed_at": None,
            "agent_results": [],
            "node_trace": [],
        }
        self.graph.invoke(initial, self._graph_config(thread_id))
        return self.get_run(thread_id)

    def get_run(self, thread_id: str) -> MultiAgentRunResult:
        snapshot = self.graph.get_state(self._graph_config(thread_id))
        values = dict(snapshot.values)
        return MultiAgentRunResult(
            thread_id=thread_id,
            run_id=str(values.get("run_id", thread_id)),
            values=values,
        )

    def _build_llm_call(self, model: str, temperature: float, max_tokens: int) -> LLMCall:
        def call(messages: list[dict[str, str]], timeout_seconds: int) -> str:
            # Each specialist gets a bounded provider timeout. The deadline is a
            # runtime contract, not a suggestion left to an individual prompt.
            scoped_config = replace(self.config, request_timeout_seconds=timeout_seconds)
            return complete_chat_completion(
                config=scoped_config,
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        return call

    def _graph_config(self, thread_id: str) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": max(40, self.config.multi_agent_max_delegations * 4 + 8),
        }
