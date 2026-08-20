from __future__ import annotations

import time
from typing import Any

from multi_agent.agents.base import AgentConfig, AgentExecutionError, BaseAgent
from multi_agent.agents.contracts import DelegatedTask, WritingResult
from multi_agent.agents.results import AgentResult
from llm.groq_client import GroqClientError


class WriterAgent(BaseAgent):
    def run(
        self,
        task: DelegatedTask,
        *,
        research: dict[str, Any] | None,
        style_memory: list[dict[str, Any]],
        revision_feedback: dict[str, Any] | None,
        retry_count: int,
    ) -> AgentResult:
        started = time.perf_counter()
        try:
            # The writer gets selected style preferences and normalized research only.
            # It never receives raw RAG chunks, all stored memories, or tool access.
            payload = {
                "task": task.to_dict(),
                "research": research or {"claims": [], "sources": [], "gaps": ["No research was supplied."]},
                "style_preferences": style_memory,
                "revision_feedback": revision_feedback,
            }
            result = self.request_json(payload, WritingResult.from_dict)
            return AgentResult(
                agent_name=task.assigned_agent,
                task_id=task.task_id,
                status="completed",
                output=result.to_dict(),
                duration_seconds=time.perf_counter() - started,
                retry_count=retry_count,
                metadata={"rag_used": False, "tools_used": [], "model": self.config.model},
            )
        except (AgentExecutionError, GroqClientError, ValueError) as exc:
            return AgentResult(
                agent_name=task.assigned_agent,
                task_id=task.task_id,
                status="failed",
                output=None,
                error=str(exc),
                duration_seconds=time.perf_counter() - started,
                retry_count=retry_count,
                metadata={"rag_used": False, "tools_used": [], "model": self.config.model},
            )
