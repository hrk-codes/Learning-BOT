from __future__ import annotations

import time
from typing import Any

from multi_agent.agents.base import AgentConfig, AgentExecutionError, BaseAgent
from multi_agent.agents.contracts import DelegatedTask, ReviewResult
from multi_agent.agents.results import AgentResult
from llm.groq_client import GroqClientError


class ReviewerAgent(BaseAgent):
    def run(
        self,
        task: DelegatedTask,
        *,
        draft: dict[str, Any],
        research: dict[str, Any] | None,
        retry_count: int,
    ) -> AgentResult:
        started = time.perf_counter()
        try:
            payload = {
                "task": task.to_dict(),
                "draft": draft,
                "research": research or {"claims": [], "sources": [], "gaps": ["No research was supplied."]},
                "criteria": ["instruction following", "evidence adherence", "clear limitations"],
            }
            result = self.request_json(payload, ReviewResult.from_dict)
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
