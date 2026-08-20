from __future__ import annotations

import time
from typing import Any

from multi_agent.agents.base import AgentConfig, AgentExecutionError, BaseAgent
from multi_agent.agents.contracts import DelegatedTask, ResearchResult
from multi_agent.agents.results import AgentResult
from llm.groq_client import GroqClientError
from rag.pipeline import RagPipeline, RagPipelineError
from tools.manager import ToolManager


class ResearcherAgent(BaseAgent):
    def run(
        self,
        task: DelegatedTask,
        *,
        rag_pipeline: RagPipeline | None,
        rag_top_k: int,
        rag_min_score: float,
        tool_manager: ToolManager,
        retry_count: int,
    ) -> AgentResult:
        started = time.perf_counter()
        evidence: list[dict[str, Any]] = []
        rag_used = False
        tools_used: list[str] = []
        try:
            if task.use_rag and rag_pipeline is not None:
                retrieval = rag_pipeline.retrieve(task.goal, top_k=rag_top_k, min_score=rag_min_score)
                rag_used = True
                for chunk in retrieval.chunks:
                    evidence.append(
                        {
                            "source_id": chunk.chunk_id,
                            "label": f"{chunk.metadata.get('filename', 'Document')} page {chunk.metadata.get('page_number', '?')}",
                            "kind": "rag_chunk",
                            "excerpt": chunk.text[:1400],
                            "score": round(chunk.score, 4),
                        }
                    )

            # Search is intentionally the only Stage 4 capability exposed here.
            # The writer and reviewer never receive ToolManager access.
            if not evidence and "search.web" in task.allowed_tools:
                search = tool_manager.execute_tool("search.web", {"query": task.goal})
                tools_used.append("search.web")
                if search.success and isinstance(search.result, dict):
                    for index, item in enumerate(search.result.get("results", [])):
                        if isinstance(item, dict):
                            evidence.append(
                                {
                                    "source_id": f"search-{index + 1}",
                                    "label": str(item.get("title", "Search result")),
                                    "kind": "web_search",
                                    "excerpt": str(item.get("snippet", ""))[:1000],
                                }
                            )

            payload = {
                "task": task.to_dict(),
                "evidence": evidence,
                "instruction": "When no evidence is supplied, state the gap instead of inventing a source.",
            }
            result = self.request_json(payload, ResearchResult.from_dict)
            return AgentResult(
                agent_name=task.assigned_agent,
                task_id=task.task_id,
                status="completed",
                output=result.to_dict(),
                sources=result.sources,
                duration_seconds=time.perf_counter() - started,
                retry_count=retry_count,
                metadata={"rag_used": rag_used, "tools_used": tools_used, "model": self.config.model},
            )
        except (AgentExecutionError, GroqClientError, RagPipelineError, ValueError) as exc:
            return AgentResult(
                agent_name=task.assigned_agent,
                task_id=task.task_id,
                status="failed",
                output=None,
                error=str(exc),
                duration_seconds=time.perf_counter() - started,
                retry_count=retry_count,
                metadata={"rag_used": rag_used, "tools_used": tools_used, "model": self.config.model},
            )
