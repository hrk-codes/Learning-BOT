from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from config import get_config
from llm.groq_client import LatencyMetrics
from multi_agent.runtime import MultiAgentRuntime
from platform_api.config import PlatformSettings
from platform_api.models import Document, PlatformMemory, ToolConfig
from rag.embeddings.embedder import SentenceTransformerEmbedder
from rag.ingestion.chunker import FixedWindowChunker
from rag.ingestion.parser import PdfParser
from rag.pipeline import RagPipeline
from rag.storage.vector_store import JsonVectorStore
from tools.factory import build_default_registry
from tools.manager import ToolManager


@dataclass(frozen=True)
class RuntimeResult:
    answer: str
    trace: list[dict[str, Any]]
    metrics: list[LatencyMetrics]


class Stage10RuntimeAdapter:
    """Keeps Stage 10 intelligence behind a platform-owned execution contract."""

    def __init__(self, settings: PlatformSettings) -> None:
        self.settings = settings

    def execute(
        self,
        db: Session,
        *,
        user_id: str,
        workspace_id: str,
        goal: str,
        on_trace_event=None,
    ) -> RuntimeResult:
        app_config = get_config()
        checkpoint = self.settings.object_store_path.parent / "checkpoints" / f"{workspace_id}.db"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        app_config = replace(
            app_config,
            multi_agent_checkpoint_db_path=checkpoint,
            multi_agent_max_delegations=min(
                app_config.multi_agent_max_delegations, self.settings.max_llm_calls
            ),
        )
        registry = build_default_registry()
        configured = db.scalars(
            select(ToolConfig).where(
                ToolConfig.user_id == user_id,
                ToolConfig.workspace_id == workspace_id,
                ToolConfig.enabled.is_(True),
            )
        ).all()
        enabled = {item.tool_name for item in configured}
        manager = ToolManager(
            registry,
            enabled_tools=enabled,
            authorized_permissions={"safe", "read_only_external"},
        )
        metrics: list[LatencyMetrics] = []

        def record_metric(metric: LatencyMetrics) -> None:
            metrics.append(metric)
            usage = metric.provider_usage
            total_tokens = sum(
                int(item.provider_usage.get("prompt_tokens", item.provider_usage.get("input_tokens", 0)))
                + int(item.provider_usage.get("completion_tokens", item.provider_usage.get("output_tokens", 0)))
                for item in metrics
            )
            if len(metrics) > self.settings.max_llm_calls:
                raise RuntimeError("The run exceeded its configured LLM call budget.")
            if total_tokens > self.settings.max_run_tokens:
                raise RuntimeError("The run exceeded its configured token budget.")

        rag = self._rag_pipeline(db, user_id, workspace_id, app_config)
        runtime = MultiAgentRuntime(
            config=app_config,
            model=app_config.groq_fast_model,
            final_model=app_config.groq_final_model,
            temperature=app_config.default_temperature,
            max_tokens=app_config.groq_fast_max_tokens,
            final_max_tokens=app_config.default_max_tokens,
            tool_manager=manager,
            rag_pipeline=rag,
            rag_top_k=app_config.rag_top_k,
            rag_min_score=app_config.rag_min_score,
            latency_callback=record_metric,
        )
        result = runtime.start(
            goal=goal,
            user_id=user_id,
            conversation_context=[],
            memory_context=self._memory_context(db, user_id, workspace_id),
            knowledge_base=self._knowledge_summary(db, user_id, workspace_id),
            on_trace_event=on_trace_event,
        )
        return RuntimeResult(
            answer=result.final_answer,
            trace=list(result.values.get("node_trace", [])),
            metrics=metrics,
        )

    def _memory_context(self, db: Session, user_id: str, workspace_id: str) -> dict[str, Any]:
        records = db.scalars(
            select(PlatformMemory)
            .where(
                PlatformMemory.user_id == user_id,
                PlatformMemory.is_active.is_(True),
                (PlatformMemory.workspace_id == workspace_id) | (PlatformMemory.workspace_id.is_(None)),
            )
            .order_by(PlatformMemory.updated_at.desc())
            .limit(12)
        ).all()
        return {
            "records": [
                {"id": item.id, "type": item.memory_type, "fact": item.content}
                for item in records
            ]
        }

    def _knowledge_summary(self, db: Session, user_id: str, workspace_id: str) -> dict[str, Any]:
        documents = db.scalars(
            select(Document).where(
                Document.user_id == user_id,
                Document.workspace_id == workspace_id,
                Document.status == "indexed",
            )
        ).all()
        return {
            "available": bool(documents),
            "document_count": len(documents),
            "documents": [{"id": item.id, "filename": item.filename} for item in documents],
        }

    def _rag_pipeline(self, db: Session, user_id: str, workspace_id: str, app_config):
        indexed = db.scalar(
            select(Document.id).where(
                Document.user_id == user_id,
                Document.workspace_id == workspace_id,
                Document.status == "indexed",
            ).limit(1)
        )
        if indexed is None:
            return None
        return build_workspace_rag(self.settings, workspace_id, app_config)


def build_workspace_rag(settings: PlatformSettings, workspace_id: str, app_config) -> RagPipeline:
    documents_root = settings.object_store_path / "rag" / workspace_id
    vector_path = settings.platform_vector_root / workspace_id / "index.json"
    return RagPipeline(
        documents_root=documents_root,
        vector_store=JsonVectorStore(vector_path),
        parser=PdfParser(),
        chunker=FixedWindowChunker(app_config.rag_chunk_size, app_config.rag_chunk_overlap),
        embedder=SentenceTransformerEmbedder(app_config.rag_embedding_model),
        max_upload_mb=app_config.rag_max_upload_mb,
        default_top_k=app_config.rag_top_k,
        default_min_score=app_config.rag_min_score,
        max_context_chars=app_config.rag_context_max_chars,
    )
