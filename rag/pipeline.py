import hashlib
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable
from typing import Any

from rag.embeddings.embedder import EmbeddingProvider
from rag.ingestion.chunker import FixedWindowChunker
from rag.ingestion.loader import save_original_document, validate_pdf_upload
from rag.ingestion.parser import PdfParser
from rag.models import IndexingResult, KnowledgeBaseStats, RetrievalResult
from rag.retrieval.retriever import Retriever
from rag.storage.vector_store import JsonVectorStore


logger = logging.getLogger(__name__)
ProgressCallback = Callable[[str], None]


class RagPipelineError(Exception):
    """Raised when an ingestion or retrieval stage cannot complete safely."""


class RagPipeline:
    def __init__(
        self,
        documents_root: Path,
        vector_store: JsonVectorStore,
        parser: PdfParser,
        chunker: FixedWindowChunker,
        embedder: EmbeddingProvider,
        max_upload_mb: int = 15,
        default_top_k: int = 4,
        default_min_score: float = 0.25,
        max_context_chars: int = 8000,
    ) -> None:
        self.documents_root = documents_root
        self.vector_store = vector_store
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.retriever = Retriever(embedder, vector_store)
        self.max_upload_mb = max_upload_mb
        self.default_top_k = default_top_k
        self.default_min_score = default_min_score
        self.max_context_chars = max_context_chars

    def index_pdf(
        self,
        filename: str,
        content: bytes,
        version: str = "1",
        user_id: str = "local",
        force: bool = False,
        document_id: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> IndexingResult:
        try:
            _notify(progress_callback, "Uploading original PDF")
            safe_name = validate_pdf_upload(filename, content, self.max_upload_mb)
            content_hash = hashlib.sha256(content).hexdigest()
            document_id = document_id or hashlib.sha256(
                f"{safe_name}:{version}:{content_hash}".encode("utf-8")
            ).hexdigest()[:20]
            existing = self.vector_store.get_document(document_id)
            if existing and existing.get("status") == "indexed" and not force:
                _notify(progress_callback, "Existing index is already current")
                return IndexingResult(
                    document_id=document_id,
                    filename=safe_name,
                    page_count=int(existing.get("page_count", 0)),
                    chunk_count=int(existing.get("chunk_count", 0)),
                    embedding_count=int(existing.get("embedding_count", 0)),
                    embedding_seconds=0.0,
                    status="indexed",
                    reused_existing_index=True,
                )

            indexed_at = _utc_now()
            original_path = save_original_document(
                self.documents_root, document_id, safe_name, content
            )
            metadata: dict[str, Any] = {
                "document_id": document_id,
                "filename": safe_name,
                "source": safe_name,
                "source_type": "pdf",
                "version": version.strip() or "1",
                "content_hash": content_hash,
                "indexed_at": indexed_at,
                "created_at": existing.get("created_at", indexed_at) if existing else indexed_at,
                "status": "processing",
                "raw_path": str(original_path),
                "user_id": user_id,
            }
            self.vector_store.set_document_record(metadata)

            _notify(progress_callback, "Parsing text by page")
            pages = self.parser.parse(content)
            _notify(progress_callback, "Creating deterministic chunks and metadata")
            chunks = self.chunker.chunk_pages(pages, metadata)

            _notify(progress_callback, "Embedding document chunks")
            embedding_started = time.perf_counter()
            # Chunks are embedded during ingestion and persisted. Recomputing all
            # document vectors for every question would mix ingestion cost into query time.
            embeddings = self.embedder.embed_documents([chunk.text for chunk in chunks])
            embedding_seconds = time.perf_counter() - embedding_started

            final_metadata = {
                **metadata,
                "status": "indexed",
                "page_count": len(pages),
                "chunk_count": len(chunks),
                "embedding_count": len(embeddings),
                "embedding_model": self.embedder.model_name,
                "embedding_seconds": round(embedding_seconds, 6),
                "chunk_size": self.chunker.chunk_size,
                "chunk_overlap": self.chunker.chunk_overlap,
                "failure_reason": None,
            }
            _notify(progress_callback, "Writing vectors and document metadata")
            self.vector_store.upsert_document(final_metadata, chunks, embeddings)
            _notify(progress_callback, "Completed")
            logger.info(
                "RAG INDEX document_id=%s filename=%s pages=%s chunks=%s embedding_seconds=%.4f",
                document_id,
                safe_name,
                len(pages),
                len(chunks),
                embedding_seconds,
            )
            return IndexingResult(
                document_id=document_id,
                filename=safe_name,
                page_count=len(pages),
                chunk_count=len(chunks),
                embedding_count=len(embeddings),
                embedding_seconds=embedding_seconds,
                status="indexed",
            )
        except Exception as exc:
            if document_id:
                current = self.vector_store.get_document(document_id) or {
                    "document_id": document_id,
                    "filename": Path(filename).name,
                    "source": Path(filename).name,
                    "version": version,
                    "user_id": user_id,
                    "created_at": _utc_now(),
                }
                try:
                    self.vector_store.set_document_record(
                        {
                            **current,
                            "status": "failed",
                            "failure_reason": str(exc),
                            "indexed_at": _utc_now(),
                        }
                    )
                except Exception:
                    logger.exception("RAG failed to persist document failure state")
            logger.exception("RAG INDEX FAILED filename=%s", Path(filename).name)
            if isinstance(exc, RagPipelineError):
                raise
            raise RagPipelineError(str(exc)) from exc

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
        user_id: str = "local",
    ) -> RetrievalResult:
        try:
            return self.retriever.retrieve(
                query=query,
                top_k=top_k or self.default_top_k,
                min_score=self.default_min_score if min_score is None else min_score,
                user_id=user_id,
            )
        except Exception as exc:
            raise RagPipelineError(f"Knowledge retrieval failed: {exc}") from exc

    def reindex_document(self, document_id: str) -> IndexingResult:
        document = self.vector_store.get_document(document_id)
        if document is None:
            raise RagPipelineError(f"Unknown document: {document_id}")
        raw_path = Path(document.get("raw_path", ""))
        if not raw_path.is_file():
            raise RagPipelineError("The original PDF is missing, so it cannot be re-indexed.")
        return self.index_pdf(
            filename=document["filename"],
            content=raw_path.read_bytes(),
            version=str(document.get("version", "1")),
            user_id=str(document.get("user_id", "local")),
            force=True,
            document_id=document_id,
        )

    def delete_document(self, document_id: str) -> bool:
        document = self.vector_store.get_document(document_id)
        deleted = self.vector_store.delete_document(document_id)
        if deleted and document:
            raw_path = Path(document.get("raw_path", ""))
            document_dir = raw_path.parent
            if document_dir.is_dir() and document_dir.parent.resolve() == self.documents_root.resolve():
                shutil.rmtree(document_dir)
        logger.info("RAG DELETE document_id=%s deleted=%s", document_id, deleted)
        return deleted

    def list_documents(self) -> list[dict[str, Any]]:
        return self.vector_store.list_documents()

    def stats(self) -> KnowledgeBaseStats:
        return self.vector_store.stats()

    def describe_for_agent(self) -> dict[str, Any]:
        documents = [
            {
                "document_id": doc.get("document_id"),
                "filename": doc.get("filename"),
                "version": doc.get("version"),
                "status": doc.get("status"),
            }
            for doc in self.list_documents()
            if doc.get("status") == "indexed"
        ]
        return {
            "available": bool(documents),
            "documents": documents,
            "instruction": (
                "Use RETRIEVE_KNOWLEDGE only for questions that require facts from these documents."
            ),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _notify(callback: ProgressCallback | None, stage: str) -> None:
    if callback is not None:
        callback(stage)
