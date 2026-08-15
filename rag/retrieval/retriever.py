import hashlib
import logging
import re
import time

from rag.embeddings.embedder import EmbeddingProvider
from rag.models import RetrievalResult
from rag.storage.vector_store import JsonVectorStore


logger = logging.getLogger(__name__)


class QueryProcessor:
    def process(self, query: str) -> str:
        # Stage 5 starts with transparent normalization. Conversation-aware
        # rewriting belongs here later instead of being hidden in the retriever.
        normalized = re.sub(r"\s+", " ", query).strip()
        if not normalized:
            raise ValueError("The retrieval query cannot be empty.")
        return normalized


class Retriever:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        vector_store: JsonVectorStore,
        query_processor: QueryProcessor | None = None,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.query_processor = query_processor or QueryProcessor()

    def retrieve(
        self,
        query: str,
        top_k: int,
        min_score: float,
        user_id: str = "local",
    ) -> RetrievalResult:
        processed_query = self.query_processor.process(query)

        embedding_started = time.perf_counter()
        query_embedding = self.embedder.embed_query(processed_query)
        embedding_seconds = time.perf_counter() - embedding_started

        retrieval_started = time.perf_counter()
        chunks = self.vector_store.search(query_embedding, top_k, min_score, user_id)
        retrieval_seconds = time.perf_counter() - retrieval_started

        query_hash = hashlib.sha256(processed_query.encode("utf-8")).hexdigest()[:12]
        logger.info(
            "RAG RETRIEVE query_hash=%s top_k=%s results=%s chunks=%s scores=%s sources=%s",
            query_hash,
            top_k,
            len(chunks),
            [chunk.chunk_id for chunk in chunks],
            [round(chunk.score, 4) for chunk in chunks],
            [chunk.metadata.get("filename") for chunk in chunks],
        )
        logger.debug("RAG QUERY %s", processed_query)
        return RetrievalResult(
            query=processed_query,
            chunks=chunks,
            query_embedding_seconds=embedding_seconds,
            retrieval_seconds=retrieval_seconds,
            top_k=top_k,
            min_score=min_score,
        )
