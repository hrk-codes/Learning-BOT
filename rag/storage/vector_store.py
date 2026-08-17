import json
import math
from pathlib import Path
from typing import Any

from rag.models import DocumentChunk, KnowledgeBaseStats, RetrievedChunk


class VectorStoreError(Exception):
    """Raised when the local retrieval index is invalid or unavailable."""


class JsonVectorStore:
    SCHEMA_VERSION = 1

    def __init__(self, path: Path, embedding_model: str) -> None:
        self.path = path
        self.embedding_model = embedding_model

    def set_document_record(self, metadata: dict[str, Any]) -> None:
        data = self._read()
        data["documents"][metadata["document_id"]] = metadata
        self._write(data)

    def upsert_document(
        self,
        metadata: dict[str, Any],
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        if not chunks or len(chunks) != len(embeddings):
            raise VectorStoreError("Each indexed chunk must have exactly one embedding.")

        dimension = len(embeddings[0])
        if dimension == 0 or any(len(vector) != dimension for vector in embeddings):
            raise VectorStoreError("Embeddings must share one non-zero dimension.")

        data = self._read()
        existing_dimension = data.get("embedding_dimension")
        if existing_dimension not in {None, dimension} and data["chunks"]:
            raise VectorStoreError(
                "The embedding dimension changed. Delete or rebuild the local vector index."
            )

        document_id = metadata["document_id"]
        # Re-indexing replaces every chunk for this document as one atomic
        # update, preventing stale vectors from an older document version.
        data["chunks"] = [
            item for item in data["chunks"] if item.get("document_id") != document_id
        ]
        data["chunks"].extend(
            {
                **chunk.to_dict(),
                "embedding": vector,
            }
            for chunk, vector in zip(chunks, embeddings)
        )
        data["documents"][document_id] = metadata
        data["embedding_dimension"] = dimension
        self._write(data)

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        min_score: float,
        user_id: str,
    ) -> list[RetrievedChunk]:
        if not query_embedding:
            raise VectorStoreError("The query embedding is empty.")
        if top_k < 1:
            raise VectorStoreError("top_k must be at least 1.")

        data = self._read()
        dimension = data.get("embedding_dimension")
        if dimension is not None and len(query_embedding) != dimension:
            raise VectorStoreError(
                "The query embedding is incompatible with the stored document embeddings."
            )

        ranked: list[RetrievedChunk] = []
        for item in data["chunks"]:
            metadata = item.get("metadata", {})
            # Stage 5 is single-user, but this access field prevents the
            # retrieval API from assuming that every document is globally visible.
            if metadata.get("user_id", "local") != user_id:
                continue
            score = cosine_similarity(query_embedding, item["embedding"])
            if score < min_score:
                continue
            ranked.append(
                RetrievedChunk(
                    chunk_id=item["chunk_id"],
                    document_id=item["document_id"],
                    text=item["text"],
                    score=score,
                    metadata=metadata,
                )
            )

        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:top_k]

    def list_documents(self) -> list[dict[str, Any]]:
        documents = list(self._read()["documents"].values())
        return sorted(documents, key=lambda item: item.get("indexed_at", ""), reverse=True)

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        return self._read()["documents"].get(document_id)

    def delete_document(self, document_id: str) -> bool:
        data = self._read()
        existed = document_id in data["documents"]
        data["documents"].pop(document_id, None)
        data["chunks"] = [
            item for item in data["chunks"] if item.get("document_id") != document_id
        ]
        if existed:
            self._write(data)
        return existed

    def stats(self) -> KnowledgeBaseStats:
        data = self._read()
        indexed = [
            doc for doc in data["documents"].values() if doc.get("status") == "indexed"
        ]
        return KnowledgeBaseStats(
            document_count=len(indexed),
            chunk_count=len(data["chunks"]),
            embedding_count=len(data["chunks"]),
        )

    def _empty(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "embedding_model": self.embedding_model,
            "embedding_dimension": None,
            "documents": {},
            "chunks": [],
        }

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise VectorStoreError(f"The local vector index could not be read: {exc}") from exc

        if not isinstance(data, dict) or not isinstance(data.get("documents"), dict):
            raise VectorStoreError("The local vector index has an invalid structure.")
        if not isinstance(data.get("chunks"), list):
            raise VectorStoreError("The local vector index has an invalid chunk list.")
        if data.get("schema_version") != self.SCHEMA_VERSION:
            raise VectorStoreError("The local vector index schema version is unsupported.")
        if data.get("embedding_model") != self.embedding_model and data["chunks"]:
            raise VectorStoreError(
                "The configured embedding model differs from the stored index. Re-index the documents."
            )
        return data

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            temporary.replace(self.path)
        except OSError as exc:
            raise VectorStoreError(f"The local vector index could not be written: {exc}") from exc


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise VectorStoreError("Cannot compare embeddings with different dimensions.")
    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)
