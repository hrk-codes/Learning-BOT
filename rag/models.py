from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedPage:
    page_number: int
    text: str


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    query_embedding_seconds: float
    retrieval_seconds: float
    top_k: int
    min_score: float

    @property
    def total_seconds(self) -> float:
        return self.query_embedding_seconds + self.retrieval_seconds


@dataclass(frozen=True)
class IndexingResult:
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    embedding_count: int
    embedding_seconds: float
    status: str
    reused_existing_index: bool = False


@dataclass(frozen=True)
class KnowledgeBaseStats:
    document_count: int
    chunk_count: int
    embedding_count: int
