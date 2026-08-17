from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROFILE = "profile"
    PROCEDURAL = "procedural"
    PROJECT = "project"


class MemoryScope(str, Enum):
    USER = "user"
    PROJECT = "project"
    CONVERSATION = "conversation"
    TASK = "task"
    GLOBAL = "global"


class MemorySource(str, Enum):
    USER_EXPLICIT = "user_explicit"
    CONVERSATION = "conversation"
    AGENT_INFERENCE = "agent_inference"
    TOOL_RESULT = "tool_result"
    SYSTEM = "system"
    IMPORTED = "imported"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"
    DELETED = "deleted"
    ARCHIVED = "archived"


@dataclass(frozen=True)
class MemoryCandidate:
    memory_type: MemoryType
    scope: MemoryScope
    content: str
    source: MemorySource
    confidence: float
    importance: float
    key: str = ""
    project_id: str | None = None
    valid_from: str | None = None
    valid_until: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    user_id: str
    project_id: str | None
    memory_type: MemoryType
    scope: MemoryScope
    key: str
    content: str
    normalized_content: str
    source: MemorySource
    confidence: float
    importance: float
    status: MemoryStatus
    created_at: str
    updated_at: str
    valid_from: str
    valid_until: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "type": self.memory_type.value,
            "scope": self.scope.value,
            "project_id": self.project_id,
            "key": self.key,
            "content": self.content,
            "source": self.source.value,
            "confidence": self.confidence,
            "importance": self.importance,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
        }


@dataclass(frozen=True)
class RankedMemory:
    memory: MemoryRecord
    score: float
    lexical_relevance: float
    scope_match: float
    recency: float

    def to_debug_dict(self) -> dict[str, Any]:
        result = self.memory.to_public_dict()
        result.update(
            {
                "retrieval_score": round(self.score, 4),
                "lexical_relevance": round(self.lexical_relevance, 4),
                "scope_match": round(self.scope_match, 4),
                "recency": round(self.recency, 4),
            }
        )
        return result


@dataclass(frozen=True)
class MemoryEvent:
    event_id: int
    memory_id: str | None
    user_id: str
    project_id: str | None
    event_type: str
    created_at: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RememberResult:
    action: str
    memory: MemoryRecord | None
    reason: str = ""
    superseded_ids: tuple[str, ...] = ()
    write_seconds: float = 0.0


@dataclass(frozen=True)
class RetrievalMetrics:
    database_seconds: float = 0.0
    ranking_seconds: float = 0.0
    total_seconds: float = 0.0
    candidate_count: int = 0
    retrieved_count: int = 0


@dataclass(frozen=True)
class RetrievalResult:
    memories: tuple[RankedMemory, ...] = ()
    metrics: RetrievalMetrics = field(default_factory=RetrievalMetrics)


@dataclass(frozen=True)
class MemoryContext:
    payload: dict[str, Any]
    included_ids: tuple[str, ...]
    character_count: int
    approximate_tokens: int


@dataclass(frozen=True)
class ExtractionResult:
    candidates: tuple[MemoryCandidate, ...]
    elapsed_seconds: float
