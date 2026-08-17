from __future__ import annotations

import math
import re
import time
from datetime import datetime, timezone

from memory.models import MemoryRecord, MemoryScope, RankedMemory


_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "do", "for", "from", "how",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "should", "that",
    "the", "this", "to", "user", "what", "when", "where", "which", "with",
}


class MemoryRanker:
    """Deterministic V1 ranking isolated from storage and agent orchestration."""

    def rank(
        self,
        query: str,
        memories: list[MemoryRecord],
        *,
        project_id: str | None,
        limit: int,
    ) -> tuple[list[RankedMemory], float]:
        started = time.perf_counter()
        query_tokens = _tokens(query)
        ranked: list[RankedMemory] = []

        for memory in memories:
            lexical = _lexical_relevance(query_tokens, _tokens(memory.content))
            # Scope, recency, importance, and confidence rank relevant candidates;
            # they cannot make an unrelated memory relevant on their own.
            if lexical <= 0:
                continue
            scope_match = _scope_match(memory, project_id)
            recency = _recency(memory.updated_at)
            score = (
                0.45 * lexical
                + 0.15 * scope_match
                + 0.15 * memory.importance
                + 0.15 * memory.confidence
                + 0.10 * recency
            )
            ranked.append(
                RankedMemory(
                    memory=memory,
                    score=score,
                    lexical_relevance=lexical,
                    scope_match=scope_match,
                    recency=recency,
                )
            )

        ranked.sort(
            key=lambda item: (
                item.score,
                item.memory.importance,
                item.memory.updated_at,
            ),
            reverse=True,
        )
        elapsed = time.perf_counter() - started
        return ranked[: max(0, limit)], elapsed


def _tokens(text: str) -> set[str]:
    return {
        _stem(token)
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _STOP_WORDS and len(token) > 1
    }


def _stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 3:
            return token[: -len(suffix)]
    return token


def _lexical_relevance(query_tokens: set[str], memory_tokens: set[str]) -> float:
    if not query_tokens or not memory_tokens:
        return 0.0
    overlap = len(query_tokens & memory_tokens)
    if overlap == 0:
        return 0.0
    return min(1.0, overlap / max(1, min(len(query_tokens), len(memory_tokens))))


def _scope_match(memory: MemoryRecord, project_id: str | None) -> float:
    if memory.scope == MemoryScope.PROJECT:
        return 1.0 if project_id and memory.project_id == project_id else 0.0
    if memory.scope == MemoryScope.USER:
        return 0.9
    return 0.5


def _recency(timestamp: str) -> float:
    try:
        changed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        age_days = max(0.0, (datetime.now(timezone.utc) - changed).total_seconds() / 86400)
    except (TypeError, ValueError):
        return 0.0
    return math.exp(-age_days / 365.0)
