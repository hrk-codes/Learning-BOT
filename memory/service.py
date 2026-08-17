from __future__ import annotations

import logging
import time
import uuid

from memory.context_builder import MemoryContextBuilder
from memory.extractor import MemoryExtractor
from memory.models import (
    ExtractionResult,
    MemoryCandidate,
    MemoryContext,
    MemoryEvent,
    MemoryRecord,
    MemorySource,
    MemoryStatus,
    MemoryType,
    RememberResult,
    RetrievalMetrics,
    RetrievalResult,
)
from memory.policy import MemoryPolicy, normalize_content, utc_now_iso
from memory.ranker import MemoryRanker
from memory.repository import MemoryRepositoryError, SQLiteMemoryRepository


logger = logging.getLogger(__name__)


class MemoryServiceError(Exception):
    """Raised when long-term memory cannot safely complete an operation."""


class MemoryService:
    def __init__(
        self,
        repository: SQLiteMemoryRepository,
        *,
        enabled: bool,
        retrieval_limit: int,
        context_max_characters: int,
        policy: MemoryPolicy | None = None,
        extractor: MemoryExtractor | None = None,
        ranker: MemoryRanker | None = None,
        context_builder: MemoryContextBuilder | None = None,
    ) -> None:
        self.repository = repository
        self.enabled = enabled
        self.retrieval_limit = max(1, retrieval_limit)
        self.context_max_characters = max(512, context_max_characters)
        self.policy = policy or MemoryPolicy()
        self.extractor = extractor or MemoryExtractor()
        self.ranker = ranker or MemoryRanker()
        self.context_builder = context_builder or MemoryContextBuilder()

    def remember(
        self,
        candidate: MemoryCandidate,
        *,
        user_id: str,
        project_id: str | None,
    ) -> RememberResult:
        started = time.perf_counter()
        if not self.enabled:
            logger.info("MEMORY REJECTED reason=disabled user_id=%s", user_id)
            return RememberResult(action="rejected", memory=None, reason="Long-term memory is OFF.")

        validation = self.policy.validate(candidate, user_id=user_id, project_id=project_id)
        if not validation.accepted or validation.candidate is None:
            logger.info("MEMORY REJECTED reason=policy user_id=%s", user_id)
            return RememberResult(action="rejected", memory=None, reason=validation.reason)
        accepted = validation.candidate
        normalized = normalize_content(accepted.content)

        try:
            duplicate = self.repository.find_duplicate(
                user_id=user_id,
                project_id=accepted.project_id,
                scope=accepted.scope,
                normalized_content=normalized,
            )
            if duplicate:
                logger.info("MEMORY REJECTED reason=duplicate user_id=%s memory_id=%s", user_id, duplicate.memory_id)
                return RememberResult(
                    action="duplicate",
                    memory=duplicate,
                    reason="An equivalent active memory already exists.",
                    write_seconds=time.perf_counter() - started,
                )

            conflicts = self.repository.find_active_by_key(
                user_id=user_id,
                project_id=accepted.project_id,
                scope=accepted.scope,
                key=accepted.key,
            )
            if conflicts and not _can_supersede(accepted, conflicts):
                logger.info("MEMORY REJECTED reason=lower_authority_conflict user_id=%s", user_id)
                return RememberResult(
                    action="rejected",
                    memory=None,
                    reason="A lower-confidence inference cannot override an explicit memory.",
                    write_seconds=time.perf_counter() - started,
                )

            now = utc_now_iso()
            record = MemoryRecord(
                memory_id=str(uuid.uuid4()),
                user_id=user_id,
                project_id=accepted.project_id,
                memory_type=accepted.memory_type,
                scope=accepted.scope,
                key=accepted.key,
                content=accepted.content,
                normalized_content=normalized,
                source=accepted.source,
                confidence=accepted.confidence,
                importance=accepted.importance,
                status=MemoryStatus.ACTIVE,
                created_at=now,
                updated_at=now,
                valid_from=accepted.valid_from or now,
                valid_until=accepted.valid_until,
                metadata=accepted.metadata,
            )

            # Conflict handling and insertion share one repository transaction. If
            # the new write fails, the previous active fact remains authoritative.
            self.repository.store(record, supersede=conflicts)
        except MemoryRepositoryError as exc:
            raise MemoryServiceError(str(exc)) from exc

        action = "superseded" if conflicts else "created"
        elapsed = time.perf_counter() - started
        logger.info(
            "MEMORY ACCEPTED action=%s user_id=%s memory_id=%s type=%s scope=%s write_seconds=%.4f",
            action,
            user_id,
            record.memory_id,
            record.memory_type.value,
            record.scope.value,
            elapsed,
        )
        return RememberResult(
            action=action,
            memory=record,
            superseded_ids=tuple(memory.memory_id for memory in conflicts),
            write_seconds=elapsed,
        )

    def extract_and_remember(
        self,
        text: str,
        *,
        user_id: str,
        project_id: str | None,
    ) -> tuple[ExtractionResult, tuple[RememberResult, ...]]:
        if not self.enabled:
            return ExtractionResult(candidates=(), elapsed_seconds=0.0), ()
        extraction = self.extractor.extract(text, project_id=project_id)
        logger.info(
            "MEMORY CANDIDATES user_id=%s count=%s extraction_seconds=%.4f",
            user_id,
            len(extraction.candidates),
            extraction.elapsed_seconds,
        )
        results = tuple(
            self.remember(candidate, user_id=user_id, project_id=project_id)
            for candidate in extraction.candidates
        )
        return extraction, results

    def search(
        self,
        query: str,
        *,
        user_id: str,
        project_id: str | None,
        limit: int | None = None,
        memory_types: set[MemoryType] | None = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        if not self.enabled:
            return RetrievalResult(
                metrics=RetrievalMetrics(total_seconds=time.perf_counter() - started)
            )
        now = utc_now_iso()
        try:
            self.repository.expire_due(user_id=user_id, expired_at=now)
            database_started = time.perf_counter()
            candidates = self.repository.retrieve_candidates(
                user_id=user_id,
                project_id=project_id,
                memory_types=memory_types,
            )
            database_seconds = time.perf_counter() - database_started
        except MemoryRepositoryError as exc:
            raise MemoryServiceError(str(exc)) from exc

        selected, ranking_seconds = self.ranker.rank(
            query,
            candidates,
            project_id=project_id,
            limit=limit or self.retrieval_limit,
        )
        total = time.perf_counter() - started
        logger.info(
            "MEMORY RETRIEVED user_id=%s candidates=%s selected=%s database_seconds=%.4f ranking_seconds=%.4f total_seconds=%.4f",
            user_id,
            len(candidates),
            len(selected),
            database_seconds,
            ranking_seconds,
            total,
        )
        return RetrievalResult(
            memories=tuple(selected),
            metrics=RetrievalMetrics(
                database_seconds=database_seconds,
                ranking_seconds=ranking_seconds,
                total_seconds=total,
                candidate_count=len(candidates),
                retrieved_count=len(selected),
            ),
        )

    def build_context(self, retrieval: RetrievalResult) -> MemoryContext:
        return self.context_builder.build(
            retrieval,
            max_characters=self.context_max_characters,
        )

    def list_memories(
        self,
        *,
        user_id: str,
        project_id: str | None = None,
        active_only: bool = True,
    ) -> list[MemoryRecord]:
        statuses = {MemoryStatus.ACTIVE} if active_only else None
        try:
            return self.repository.list_memories(
                user_id=user_id,
                project_id=project_id,
                statuses=statuses,
            )
        except MemoryRepositoryError as exc:
            raise MemoryServiceError(str(exc)) from exc

    def get_profile(self, *, user_id: str) -> list[MemoryRecord]:
        return [
            memory
            for memory in self.list_memories(user_id=user_id)
            if memory.memory_type == MemoryType.PROFILE and memory.scope.value == "user"
        ]

    def list_events(self, *, user_id: str, limit: int = 50) -> list[MemoryEvent]:
        try:
            return self.repository.list_events(user_id=user_id, limit=limit)
        except MemoryRepositoryError as exc:
            raise MemoryServiceError(str(exc)) from exc

    def forget_memory(self, *, user_id: str, memory_id: str) -> int:
        try:
            deleted = self.repository.delete_memories(
                user_id=user_id,
                memory_ids=(memory_id,),
                deleted_at=utc_now_iso(),
            )
        except MemoryRepositoryError as exc:
            raise MemoryServiceError(str(exc)) from exc
        logger.info("MEMORY DELETED user_id=%s count=%s", user_id, deleted)
        return deleted

    def forget(self, query: str, *, user_id: str, project_id: str | None) -> list[MemoryRecord]:
        try:
            candidates = self.repository.retrieve_candidates(
                user_id=user_id,
                project_id=project_id,
            )
        except MemoryRepositoryError as exc:
            raise MemoryServiceError(str(exc)) from exc
        ranked, _ = self.ranker.rank(query, candidates, project_id=project_id, limit=5)
        if not ranked:
            return []
        best = ranked[0]
        selected = [item.memory for item in ranked if item.score >= best.score - 0.08]
        try:
            self.repository.delete_memories(
                user_id=user_id,
                memory_ids=(memory.memory_id for memory in selected),
                deleted_at=utc_now_iso(),
            )
        except MemoryRepositoryError as exc:
            raise MemoryServiceError(str(exc)) from exc
        logger.info("MEMORY DELETED user_id=%s count=%s", user_id, len(selected))
        return selected

    def clear_scope(
        self,
        *,
        user_id: str,
        project_id: str | None = None,
    ) -> int:
        memories = self.list_memories(
            user_id=user_id,
            project_id=project_id,
            active_only=False,
        )
        try:
            deleted = self.repository.delete_memories(
                user_id=user_id,
                memory_ids=(memory.memory_id for memory in memories),
                deleted_at=utc_now_iso(),
            )
        except MemoryRepositoryError as exc:
            raise MemoryServiceError(str(exc)) from exc
        logger.info(
            "MEMORY SCOPE CLEARED user_id=%s project_id=%s count=%s",
            user_id,
            project_id,
            deleted,
        )
        return deleted


def _can_supersede(candidate: MemoryCandidate, conflicts: list[MemoryRecord]) -> bool:
    if candidate.source == MemorySource.USER_EXPLICIT:
        return True
    explicit_confidence = max(
        (
            memory.confidence
            for memory in conflicts
            if memory.source == MemorySource.USER_EXPLICIT
        ),
        default=0.0,
    )
    return candidate.confidence >= explicit_confidence
