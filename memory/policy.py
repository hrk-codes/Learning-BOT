from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from memory.models import MemoryCandidate, MemoryScope, MemorySource, MemoryType


_SPACE_PATTERN = re.compile(r"\s+")
_KEY_PATTERN = re.compile(r"[^a-z0-9_.-]+")
_SECRET_PATTERNS = (
    re.compile(r"\bgsk_[A-Za-z0-9_-]{12,}\b", re.IGNORECASE),
    re.compile(r"\b(?:api[_ -]?key|password|secret)\s*[:=]\s*\S+", re.IGNORECASE),
)
_INSTRUCTION_PATTERNS = (
    re.compile(r"ignore (?:all |the )?(?:previous|system) instructions", re.IGNORECASE),
    re.compile(r"override (?:the )?(?:system|developer) prompt", re.IGNORECASE),
)


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    candidate: MemoryCandidate | None = None
    reason: str = ""


class MemoryPolicy:
    """Runtime policy boundary between proposed and persistent memory."""

    _confidence_caps = {
        MemorySource.USER_EXPLICIT: 1.0,
        MemorySource.CONVERSATION: 0.9,
        MemorySource.AGENT_INFERENCE: 0.65,
        MemorySource.TOOL_RESULT: 0.8,
        MemorySource.SYSTEM: 0.9,
        MemorySource.IMPORTED: 0.8,
    }

    def validate(
        self,
        candidate: MemoryCandidate,
        *,
        user_id: str,
        project_id: str | None,
    ) -> ValidationResult:
        content = _SPACE_PATTERN.sub(" ", candidate.content).strip()
        if not user_id.strip():
            return ValidationResult(False, reason="A non-empty user scope is required.")
        if not content:
            return ValidationResult(False, reason="Memory content cannot be empty.")
        if len(content) > 1000:
            return ValidationResult(False, reason="Memory content exceeds the 1,000 character limit.")
        if any(pattern.search(content) for pattern in _SECRET_PATTERNS):
            return ValidationResult(False, reason="Secrets and credentials cannot be stored in memory.")
        if any(pattern.search(content) for pattern in _INSTRUCTION_PATTERNS):
            return ValidationResult(False, reason="Stored instructions cannot override runtime policy.")

        # V1 exposes user and project memory only. The wider scope enum keeps the
        # contract extensible without accidentally enabling cross-user global reads.
        if candidate.scope not in {MemoryScope.USER, MemoryScope.PROJECT}:
            return ValidationResult(False, reason=f"Scope {candidate.scope.value!r} is not enabled in V1.")
        if candidate.scope == MemoryScope.PROJECT and not project_id:
            return ValidationResult(False, reason="Project-scoped memory requires a project ID.")
        if candidate.project_id and candidate.project_id != project_id:
            return ValidationResult(False, reason="A memory cannot write into a different project scope.")

        confidence = _clamp(candidate.confidence)
        importance = _clamp(candidate.importance)
        confidence = min(confidence, self._confidence_caps[candidate.source])
        key = normalize_key(candidate.key) or build_memory_key(candidate.memory_type, content)

        if candidate.valid_from and not _is_valid_timestamp(candidate.valid_from):
            return ValidationResult(False, reason="valid_from must be an ISO-8601 timestamp.")
        if candidate.valid_until and not _is_valid_timestamp(candidate.valid_until):
            return ValidationResult(False, reason="valid_until must be an ISO-8601 timestamp.")

        validated = replace(
            candidate,
            content=content,
            confidence=confidence,
            importance=importance,
            key=key,
            project_id=project_id if candidate.scope == MemoryScope.PROJECT else None,
        )
        return ValidationResult(True, candidate=validated)


def normalize_content(content: str) -> str:
    words = re.findall(r"[a-z0-9]+", content.lower())
    return " ".join(words)


def normalize_key(value: str) -> str:
    normalized = _KEY_PATTERN.sub("_", value.strip().lower())
    return normalized.strip("_.-")[:120]


def build_memory_key(memory_type: MemoryType, content: str) -> str:
    tokens = normalize_content(content).split()
    meaningful = [token for token in tokens if token not in {"user", "users", "the", "a", "an", "is"}]
    suffix = "_".join(meaningful[:6]) or "fact"
    return normalize_key(f"{memory_type.value}.{suffix}")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _is_valid_timestamp(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True
