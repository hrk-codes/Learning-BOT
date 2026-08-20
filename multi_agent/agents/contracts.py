from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ContractValidationError(ValueError):
    """Raised when an agent artifact does not meet its declared contract."""


class AgentName(str, Enum):
    MANAGER = "manager"
    RESEARCHER = "researcher"
    WRITER = "writer"
    REVIEWER = "reviewer"


class ManagerAction(str, Enum):
    DELEGATE_RESEARCH = "delegate_research"
    DELEGATE_WRITING = "delegate_writing"
    DELEGATE_REVIEW = "delegate_review"
    REVISE = "revise"
    FINISH = "finish"


class ReviewStatus(str, Enum):
    APPROVED = "approved"
    REVISION_REQUIRED = "revision_required"
    RESEARCH_REQUIRED = "research_required"


@dataclass(frozen=True)
class DelegatedTask:
    task_id: str
    assigned_agent: AgentName
    goal: str
    expected_output: str
    constraints: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    use_rag: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "assigned_agent": self.assigned_agent.value,
            "goal": self.goal,
            "expected_output": self.expected_output,
            "constraints": list(self.constraints),
            "allowed_tools": list(self.allowed_tools),
            "use_rag": self.use_rag,
        }


@dataclass(frozen=True)
class ManagerDecision:
    action: ManagerAction
    reason: str
    task: DelegatedTask | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "task": self.task.to_dict() if self.task else None,
        }


@dataclass(frozen=True)
class ResearchClaim:
    claim: str
    source_ids: tuple[str, ...]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "source_ids": list(self.source_ids),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ResearchResult:
    claims: tuple[ResearchClaim, ...]
    sources: tuple[dict[str, Any], ...]
    gaps: tuple[str, ...]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "research_result",
            "claims": [claim.to_dict() for claim in self.claims],
            "sources": list(self.sources),
            "gaps": list(self.gaps),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ResearchResult":
        data = _object(value, "research result")
        claims = data.get("claims")
        sources = data.get("sources")
        gaps = data.get("gaps", [])
        confidence = _confidence(data.get("confidence"), "research confidence")
        if not isinstance(claims, list) or not claims:
            raise ContractValidationError("research claims must contain at least one item")
        if not isinstance(sources, list):
            raise ContractValidationError("research sources must be a list")
        if not isinstance(gaps, list) or not all(isinstance(item, str) for item in gaps):
            raise ContractValidationError("research gaps must be a list of strings")
        parsed_claims: list[ResearchClaim] = []
        for index, item in enumerate(claims):
            claim = _object(item, f"claim {index}")
            text = _text(claim.get("claim"), f"claim {index}")
            source_ids = claim.get("source_ids")
            if not isinstance(source_ids, list) or not all(isinstance(value, str) for value in source_ids):
                raise ContractValidationError(f"claim {index} source_ids must be a list of strings")
            parsed_claims.append(
                ResearchClaim(text, tuple(source_ids), _confidence(claim.get("confidence"), f"claim {index} confidence"))
            )
        source_ids: set[str] = set()
        for source in sources:
            if not isinstance(source, dict) or not isinstance(source.get("source_id"), str):
                raise ContractValidationError("research sources must contain source_id strings")
            source_ids.add(source["source_id"])
        for claim in parsed_claims:
            if not claim.source_ids:
                raise ContractValidationError("each research claim must name at least one source_id")
            if not set(claim.source_ids).issubset(source_ids):
                raise ContractValidationError("research claim references an unknown source_id")
        return cls(tuple(parsed_claims), tuple(sources), tuple(gaps), confidence)


@dataclass(frozen=True)
class WritingResult:
    draft: str

    def to_dict(self) -> dict[str, Any]:
        return {"type": "writing_result", "draft": self.draft}

    @classmethod
    def from_dict(cls, value: object) -> "WritingResult":
        data = _object(value, "writing result")
        return cls(draft=_text(data.get("draft"), "draft"))


@dataclass(frozen=True)
class ReviewResult:
    status: ReviewStatus
    issues: tuple[str, ...] = ()
    feedback: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "review_result",
            "status": self.status.value,
            "issues": list(self.issues),
            "feedback": self.feedback,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ReviewResult":
        data = _object(value, "review result")
        try:
            status = ReviewStatus(data.get("status"))
        except ValueError as exc:
            raise ContractValidationError("review status must be approved, revision_required, or research_required") from exc
        issues = data.get("issues", [])
        if not isinstance(issues, list) or not all(isinstance(item, str) for item in issues):
            raise ContractValidationError("review issues must be a list of strings")
        feedback = data.get("feedback", "")
        if not isinstance(feedback, str):
            raise ContractValidationError("review feedback must be a string")
        if status != ReviewStatus.APPROVED and not issues:
            raise ContractValidationError("a non-approved review must identify at least one issue")
        return cls(status=status, issues=tuple(issues), feedback=feedback.strip())


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractValidationError(f"{label} must be a JSON object")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{label} must be a non-empty string")
    return value.strip()


def _confidence(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ContractValidationError(f"{label} must be a number between 0 and 1")
    result = float(value)
    if not 0 <= result <= 1:
        raise ContractValidationError(f"{label} must be a number between 0 and 1")
    return result
