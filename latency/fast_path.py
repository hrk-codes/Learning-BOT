from __future__ import annotations

from dataclasses import dataclass

from planner.planning_need import PlanningNeedDetector


@dataclass(frozen=True)
class FastPathDecision:
    eligible: bool
    reason: str


# This is intentionally conservative. A false negative only uses the existing
# richer workflow; a false positive could skip a capability the user expected.
_CAPABILITY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("RAG or document retrieval", ("pdf", "document", "knowledge base", "indexed", "upload")),
    ("current or external information", ("latest", "current", "today", "news", "weather", "price", "stock", "search web")),
    ("tool or side-effect request", ("calculate", "tool", "send", "email", "delete", "create file", "write file")),
    ("long-term memory command", ("remember", "forget", "what do you remember", "my preference", "memory")),
)


def classify_fast_path(goal: str) -> FastPathDecision:
    """Identify questions that can safely use exactly one streamed LLM call."""

    text = goal.strip().lower()
    if not text:
        return FastPathDecision(False, "empty request")

    for reason, patterns in _CAPABILITY_PATTERNS:
        if any(pattern in text for pattern in patterns):
            return FastPathDecision(False, reason)

    planning = PlanningNeedDetector().detect(goal)
    if planning.needs_planning:
        return FastPathDecision(False, "complex planning request")

    return FastPathDecision(True, "simple conversational question")
