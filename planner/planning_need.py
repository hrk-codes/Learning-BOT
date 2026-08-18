import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PlanningDecision:
    needs_planning: bool
    score: int
    reasons: tuple[str, ...]


class PlanningNeedDetector:
    """Cheap deterministic gate that avoids paying planning cost for simple chat."""

    COMPLEX_PATTERNS = {
        "comparison or evaluation": r"\b(compare|evaluate|assess|trade-?offs?)\b",
        "research request": r"\b(research|investigate|deep dive|find evidence)\b",
        "explicit sequence": r"\b(first|then|after that|finally|step by step)\b",
        "multi-source synthesis": r"\b(based on|using my|from the document|according to)\b",
        "deliverable planning": r"\b(roadmap|implementation plan|strategy|recommendation)\b",
    }

    def detect(self, goal: str) -> PlanningDecision:
        normalized = " ".join(goal.lower().split())
        score = 0
        reasons: list[str] = []

        for reason, pattern in self.COMPLEX_PATTERNS.items():
            if re.search(pattern, normalized):
                score += 2
                reasons.append(reason)

        if len(re.findall(r"\b(and|also|plus)\b", normalized)) >= 2:
            score += 2
            reasons.append("multiple requested outcomes")
        if len(normalized.split()) >= 35:
            score += 1
            reasons.append("long multi-part goal")
        if "\n" in goal and len([line for line in goal.splitlines() if line.strip()]) >= 3:
            score += 2
            reasons.append("multi-line requirements")

        return PlanningDecision(
            needs_planning=score >= 3,
            score=score,
            reasons=tuple(reasons),
        )

