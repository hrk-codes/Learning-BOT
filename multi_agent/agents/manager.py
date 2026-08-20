from __future__ import annotations

import re
from typing import Any

from multi_agent.agents.base import AgentConfig, BaseAgent
from multi_agent.agents.contracts import (
    AgentName,
    ContractValidationError,
    DelegatedTask,
    ManagerAction,
    ManagerDecision,
    ReviewStatus,
)


class ManagerAgent(BaseAgent):
    """Coordinates specialists through a deliberately small, inspectable action set."""

    def decide(self, state: dict[str, Any]) -> ManagerDecision:
        """Select a next delegation from workflow facts, never free-form model prose.

        The manager is purposefully policy-driven in V1: reliable workflow routing is
        more valuable to teach than giving a general LLM unrestricted orchestration.
        Its model call is reserved for final synthesis, where language generation adds
        value without becoming an execution authority.
        """

        if int(state.get("delegation_count", 0)) >= int(state["max_delegations"]):
            return ManagerDecision(ManagerAction.FINISH, "Delegation limit reached; finish with recorded limitations.")

        research = state.get("research_result")
        draft = state.get("draft_result")
        review = state.get("review_result")
        needs_research = bool(state.get("needs_research"))
        needs_writing = bool(state.get("needs_writing"))
        needs_review = bool(state.get("needs_review"))

        if needs_research and _needs_retry_or_work(research, state, AgentName.RESEARCHER):
            return self._research_decision(state, review)

        if needs_writing and _needs_retry_or_work(draft, state, AgentName.WRITER):
            return self._writer_decision(state, review)

        if needs_review and draft and draft.get("status") == "completed":
            if review is None or _needs_retry_or_work(review, state, AgentName.REVIEWER):
                return self._review_decision(state)
            review_status = (review.get("output") or {}).get("status")
            if review_status in {ReviewStatus.REVISION_REQUIRED.value, ReviewStatus.RESEARCH_REQUIRED.value}:
                if int(state.get("revision_count", 0)) < int(state["max_review_revisions"]):
                    if review_status == ReviewStatus.RESEARCH_REQUIRED.value:
                        return self._research_decision(state, review, force=True)
                    return self._writer_decision(state, review, force=True)

        return ManagerDecision(ManagerAction.FINISH, "Available specialist work is sufficient or bounded limits were reached.")

    def synthesize(self, state: dict[str, Any]) -> str:
        artifacts = {
            "goal": state["goal"],
            "research": _completed_output(state.get("research_result")),
            "draft": _completed_output(state.get("draft_result")),
            "review": _completed_output(state.get("review_result")),
            "limitations": _limitations(state),
        }

        def validate(value: object) -> str:
            if not isinstance(value, dict):
                raise ContractValidationError("final response must be a JSON object")
            answer = value.get("final_answer")
            if not isinstance(answer, str) or not answer.strip():
                raise ContractValidationError("final_answer must be a non-empty string")
            return answer.strip()

        try:
            return self.request_json(artifacts, validate)
        except Exception as exc:
            # A coordination runtime should still return useful verified artifacts
            # when synthesis fails, rather than discarding completed specialist work.
            draft = artifacts["draft"] or {}
            text = draft.get("draft") if isinstance(draft, dict) else None
            if isinstance(text, str) and text.strip():
                return text.strip() + "\n\nNote: final manager synthesis was unavailable."
            research = artifacts["research"] or {}
            claims = research.get("claims", []) if isinstance(research, dict) else []
            if claims:
                return "Research findings:\n" + "\n".join(
                    f"- {claim.get('claim', '')}" for claim in claims if isinstance(claim, dict)
                )
            return f"The multi-agent workflow ended with a synthesis error: {exc}"

    def _research_decision(
        self, state: dict[str, Any], review: dict[str, Any] | None, *, force: bool = False
    ) -> ManagerDecision:
        suffix = " Review feedback requires additional evidence." if review else ""
        constraints = ["Return evidence, sources, confidence, and explicit gaps."]
        if review:
            constraints.append(_review_feedback(review))
        return ManagerDecision(
            ManagerAction.DELEGATE_RESEARCH,
            "The goal requires evidence before a final response can be trusted." + suffix,
            DelegatedTask(
                task_id=f"research-{int(state.get('delegation_count', 0)) + 1}",
                assigned_agent=AgentName.RESEARCHER,
                goal=state["goal"],
                expected_output="Structured evidence with source provenance.",
                constraints=tuple(constraints),
                allowed_tools=tuple(state.get("researcher_tools", [])),
                use_rag=bool(state.get("knowledge_base", {}).get("available")),
            ),
        )

    def _writer_decision(
        self, state: dict[str, Any], review: dict[str, Any] | None, *, force: bool = False
    ) -> ManagerDecision:
        constraints = ["Use only supplied evidence for factual claims."]
        if review:
            constraints.append(_review_feedback(review))
        return ManagerDecision(
            ManagerAction.REVISE if force else ManagerAction.DELEGATE_WRITING,
            "A draft is needed from the approved structured evidence.",
            DelegatedTask(
                task_id=f"writing-{int(state.get('delegation_count', 0)) + 1}",
                assigned_agent=AgentName.WRITER,
                goal=state["goal"],
                expected_output="A user-facing draft grounded in supplied evidence.",
                constraints=tuple(constraints),
            ),
        )

    def _review_decision(self, state: dict[str, Any]) -> ManagerDecision:
        return ManagerDecision(
            ManagerAction.DELEGATE_REVIEW,
            "The user requested verification, so an independent reviewer must evaluate the draft.",
            DelegatedTask(
                task_id=f"review-{int(state.get('delegation_count', 0)) + 1}",
                assigned_agent=AgentName.REVIEWER,
                goal=state["goal"],
                expected_output="A pass/revise/research decision with structured feedback.",
                constraints=("Do not rewrite the draft.",),
            ),
        )


def classify_goal(goal: str, knowledge_base: dict[str, Any]) -> dict[str, bool]:
    text = goal.lower()
    research_words = ("research", "compare", "evidence", "source", "fact check", "from document", "from pdf")
    writing_words = ("write", "report", "draft", "article", "post", "proposal", "email")
    review_words = ("review", "verify", "carefully", "fact-check", "quality check", "audit")
    return {
        "needs_research": _contains_any_phrase(text, research_words),
        "needs_writing": _contains_any_phrase(text, writing_words),
        "needs_review": _contains_any_phrase(text, review_words),
    }


def _contains_any_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    return any(re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text) for phrase in phrases)


def _needs_retry_or_work(result: dict[str, Any] | None, state: dict[str, Any], agent: AgentName) -> bool:
    if result is None:
        return True
    if result.get("status") == "completed":
        return False
    attempts = int((state.get("agent_attempts") or {}).get(agent.value, 0))
    return attempts <= int(state["max_agent_retries"])


def _completed_output(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result and result.get("status") == "completed" and isinstance(result.get("output"), dict):
        return result["output"]
    return None


def _review_feedback(review: dict[str, Any]) -> str:
    output = review.get("output") or {}
    feedback = output.get("feedback") if isinstance(output, dict) else ""
    issues = output.get("issues") if isinstance(output, dict) else []
    return "Review feedback: " + (feedback or "; ".join(issues) or "Revise for accuracy.")


def _limitations(state: dict[str, Any]) -> list[str]:
    limitations: list[str] = []
    for result in (state.get("research_result"), state.get("draft_result"), state.get("review_result")):
        if result and result.get("status") != "completed" and result.get("error"):
            limitations.append(str(result["error"]))
    if int(state.get("delegation_count", 0)) >= int(state["max_delegations"]):
        limitations.append("The bounded delegation limit was reached.")
    return limitations
