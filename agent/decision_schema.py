import json
from dataclasses import dataclass
from typing import Literal


AllowedAction = Literal["ANALYZE", "PLAN", "CONTINUE", "FINISH"]
ALLOWED_ACTIONS = {"ANALYZE", "PLAN", "CONTINUE", "FINISH"}


class DecisionParseError(Exception):
    """Raised when the LLM response does not satisfy the decision contract."""


@dataclass(frozen=True)
class AgentDecision:
    action: AllowedAction
    content: str
    finished: bool
    status: str


def parse_agent_decision(raw_text: str) -> AgentDecision:
    data = _parse_json_object(raw_text)

    raw_action = data.get("action")
    action = raw_action.upper() if isinstance(raw_action, str) else raw_action
    content = data.get("content")
    finished = data.get("finished")
    status = data.get("status", "")

    if action not in ALLOWED_ACTIONS:
        raise DecisionParseError(f"Unknown action: {action!r}")
    if not isinstance(content, str) or not content.strip():
        raise DecisionParseError("Decision content must be a non-empty string.")
    if not isinstance(finished, bool):
        raise DecisionParseError("Decision finished must be true or false.")
    if action == "FINISH" and not finished:
        raise DecisionParseError("FINISH decisions must set finished=true.")
    if action != "FINISH" and finished:
        raise DecisionParseError("Only FINISH may set finished=true.")

    return AgentDecision(
        action=action,
        content=content.strip(),
        finished=finished,
        status=str(status).strip() or "decision accepted",
    )


def _parse_json_object(raw_text: str) -> dict:
    """Parse a JSON object, allowing a model to wrap it in explanatory text.

    Production systems usually use native structured outputs or tool calls.
    Stage 3 keeps raw JSON visible so the decision contract is easy to inspect.
    """
    text = raw_text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise DecisionParseError("LLM response did not contain a JSON object.")
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise DecisionParseError(f"LLM response contained malformed JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise DecisionParseError("LLM decision must be a JSON object.")
    return parsed
