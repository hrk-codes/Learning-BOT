from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from multi_agent.agents.contracts import ContractValidationError


class AgentExecutionError(RuntimeError):
    """Raised for a bounded specialist failure that the manager can handle."""


LLMCall = Callable[[list[dict[str, str]], int], str]
_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass(frozen=True)
class AgentConfig:
    name: str
    system_prompt: str
    model: str
    temperature: float
    max_tokens: int
    timeout_seconds: int
    max_retries: int
    allowed_tools: tuple[str, ...] = ()
    allow_rag: bool = False
    allow_memory: bool = False


class BaseAgent:
    def __init__(self, config: AgentConfig, llm_call: LLMCall) -> None:
        self.config = config
        self._llm_call = llm_call

    def request_json(self, payload: dict[str, Any], validator: Callable[[object], Any]) -> Any:
        """Ask once, then use one bounded repair request for invalid JSON/schema output."""

        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {
                "role": "user",
                "content": "Return only one JSON object for this contract input:\n"
                + json.dumps(payload, ensure_ascii=False),
            },
        ]
        last_error = ""
        for attempt in range(self.config.max_retries + 1):
            started = time.perf_counter()
            raw = self._llm_call(messages, self.config.timeout_seconds)
            elapsed = time.perf_counter() - started
            try:
                return validator(_parse_json_object(raw))
            except ContractValidationError as exc:
                last_error = str(exc)
                if attempt >= self.config.max_retries:
                    break
                messages = [
                    {"role": "system", "content": self.config.system_prompt},
                    {
                        "role": "user",
                        "content": (
                            "Your last output failed schema validation: "
                            f"{last_error}. Return corrected JSON only. Contract input:\n"
                            + json.dumps(payload, ensure_ascii=False)
                        ),
                    },
                ]
        raise AgentExecutionError(
            f"{self.config.name} returned invalid structured output after "
            f"{self.config.max_retries + 1} attempt(s): {last_error}"
        )


def _parse_json_object(raw: str) -> dict[str, Any]:
    candidate = raw.strip()
    match = _FENCED_JSON.search(candidate)
    if match:
        candidate = match.group(1)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ContractValidationError("response was not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ContractValidationError("response must be one JSON object")
    return parsed
