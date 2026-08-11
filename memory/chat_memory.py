import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


logger = logging.getLogger(__name__)

Role = Literal["user", "assistant"]
Message = dict[str, str]


@dataclass
class LoadResult:
    messages: list[Message]
    warning: str | None = None


class ChatMemory:
    def __init__(self, history_path: Path, recent_message_limit: int) -> None:
        self.history_path = history_path
        self.recent_message_limit = recent_message_limit

    def load_history(self) -> LoadResult:
        if not self.history_path.exists():
            logger.info("History file does not exist yet path=%s", self.history_path)
            return LoadResult(messages=[])

        try:
            raw_history = json.loads(self.history_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.error("History file contains invalid JSON path=%s error=%s", self.history_path, exc)
            return LoadResult(
                messages=[],
                warning="history.json is corrupted, so the app started with empty memory.",
            )

        if not isinstance(raw_history, list):
            logger.error("History file is not a list path=%s", self.history_path)
            return LoadResult(
                messages=[],
                warning="history.json has the wrong shape, so the app started with empty memory.",
            )

        messages = [message for message in raw_history if is_valid_message(message)]
        skipped = len(raw_history) - len(messages)
        if skipped:
            logger.warning("Skipped invalid history messages count=%s", skipped)
            return LoadResult(
                messages=messages,
                warning=f"Skipped {skipped} invalid message(s) from history.json.",
            )

        return LoadResult(messages=messages)

    def save_history(self, messages: list[Message]) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.write_text(
            json.dumps(messages, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Saved conversation history message_count=%s", len(messages))

    def add_message(self, messages: list[Message], role: Role, content: str) -> list[Message]:
        clean_content = content.strip()
        if not clean_content:
            return messages

        messages.append({"role": role, "content": clean_content})
        return messages

    def get_recent_history(self, messages: list[Message]) -> list[Message]:
        if self.recent_message_limit <= 0:
            return []
        return messages[-self.recent_message_limit :]

    def build_context(self, system_prompt: str, messages: list[Message]) -> list[Message]:
        return [{"role": "system", "content": system_prompt}, *self.get_recent_history(messages)]

    def clear_history(self) -> list[Message]:
        self.save_history([])
        return []


def is_valid_message(message: object) -> bool:
    if not isinstance(message, dict):
        return False
    if message.get("role") not in {"user", "assistant"}:
        return False
    if not isinstance(message.get("content"), str):
        return False
    return bool(message["content"].strip())
