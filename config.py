import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AppConfig:
    groq_api_key: str | None
    groq_api_url: str
    default_model: str
    default_temperature: float
    default_max_tokens: int
    request_timeout_seconds: int
    history_path: Path
    recent_message_limit: int


def get_config() -> AppConfig:
    history_path = Path(os.getenv("CHAT_HISTORY_PATH", "memory/history.json"))
    if not history_path.is_absolute():
        history_path = PROJECT_ROOT / history_path

    return AppConfig(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        groq_api_url="https://api.groq.com/openai/v1/chat/completions",
        default_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        default_temperature=float(os.getenv("GROQ_TEMPERATURE", "0.7")),
        default_max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "512")),
        request_timeout_seconds=int(os.getenv("GROQ_TIMEOUT_SECONDS", "60")),
        history_path=history_path,
        recent_message_limit=int(os.getenv("RECENT_MESSAGE_LIMIT", "10")),
    )
