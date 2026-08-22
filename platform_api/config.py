from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PlatformSettings:
    environment: str
    database_url: str
    jwt_secret: str
    access_token_minutes: int
    allowed_origins: tuple[str, ...]
    auto_create_schema: bool
    embedded_worker: bool
    worker_poll_seconds: float
    request_rate_limit: int
    run_rate_limit: int
    max_run_seconds: int
    max_llm_calls: int
    max_run_tokens: int
    object_store_path: Path
    platform_memory_path: Path
    platform_vector_root: Path
    log_level: str

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    def validate(self) -> None:
        if self.environment not in {"development", "staging", "production", "test"}:
            raise ValueError("PLATFORM_ENV must be development, staging, production, or test.")
        if self.is_production and self.jwt_secret == "development-only-change-me-at-least-32-bytes":
            raise ValueError("PLATFORM_JWT_SECRET must be replaced in production.")
        if self.is_production and self.auto_create_schema:
            raise ValueError("Production must use Alembic migrations, not AUTO_CREATE_SCHEMA.")
        if self.is_production and not self.database_url.startswith("postgresql+"):
            raise ValueError("Production requires a PostgreSQL DATABASE_URL.")


@lru_cache(maxsize=1)
def get_platform_settings() -> PlatformSettings:
    environment = os.getenv("PLATFORM_ENV", "development").strip().lower()
    default_db = "sqlite:///platform_data/platform.db"
    secret = os.getenv(
        "PLATFORM_JWT_SECRET", "development-only-change-me-at-least-32-bytes"
    )
    if environment == "test" and not secret:
        secret = secrets.token_hex(32)
    settings = PlatformSettings(
        environment=environment,
        database_url=os.getenv("DATABASE_URL", default_db).strip(),
        jwt_secret=secret,
        access_token_minutes=int(os.getenv("ACCESS_TOKEN_MINUTES", "60")),
        allowed_origins=tuple(
            origin.strip()
            for origin in os.getenv(
                "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
            ).split(",")
            if origin.strip()
        ),
        auto_create_schema=_bool("AUTO_CREATE_SCHEMA", environment != "production"),
        embedded_worker=_bool("EMBEDDED_WORKER", environment == "development"),
        worker_poll_seconds=float(os.getenv("WORKER_POLL_SECONDS", "0.5")),
        request_rate_limit=int(os.getenv("REQUEST_RATE_LIMIT_PER_MINUTE", "120")),
        run_rate_limit=int(os.getenv("RUN_RATE_LIMIT_PER_MINUTE", "10")),
        max_run_seconds=int(os.getenv("MAX_RUN_SECONDS", "300")),
        max_llm_calls=int(os.getenv("MAX_RUN_LLM_CALLS", "16")),
        max_run_tokens=int(os.getenv("MAX_RUN_TOKENS", "16000")),
        object_store_path=_path("OBJECT_STORE_PATH", "platform_data/objects"),
        platform_memory_path=_path("PLATFORM_MEMORY_PATH", "platform_data/memory.db"),
        platform_vector_root=_path("PLATFORM_VECTOR_ROOT", "platform_data/vectors"),
        log_level=os.getenv("PLATFORM_LOG_LEVEL", "INFO").upper(),
    )
    settings.validate()
    return settings


def _path(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default))
    return value if value.is_absolute() else PROJECT_ROOT / value


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
