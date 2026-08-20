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
    rate_limit_max_retries: int
    rate_limit_max_wait_seconds: float
    history_path: Path
    recent_message_limit: int
    long_term_memory_enabled: bool
    long_term_memory_db_path: Path
    long_term_memory_user_id: str
    long_term_memory_project_id: str
    long_term_memory_retrieval_limit: int
    long_term_memory_context_max_chars: int
    max_agent_iterations: int
    planner_enabled: bool
    planner_temperature: float
    planner_min_output_tokens: int
    planner_max_tasks: int
    planner_max_revisions: int
    planner_max_execution_steps: int
    planner_max_task_retries: int
    planner_max_repair_attempts: int
    langgraph_enabled: bool
    langgraph_checkpoint_db_path: Path
    multi_agent_enabled: bool
    multi_agent_checkpoint_db_path: Path
    multi_agent_max_delegations: int
    multi_agent_max_agent_retries: int
    multi_agent_max_review_revisions: int
    multi_agent_timeout_seconds: int
    multi_agent_output_repair_attempts: int
    approval_db_path: Path
    approval_timeout_seconds: int
    side_effect_permission_enabled: bool
    rag_documents_path: Path
    rag_vector_store_path: Path
    rag_embedding_model: str
    rag_chunk_size: int
    rag_chunk_overlap: int
    rag_top_k: int
    rag_min_score: float
    rag_context_max_chars: int
    rag_max_upload_mb: int


def get_config() -> AppConfig:
    history_path = Path(os.getenv("CHAT_HISTORY_PATH", "memory/history.json"))
    if not history_path.is_absolute():
        history_path = PROJECT_ROOT / history_path

    rag_documents_path = Path(os.getenv("RAG_DOCUMENTS_PATH", "documents/raw"))
    if not rag_documents_path.is_absolute():
        rag_documents_path = PROJECT_ROOT / rag_documents_path

    rag_vector_store_path = Path(os.getenv("RAG_VECTOR_STORE_PATH", "vector_store/index.json"))
    if not rag_vector_store_path.is_absolute():
        rag_vector_store_path = PROJECT_ROOT / rag_vector_store_path

    long_term_memory_db_path = Path(
        os.getenv("LONG_TERM_MEMORY_DB_PATH", "memory/long_term_memory.db")
    )
    if not long_term_memory_db_path.is_absolute():
        long_term_memory_db_path = PROJECT_ROOT / long_term_memory_db_path

    approval_db_path = Path(os.getenv("APPROVAL_DB_PATH", "approval/approvals.db"))
    if not approval_db_path.is_absolute():
        approval_db_path = PROJECT_ROOT / approval_db_path

    langgraph_checkpoint_db_path = Path(
        os.getenv("LANGGRAPH_CHECKPOINT_DB_PATH", "graph/checkpoints.db")
    )
    if not langgraph_checkpoint_db_path.is_absolute():
        langgraph_checkpoint_db_path = PROJECT_ROOT / langgraph_checkpoint_db_path

    multi_agent_checkpoint_db_path = Path(
        os.getenv("MULTI_AGENT_CHECKPOINT_DB_PATH", "multi_agent/checkpoints.db")
    )
    if not multi_agent_checkpoint_db_path.is_absolute():
        multi_agent_checkpoint_db_path = PROJECT_ROOT / multi_agent_checkpoint_db_path

    return AppConfig(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        groq_api_url="https://api.groq.com/openai/v1/chat/completions",
        default_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        default_temperature=float(os.getenv("GROQ_TEMPERATURE", "0.7")),
        default_max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "1024")),
        request_timeout_seconds=int(os.getenv("GROQ_TIMEOUT_SECONDS", "60")),
        rate_limit_max_retries=int(os.getenv("GROQ_RATE_LIMIT_MAX_RETRIES", "2")),
        rate_limit_max_wait_seconds=float(
            os.getenv("GROQ_RATE_LIMIT_MAX_WAIT_SECONDS", "60")
        ),
        history_path=history_path,
        recent_message_limit=int(os.getenv("RECENT_MESSAGE_LIMIT", "10")),
        long_term_memory_enabled=_read_bool("LONG_TERM_MEMORY_ENABLED", True),
        long_term_memory_db_path=long_term_memory_db_path,
        long_term_memory_user_id=os.getenv("LONG_TERM_MEMORY_USER_ID", "local-user").strip(),
        long_term_memory_project_id=os.getenv(
            "LONG_TERM_MEMORY_PROJECT_ID", "learning-bot"
        ).strip(),
        long_term_memory_retrieval_limit=int(
            os.getenv("LONG_TERM_MEMORY_RETRIEVAL_LIMIT", "8")
        ),
        long_term_memory_context_max_chars=int(
            os.getenv("LONG_TERM_MEMORY_CONTEXT_MAX_CHARS", "4000")
        ),
        max_agent_iterations=int(os.getenv("MAX_AGENT_ITERATIONS", "4")),
        planner_enabled=_read_bool("PLANNER_ENABLED", True),
        planner_temperature=float(os.getenv("PLANNER_TEMPERATURE", "0.1")),
        planner_min_output_tokens=int(os.getenv("PLANNER_MIN_OUTPUT_TOKENS", "1200")),
        planner_max_tasks=int(os.getenv("PLANNER_MAX_TASKS", "8")),
        planner_max_revisions=int(os.getenv("PLANNER_MAX_REVISIONS", "2")),
        planner_max_execution_steps=int(
            os.getenv("PLANNER_MAX_EXECUTION_STEPS", "12")
        ),
        planner_max_task_retries=int(
            os.getenv("PLANNER_MAX_TASK_RETRIES", "1")
        ),
        planner_max_repair_attempts=int(
            os.getenv("PLANNER_MAX_REPAIR_ATTEMPTS", "1")
        ),
        langgraph_enabled=_read_bool("LANGGRAPH_ENABLED", True),
        langgraph_checkpoint_db_path=langgraph_checkpoint_db_path,
        multi_agent_enabled=_read_bool("MULTI_AGENT_ENABLED", True),
        multi_agent_checkpoint_db_path=multi_agent_checkpoint_db_path,
        multi_agent_max_delegations=int(os.getenv("MULTI_AGENT_MAX_DELEGATIONS", "8")),
        multi_agent_max_agent_retries=int(os.getenv("MULTI_AGENT_MAX_AGENT_RETRIES", "1")),
        multi_agent_max_review_revisions=int(
            os.getenv("MULTI_AGENT_MAX_REVIEW_REVISIONS", "1")
        ),
        multi_agent_timeout_seconds=int(os.getenv("MULTI_AGENT_TIMEOUT_SECONDS", "60")),
        multi_agent_output_repair_attempts=int(
            os.getenv("MULTI_AGENT_OUTPUT_REPAIR_ATTEMPTS", "1")
        ),
        approval_db_path=approval_db_path,
        approval_timeout_seconds=int(os.getenv("APPROVAL_TIMEOUT_SECONDS", "300")),
        side_effect_permission_enabled=_read_bool(
            "SIDE_EFFECT_PERMISSION_ENABLED", True
        ),
        rag_documents_path=rag_documents_path,
        rag_vector_store_path=rag_vector_store_path,
        rag_embedding_model=os.getenv(
            "RAG_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        rag_chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "1200")),
        rag_chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "200")),
        rag_top_k=int(os.getenv("RAG_TOP_K", "4")),
        rag_min_score=float(os.getenv("RAG_MIN_SCORE", "0.25")),
        rag_context_max_chars=int(os.getenv("RAG_CONTEXT_MAX_CHARS", "8000")),
        rag_max_upload_mb=int(os.getenv("RAG_MAX_UPLOAD_MB", "15")),
    )


def _read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
