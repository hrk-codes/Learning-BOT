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
    max_agent_iterations: int
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

    return AppConfig(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        groq_api_url="https://api.groq.com/openai/v1/chat/completions",
        default_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        default_temperature=float(os.getenv("GROQ_TEMPERATURE", "0.7")),
        default_max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "512")),
        request_timeout_seconds=int(os.getenv("GROQ_TIMEOUT_SECONDS", "60")),
        history_path=history_path,
        recent_message_limit=int(os.getenv("RECENT_MESSAGE_LIMIT", "10")),
        max_agent_iterations=int(os.getenv("MAX_AGENT_ITERATIONS", "4")),
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
