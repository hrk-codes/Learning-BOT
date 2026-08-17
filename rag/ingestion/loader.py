import re
from pathlib import Path


class DocumentLoadError(Exception):
    """Raised when an uploaded document is unsafe or unreadable."""


def validate_pdf_upload(filename: str, content: bytes, max_upload_mb: int) -> str:
    safe_name = Path(filename).name.strip()
    if not safe_name or safe_name in {".", ".."}:
        raise DocumentLoadError("The uploaded document needs a valid filename.")
    if Path(safe_name).suffix.lower() != ".pdf":
        raise DocumentLoadError("Stage 5 currently accepts PDF files only.")
    if not content:
        raise DocumentLoadError("The uploaded PDF is empty.")
    if len(content) > max_upload_mb * 1024 * 1024:
        raise DocumentLoadError(f"The PDF exceeds the {max_upload_mb} MB upload limit.")
    if b"%PDF-" not in content[:1024]:
        raise DocumentLoadError("The uploaded file does not have a valid PDF header.")

    # Remove filesystem-hostile characters while preserving a recognizable
    # source name for citations and document lifecycle operations.
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "_", safe_name).strip(" .")
    if not cleaned:
        raise DocumentLoadError("The PDF filename contains no usable characters.")
    return cleaned


def save_original_document(root: Path, document_id: str, filename: str, content: bytes) -> Path:
    document_dir = root / document_id
    document_dir.mkdir(parents=True, exist_ok=True)
    target = document_dir / filename
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(target)
    return target
