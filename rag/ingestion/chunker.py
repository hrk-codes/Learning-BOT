import hashlib
from typing import Any

from rag.models import DocumentChunk, ParsedPage


class ChunkingError(Exception):
    """Raised when chunk settings cannot produce valid retrieval units."""


class FixedWindowChunker:
    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 200) -> None:
        if chunk_size < 200:
            raise ChunkingError("chunk_size must be at least 200 characters.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ChunkingError("chunk_overlap must be non-negative and smaller than chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pages(
        self,
        pages: list[ParsedPage],
        document_metadata: dict[str, Any],
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        document_id = str(document_metadata["document_id"])

        for page in pages:
            start = 0
            page_chunk_index = 0
            while start < len(page.text):
                hard_end = min(start + self.chunk_size, len(page.text))
                end = self._find_boundary(page.text, start, hard_end)
                chunk_text = page.text[start:end].strip()
                if chunk_text:
                    identity = f"{document_id}:{page.page_number}:{start}:{end}:{chunk_text}"
                    chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
                    metadata = {
                        **document_metadata,
                        "page_number": page.page_number,
                        "chunk_index": len(chunks),
                        "page_chunk_index": page_chunk_index,
                        "char_start": start,
                        "char_end": end,
                    }
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            document_id=document_id,
                            text=chunk_text,
                            metadata=metadata,
                        )
                    )
                    page_chunk_index += 1

                if end >= len(page.text):
                    break
                # Overlap keeps facts that cross a boundary visible in both
                # neighboring chunks, trading a little duplication for recall.
                next_start = max(0, end - self.chunk_overlap)
                start = end if next_start <= start else next_start

        if not chunks:
            raise ChunkingError("The parsed document produced no non-empty chunks.")
        return chunks

    @staticmethod
    def _find_boundary(text: str, start: int, hard_end: int) -> int:
        if hard_end >= len(text):
            return len(text)

        search_start = start + int((hard_end - start) * 0.65)
        candidates = [
            text.rfind("\n\n", search_start, hard_end),
            text.rfind(". ", search_start, hard_end),
            text.rfind("\n", search_start, hard_end),
            text.rfind(" ", search_start, hard_end),
        ]
        boundary = max(candidates)
        return hard_end if boundary <= start else boundary + 1
