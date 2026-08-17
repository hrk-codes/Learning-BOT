from collections.abc import Sequence
from typing import Protocol


class EmbeddingError(Exception):
    """Raised when text cannot be converted into vectors."""


class EmbeddingProvider(Protocol):
    model_name: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load_model()
        try:
            # Document embeddings are computed once during ingestion. Query
            # time should only pay for one query vector, not the whole corpus.
            vectors = model.encode_document(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return vectors.tolist()
        except Exception as exc:
            raise EmbeddingError(f"Document embedding failed: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        model = self._load_model()
        try:
            vector = model.encode_query(
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return vector.tolist()
        except Exception as exc:
            raise EmbeddingError(f"Query embedding failed: {exc}") from exc

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:
            raise EmbeddingError(
                f"Could not load embedding model {self.model_name!r}: {exc}"
            ) from exc
        return self._model
