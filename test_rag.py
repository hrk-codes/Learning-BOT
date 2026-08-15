import math
import tempfile
from pathlib import Path

from agent.agent_loop import run_agent_loop
from agent.decision_schema import parse_agent_decision
from rag.context.context_builder import build_knowledge_observation
from rag.embeddings.embedder import EmbeddingProvider
from rag.ingestion.chunker import FixedWindowChunker
from rag.ingestion.parser import PdfParser
from rag.models import ParsedPage
from rag.pipeline import RagPipeline
from rag.storage.vector_store import JsonVectorStore
from scripts.create_sample_pdfs import SAMPLE_DOCUMENTS, create_text_pdf
from tools.factory import build_default_registry
from tools.manager import ToolManager


class TestEmbedder(EmbeddingProvider):
    model_name = "test-semantic-embedding"

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        lower = text.lower()
        concepts = [
            ("leave", "vacation", "unused", "contractor"),
            ("oauth", "authentication", "token", "authorization"),
            ("temperature", "celsius", "device", "sensor", "heat"),
            ("rate", "requests", "minute", "limit"),
        ]
        vector = [float(sum(term in lower for term in group)) for group in concepts]
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else [0.0] * len(concepts)


def build_test_pipeline(root: Path) -> RagPipeline:
    embedder = TestEmbedder()
    return RagPipeline(
        documents_root=root / "documents",
        vector_store=JsonVectorStore(root / "vectors" / "index.json", embedder.model_name),
        parser=PdfParser(),
        chunker=FixedWindowChunker(chunk_size=300, chunk_overlap=50),
        embedder=embedder,
        default_top_k=3,
        default_min_score=0.2,
        max_context_chars=2000,
    )


def index_sample(pipeline: RagPipeline, filename: str) -> None:
    pipeline.index_pdf(filename, create_text_pdf(SAMPLE_DOCUMENTS[filename]))


def test_pdf_ingestion_preserves_pages_and_metadata() -> None:
    content = create_text_pdf(SAMPLE_DOCUMENTS["employee-handbook.pdf"])
    pages = PdfParser().parse(content)
    assert len(pages) == 1
    assert "carry forward up to 10" in pages[0].text

    with tempfile.TemporaryDirectory() as temporary:
        pipeline = build_test_pipeline(Path(temporary))
        result = pipeline.index_pdf("employee-handbook.pdf", content)
        document = pipeline.list_documents()[0]
        assert result.status == "indexed"
        assert result.page_count == 1
        assert result.chunk_count >= 1
        assert result.embedding_count == result.chunk_count
        assert document["content_hash"]
        assert document["embedding_model"] == TestEmbedder.model_name


def test_semantic_retrieval_returns_source_metadata() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        pipeline = build_test_pipeline(Path(temporary))
        index_sample(pipeline, "employee-handbook.pdf")
        index_sample(pipeline, "internal-api-guide.pdf")

        result = pipeline.retrieve("How many unused leave days can go into next year?")
        assert result.chunks
        top = result.chunks[0]
        assert top.metadata["filename"] == "employee-handbook.pdf"
        assert top.metadata["page_number"] == 1
        assert "10 unused vacation days" in top.text

        observation = build_knowledge_observation(result, max_context_chars=2000)
        assert observation["evidence_found"] is True
        assert observation["chunks"][0]["page_number"] == 1


def test_no_answer_and_document_deletion() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        pipeline = build_test_pipeline(Path(temporary))
        index_sample(pipeline, "employee-handbook.pdf")
        document_id = pipeline.list_documents()[0]["document_id"]

        result = pipeline.retrieve("Where can employees park their cars?")
        assert result.chunks == []
        observation = build_knowledge_observation(result, max_context_chars=2000)
        assert observation["evidence_found"] is False
        assert "do not contain enough" in observation["instruction"]

        assert pipeline.delete_document(document_id) is True
        assert pipeline.stats().document_count == 0
        assert pipeline.stats().chunk_count == 0


def test_agent_can_retrieve_then_use_calculator_tool() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        pipeline = build_test_pipeline(Path(temporary))
        index_sample(pipeline, "product-manual.pdf")
        registry = build_default_registry()
        manager = ToolManager(registry, {"calculator.evaluate"})
        decisions = iter(
            [
                '{"action":"RETRIEVE_KNOWLEDGE","status":"manual evidence needed","content":"Retrieve the operating range.","retrieval_query":"product operating temperature range Celsius","finished":false}',
                '{"action":"TOOL_CALL","status":"conversion needed","content":"Convert the maximum temperature.","tool_name":"calculator.evaluate","tool_arguments":{"expression":"40 * 9 / 5 + 32"},"finished":false}',
                '{"action":"FINISH","status":"grounded answer complete","content":"The supported range is 5 to 40 C, and 40 C equals 104 F.","finished":true}',
            ]
        )

        state = run_agent_loop(
            goal="According to the manual, give the range and convert its maximum to Fahrenheit.",
            conversation_context=[],
            max_iterations=4,
            llm_decision_fn=lambda _messages: next(decisions),
            tool_manager=manager,
            rag_pipeline=pipeline,
            rag_top_k=2,
            rag_min_score=0.2,
        )

        assert state.status == "completed"
        assert state.rag_retrieval_count == 1
        assert state.tool_call_count == 1
        assert [step.action for step in state.trace] == [
            "RETRIEVE_KNOWLEDGE",
            "TOOL_CALL",
            "FINISH",
        ]
        assert "Sources:" in state.final_answer
        assert "product-manual.pdf - page 1" in state.final_answer


def test_retrieval_decision_contract() -> None:
    decision = parse_agent_decision(
        '{"action":"RETRIEVE_KNOWLEDGE","status":"need policy","content":"Search the handbook.","retrieval_query":"contractor leave policy","finished":false}'
    )
    assert decision.retrieval_query == "contractor leave policy"


if __name__ == "__main__":
    test_pdf_ingestion_preserves_pages_and_metadata()
    test_semantic_retrieval_returns_source_metadata()
    test_no_answer_and_document_deletion()
    test_agent_can_retrieve_then_use_calculator_tool()
    test_retrieval_decision_contract()
    print("rag tests passed")
