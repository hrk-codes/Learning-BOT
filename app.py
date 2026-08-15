import logging
from pathlib import Path

import streamlit as st

from agent.agent import run_agent
from config import get_config
from llm.groq_client import GroqClientError
from memory.chat_memory import ChatMemory
from rag.embeddings.embedder import SentenceTransformerEmbedder
from rag.ingestion.chunker import FixedWindowChunker
from rag.ingestion.parser import PdfParser
from rag.pipeline import RagPipeline, RagPipelineError
from rag.storage.vector_store import JsonVectorStore, VectorStoreError
from tools.factory import build_default_registry
from tools.manager import ToolManager


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)


@st.cache_resource(show_spinner=False)
def build_rag_pipeline(
    documents_path: str,
    vector_store_path: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    min_score: float,
    max_context_chars: int,
    max_upload_mb: int,
) -> RagPipeline:
    embedder = SentenceTransformerEmbedder(embedding_model)
    vector_store = JsonVectorStore(Path(vector_store_path), embedding_model)
    return RagPipeline(
        documents_root=Path(documents_path),
        vector_store=vector_store,
        parser=PdfParser(),
        chunker=FixedWindowChunker(chunk_size, chunk_overlap),
        embedder=embedder,
        max_upload_mb=max_upload_mb,
        default_top_k=top_k,
        default_min_score=min_score,
        max_context_chars=max_context_chars,
    )

def main() -> None:
    config = get_config()
    memory = ChatMemory(
        history_path=config.history_path,
        recent_message_limit=config.recent_message_limit,
    )
    tool_registry = build_default_registry()
    rag_pipeline = build_rag_pipeline(
        str(config.rag_documents_path),
        str(config.rag_vector_store_path),
        config.rag_embedding_model,
        config.rag_chunk_size,
        config.rag_chunk_overlap,
        config.rag_top_k,
        config.rag_min_score,
        config.rag_context_max_chars,
        config.rag_max_upload_mb,
    )

    st.set_page_config(page_title="Stage 5 RAG Agent", page_icon="AI", layout="centered")
    st.title("Stage 5 RAG Agent")

    if "messages" not in st.session_state:
        load_result = memory.load_history()
        st.session_state.messages = load_result.messages
        if load_result.warning:
            st.warning(load_result.warning)
    if "enabled_tools" not in st.session_state:
        st.session_state.enabled_tools = {tool.name for tool in tool_registry.list_tools()}

    with st.sidebar:
        st.header("Model Settings")
        model = st.text_input("Model", value=config.default_model)
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=config.default_temperature,
            step=0.1,
        )
        max_tokens = st.slider(
            "Max tokens",
            min_value=64,
            max_value=4096,
            value=config.default_max_tokens,
            step=64,
        )

        st.divider()
        if config.groq_api_key:
            st.success("GROQ_API_KEY loaded")
        else:
            st.error("GROQ_API_KEY is missing")

        st.header("Memory")
        st.caption(f"Persistent store: {config.history_path}")
        st.caption(f"Recent context limit: {config.recent_message_limit} messages")

        st.header("Agent Runtime")
        st.caption(f"Max iterations: {config.max_agent_iterations}")

        st.header("RAG Retrieval")
        rag_top_k = st.slider(
            "Top-k chunks",
            min_value=1,
            max_value=10,
            value=config.rag_top_k,
            help="More chunks improve recall but add noise and context tokens.",
        )
        rag_min_score = st.slider(
            "Minimum similarity",
            min_value=0.0,
            max_value=1.0,
            value=config.rag_min_score,
            step=0.05,
            help="Chunks below this cosine-similarity score are not sent to the agent.",
        )
        st.caption(f"Embedding model: {config.rag_embedding_model}")
        st.caption(
            f"Chunk window: {config.rag_chunk_size} characters "
            f"with {config.rag_chunk_overlap} overlap"
        )

        _render_knowledge_base(rag_pipeline)

        st.header("Agent Toolbox")
        for tool in tool_registry.list_tools():
            enabled = st.checkbox(
                tool.name,
                value=tool.name in st.session_state.enabled_tools,
                help=f"{tool.description} Permission: {tool.permission}",
            )
            if enabled:
                st.session_state.enabled_tools.add(tool.name)
            else:
                st.session_state.enabled_tools.discard(tool.name)

        with st.expander("Available tool contracts"):
            st.json([tool.to_model_description() for tool in tool_registry.list_tools()])

        if st.button("Clear memory", type="secondary"):
            st.session_state.messages = memory.clear_history()
            st.rerun()

        with st.expander("Inspect memory"):
            st.json(st.session_state.messages)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_goal = st.chat_input("Give the agent a goal")
    if not user_goal:
        return

    memory.add_message(st.session_state.messages, "user", user_goal)
    memory.save_history(st.session_state.messages)

    with st.chat_message("user"):
        st.markdown(user_goal)

    conversation_context = memory.get_recent_history(st.session_state.messages)
    tool_manager = ToolManager(
        registry=tool_registry,
        enabled_tools=set(st.session_state.enabled_tools),
    )

    with st.chat_message("assistant"):
        status_placeholder = st.empty()

        try:
            status_placeholder.info("Agent is observing the goal and deciding what to do next...")
            agent_state = run_agent(
                config=config,
                goal=user_goal,
                conversation_context=conversation_context,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tool_manager=tool_manager,
                rag_pipeline=rag_pipeline,
                rag_top_k=rag_top_k,
                rag_min_score=rag_min_score,
            )

        except GroqClientError as exc:
            st.error(str(exc))
            return

        status_placeholder.empty()
        st.markdown(agent_state.final_answer)

        with st.expander("Agent Execution"):
            st.write(
                {
                    "goal": agent_state.goal,
                    "status": agent_state.status,
                    "iterations": agent_state.iteration_count,
                    "max_iterations": agent_state.max_iterations,
                    "llm_calls": agent_state.llm_call_count,
                    "tool_calls": agent_state.tool_call_count,
                    "tool_latency_seconds": round(agent_state.total_tool_latency_seconds, 4),
                    "rag_retrievals": agent_state.rag_retrieval_count,
                    "rag_latency_seconds": round(agent_state.total_rag_latency_seconds, 4),
                    "active_tools": sorted(st.session_state.enabled_tools),
                }
            )
            for step in agent_state.trace:
                trace_data = {
                    "iteration": step.iteration,
                    "action": step.action,
                    "status": step.status,
                    "observation": step.observation,
                }
                if step.tool_name:
                    trace_data["tool_name"] = step.tool_name
                    trace_data["tool_arguments"] = step.tool_arguments
                    trace_data["tool_result"] = step.tool_result
                    trace_data["elapsed_seconds"] = round(step.elapsed_seconds, 4)
                if step.retrieval_query:
                    trace_data["retrieval_query"] = step.retrieval_query
                    trace_data["retrieved_chunk_count"] = len(step.retrieved_chunks or [])
                    trace_data["retrieval_latency_seconds"] = round(
                        step.retrieval_latency_seconds, 4
                    )
                st.write(trace_data)

        retrieval_steps = [step for step in agent_state.trace if step.retrieval_query]
        if retrieval_steps:
            with st.expander("RAG Debug"):
                for step in retrieval_steps:
                    st.write(
                        {
                            "query": step.retrieval_query,
                            "latency_seconds": round(step.retrieval_latency_seconds, 4),
                            "retrieved": [
                                {
                                    "chunk_id": chunk.get("chunk_id"),
                                    "score": round(float(chunk.get("score", 0.0)), 4),
                                    "filename": chunk.get("metadata", {}).get("filename"),
                                    "page_number": chunk.get("metadata", {}).get("page_number"),
                                    "document_id": chunk.get("document_id"),
                                }
                                for chunk in (step.retrieved_chunks or [])
                            ],
                        }
                    )

    memory.add_message(st.session_state.messages, "assistant", agent_state.final_answer)
    memory.save_history(st.session_state.messages)


def _render_knowledge_base(rag_pipeline: RagPipeline) -> None:
    st.header("Knowledge Base")
    uploaded_pdf = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Stage 5 extracts text-based PDFs. Scanned documents require future OCR support.",
    )
    version = st.text_input("Document version", value="1")

    if st.button("Index PDF", disabled=uploaded_pdf is None, type="primary"):
        assert uploaded_pdf is not None
        try:
            with st.status("Indexing PDF", expanded=True) as status:
                result = rag_pipeline.index_pdf(
                    filename=uploaded_pdf.name,
                    content=uploaded_pdf.getvalue(),
                    version=version,
                    progress_callback=status.write,
                )
                status.update(label="PDF indexed", state="complete", expanded=False)
            if result.reused_existing_index:
                st.info("This exact document version was already indexed; embeddings were reused.")
            else:
                st.success(
                    f"Indexed {result.page_count} pages into {result.chunk_count} chunks."
                )
            st.rerun()
        except RagPipelineError as exc:
            st.error(str(exc))

    try:
        stats = rag_pipeline.stats()
        st.caption(
            f"{stats.document_count} indexed documents | "
            f"{stats.chunk_count} chunks | {stats.embedding_count} embeddings"
        )
        documents = rag_pipeline.list_documents()
    except VectorStoreError as exc:
        st.error(str(exc))
        return

    if not documents:
        st.caption("No documents indexed yet.")
        return

    for document in documents:
        label = f"{document.get('filename', 'Document')} - {document.get('status', 'unknown')}"
        with st.expander(label):
            st.json(
                {
                    key: value
                    for key, value in document.items()
                    if key != "raw_path"
                }
            )
            reindex_col, delete_col = st.columns(2)
            if reindex_col.button(
                "Re-index",
                key=f"reindex-{document['document_id']}",
                use_container_width=True,
            ):
                try:
                    with st.spinner("Re-indexing document and replacing its vectors..."):
                        rag_pipeline.reindex_document(document["document_id"])
                    st.success("Document re-indexed.")
                    st.rerun()
                except RagPipelineError as exc:
                    st.error(str(exc))
            if delete_col.button(
                "Delete",
                key=f"delete-{document['document_id']}",
                use_container_width=True,
            ):
                rag_pipeline.delete_document(document["document_id"])
                st.success("Document, chunks, embeddings, and stored original deleted.")
                st.rerun()


if __name__ == "__main__":
    main()
