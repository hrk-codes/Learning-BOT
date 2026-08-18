import logging
import re
from pathlib import Path

import streamlit as st

from agent.agent import run_agent
from agent.planned_agent import run_planned_agent
from config import AppConfig, get_config
from llm.groq_client import GroqClientError
from memory.chat_memory import ChatMemory
from memory.models import MemoryCandidate, MemoryScope, MemorySource, MemoryType
from memory.repository import MemoryRepositoryError, SQLiteMemoryRepository
from memory.service import MemoryService, MemoryServiceError
from planner.models import PlanState
from planner.planner import PlannerError
from planner.planning_need import PlanningDecision, PlanningNeedDetector
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
    st.set_page_config(page_title="Stage 7 Planner Agent", page_icon="AI", layout="centered")
    st.title("Stage 7 Planner and Executor Agent")

    chat_memory = ChatMemory(
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

    if "messages" not in st.session_state:
        load_result = chat_memory.load_history()
        st.session_state.messages = load_result.messages
        if load_result.warning:
            st.warning(load_result.warning)
    if "enabled_tools" not in st.session_state:
        st.session_state.enabled_tools = {tool.name for tool in tool_registry.list_tools()}
    if "long_term_memory_enabled" not in st.session_state:
        st.session_state.long_term_memory_enabled = config.long_term_memory_enabled
    if "planner_enabled" not in st.session_state:
        st.session_state.planner_enabled = config.planner_enabled

    memory_service = _build_memory_service(config)

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

        st.header("Conversation History")
        st.caption(f"Recent context: {config.recent_message_limit} messages")
        if st.button("Clear chat history", type="secondary", use_container_width=True):
            st.session_state.messages = chat_memory.clear_history()
            st.rerun()
        with st.expander("Inspect recent conversation"):
            st.json(st.session_state.messages)

        _render_memory_center(memory_service, config)

        st.header("Agent Runtime")
        st.caption(f"Max iterations: {config.max_agent_iterations}")

        st.header("Planner Runtime")
        st.toggle(
            "Automatic planning",
            key="planner_enabled",
            help=(
                "Complex goals use a validated task graph. Simple questions keep the "
                "lower-latency Stage 6 agent path."
            ),
        )
        st.caption(
            f"Up to {config.planner_max_tasks} tasks, "
            f"{config.planner_max_revisions} revisions, and "
            f"{config.planner_max_execution_steps} execution attempts"
        )

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

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_goal = st.chat_input("Give the agent a goal")
    if not user_goal:
        return

    chat_memory.add_message(st.session_state.messages, "user", user_goal)
    chat_memory.save_history(st.session_state.messages)

    with st.chat_message("user"):
        st.markdown(user_goal)

    direct_answer = _handle_memory_command(user_goal, memory_service, config)
    if direct_answer is not None:
        with st.chat_message("assistant"):
            st.markdown(direct_answer)
        chat_memory.add_message(st.session_state.messages, "assistant", direct_answer)
        chat_memory.save_history(st.session_state.messages)
        return

    extraction_count = 0
    write_debug: list[dict] = []
    retrieval = None
    memory_context = None
    if memory_service is not None:
        try:
            extraction, writes = memory_service.extract_and_remember(
                user_goal,
                user_id=config.long_term_memory_user_id,
                project_id=config.long_term_memory_project_id,
            )
            extraction_count = len(extraction.candidates)
            write_debug = [
                {
                    "action": result.action,
                    "reason": result.reason,
                    "memory_id": result.memory.memory_id if result.memory else None,
                    "type": result.memory.memory_type.value if result.memory else None,
                    "scope": result.memory.scope.value if result.memory else None,
                    "write_seconds": round(result.write_seconds, 4),
                }
                for result in writes
            ]
            retrieval = memory_service.search(
                user_goal,
                user_id=config.long_term_memory_user_id,
                project_id=config.long_term_memory_project_id,
            )
            memory_context = memory_service.build_context(retrieval)
        except MemoryServiceError as exc:
            st.warning(f"Long-term memory is unavailable for this request: {exc}")

    conversation_context = chat_memory.get_recent_history(st.session_state.messages)
    tool_manager = ToolManager(
        registry=tool_registry,
        enabled_tools=set(st.session_state.enabled_tools),
    )
    memory_metrics = _build_memory_metrics(retrieval, memory_context)
    memory_metrics["extraction_candidate_count"] = extraction_count
    planning_decision = PlanningNeedDetector().detect(user_goal)
    use_planner = bool(
        st.session_state.planner_enabled and planning_decision.needs_planning
    )
    agent_state = None
    plan_state = None

    with st.chat_message("assistant"):
        status_placeholder = st.empty()

        try:
            if use_planner:
                status_placeholder.info("Creating and validating an execution plan...")
                plan_state = run_planned_agent(
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
                    long_term_memory_context=(
                        memory_context.payload if memory_context else None
                    ),
                    memory_search_fn=_build_planner_memory_search(
                        memory_service, config
                    ),
                    status_callback=lambda state: _update_plan_status(
                        status_placeholder, state
                    ),
                )
                final_answer = plan_state.final_answer
            else:
                status_placeholder.info(
                    "Using the direct agent path for this simple request..."
                )
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
                    long_term_memory_context=(
                        memory_context.payload if memory_context else None
                    ),
                    memory_metrics=memory_metrics,
                )
                final_answer = agent_state.final_answer
        except (GroqClientError, PlannerError) as exc:
            st.error(str(exc))
            return

        status_placeholder.empty()
        st.markdown(final_answer)

        if plan_state is not None:
            _render_plan_execution(plan_state, planning_decision)
        elif agent_state is not None:
            _render_direct_agent_execution(
                agent_state,
                extraction_count=extraction_count,
                planning_decision=planning_decision,
            )

        if extraction_count or retrieval is not None:
            with st.expander("Memory Debug"):
                st.write(
                    {
                        "memory_enabled": bool(memory_service and memory_service.enabled),
                        "extraction_candidates": extraction_count,
                        "write_results": write_debug,
                        "database_seconds": round(
                            retrieval.metrics.database_seconds if retrieval else 0.0, 4
                        ),
                        "ranking_seconds": round(
                            retrieval.metrics.ranking_seconds if retrieval else 0.0, 4
                        ),
                        "context_characters": (
                            memory_context.character_count if memory_context else 0
                        ),
                        "context_tokens_approx": (
                            memory_context.approximate_tokens if memory_context else 0
                        ),
                        "retrieved": memory_metrics.get("retrieved_memories", []),
                    }
                )

        _render_rag_debug(agent_state=agent_state, plan_state=plan_state)

    chat_memory.add_message(st.session_state.messages, "assistant", final_answer)
    chat_memory.save_history(st.session_state.messages)


def _build_planner_memory_search(memory_service, config):
    if memory_service is None or not memory_service.enabled:
        return None

    def search_memory(query: str) -> dict:
        retrieval = memory_service.search(
            query,
            user_id=config.long_term_memory_user_id,
            project_id=config.long_term_memory_project_id,
        )
        return memory_service.build_context(retrieval).payload

    return search_memory


def _update_plan_status(placeholder, state: PlanState) -> None:
    completed, total = state.progress()
    active = f" Active task: {state.active_task_id}." if state.active_task_id else ""
    placeholder.info(
        f"Plan {state.status.value}: {completed}/{total} required tasks complete."
        f" Revision {state.revision}.{active}"
    )


def _render_plan_execution(
    plan_state: PlanState, planning_decision: PlanningDecision
) -> None:
    with st.expander("Execution Plan", expanded=True):
        completed, total = plan_state.progress()
        st.progress(completed / total if total else 0.0)
        st.write(
            {
                "mode": "planned",
                "status": plan_state.status.value,
                "planning_score": planning_decision.score,
                "planning_reasons": list(planning_decision.reasons),
                "revision": plan_state.revision,
                "execution_steps": plan_state.execution_steps,
                "progress": f"{completed}/{total}",
                "output_keys": sorted(plan_state.outputs),
                **plan_state.public_summary()["metrics"],
            }
        )
        st.dataframe(
            [task.public_summary() for task in plan_state.tasks],
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("Plan lifecycle events"):
            st.json(
                [
                    {
                        "event": event.event_type,
                        "task_id": event.task_id,
                        "message": event.message,
                        "metadata": event.metadata,
                        "timestamp": event.timestamp,
                    }
                    for event in plan_state.events
                ]
            )


def _render_direct_agent_execution(
    agent_state,
    *,
    extraction_count: int,
    planning_decision: PlanningDecision,
) -> None:
    with st.expander("Agent Execution"):
        st.write(
            {
                "mode": "direct",
                "planning_score": planning_decision.score,
                "planning_reasons": list(planning_decision.reasons),
                "goal": agent_state.goal,
                "status": agent_state.status,
                "iterations": agent_state.iteration_count,
                "max_iterations": agent_state.max_iterations,
                "llm_calls": agent_state.llm_call_count,
                "tool_calls": agent_state.tool_call_count,
                "tool_latency_seconds": round(agent_state.total_tool_latency_seconds, 4),
                "rag_retrievals": agent_state.rag_retrieval_count,
                "rag_latency_seconds": round(agent_state.total_rag_latency_seconds, 4),
                "memory_candidates": extraction_count,
                "memory_retrieved": agent_state.memory_retrieved_count,
                "memory_injected": agent_state.memory_injected_count,
                "memory_retrieval_seconds": round(agent_state.memory_retrieval_seconds, 4),
                "memory_context_tokens_approx": agent_state.memory_context_tokens,
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


def _render_rag_debug(*, agent_state, plan_state: PlanState | None) -> None:
    if plan_state is not None:
        retrieval_tasks = [
            task
            for task in plan_state.tasks
            if task.result is not None and task.result.sources
        ]
        if retrieval_tasks:
            with st.expander("RAG Debug"):
                for task in retrieval_tasks:
                    st.write(
                        {
                            "task_id": task.task_id,
                            "query": task.result.metadata.get("query"),
                            "latency_seconds": round(task.result.duration_seconds, 4),
                            "retrieved": list(task.result.sources),
                        }
                    )
        return

    if agent_state is None:
        return
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


def _build_memory_service(config: AppConfig) -> MemoryService | None:
    try:
        repository = SQLiteMemoryRepository(config.long_term_memory_db_path)
        return MemoryService(
            repository,
            enabled=bool(st.session_state.long_term_memory_enabled),
            retrieval_limit=config.long_term_memory_retrieval_limit,
            context_max_characters=config.long_term_memory_context_max_chars,
        )
    except MemoryRepositoryError as exc:
        st.error(f"Long-term memory database is unavailable: {exc}")
        return None


def _render_memory_center(memory_service: MemoryService | None, config: AppConfig) -> None:
    st.header("Memory Center")
    st.toggle(
        "Long-term memory",
        key="long_term_memory_enabled",
        help="OFF prevents both long-term reads and new writes. Chat history still works.",
    )
    st.caption(f"User: {config.long_term_memory_user_id}")
    st.caption(f"Project: {config.long_term_memory_project_id}")
    st.caption("SQLite structured retrieval; semantic memory is a later isolated upgrade.")
    if memory_service is None:
        return

    with st.expander("Add an explicit memory"):
        with st.form("manual-memory-form", clear_on_submit=True):
            content = st.text_area("Memory fact", max_chars=1000)
            memory_type_value = st.selectbox(
                "Type",
                options=[memory_type.value for memory_type in MemoryType],
                index=2,
            )
            scope_value = st.selectbox(
                "Scope",
                options=[MemoryScope.USER.value, MemoryScope.PROJECT.value],
            )
            importance = st.slider("Importance", 0.0, 1.0, 0.8, 0.05)
            key = st.text_input(
                "Stable key (optional)",
                help="Use the same key for facts that should replace one another.",
            )
            submitted = st.form_submit_button(
                "Remember",
                type="primary",
                use_container_width=True,
                disabled=not memory_service.enabled,
            )
        if submitted:
            candidate = MemoryCandidate(
                memory_type=MemoryType(memory_type_value),
                scope=MemoryScope(scope_value),
                project_id=(
                    config.long_term_memory_project_id
                    if scope_value == MemoryScope.PROJECT.value
                    else None
                ),
                key=key,
                content=content,
                source=MemorySource.USER_EXPLICIT,
                confidence=0.99,
                importance=importance,
            )
            try:
                result = memory_service.remember(
                    candidate,
                    user_id=config.long_term_memory_user_id,
                    project_id=config.long_term_memory_project_id,
                )
                if result.action in {"created", "superseded"}:
                    st.success(f"Memory {result.action}.")
                elif result.action == "duplicate":
                    st.info(result.reason)
                else:
                    st.error(result.reason)
            except MemoryServiceError as exc:
                st.error(str(exc))

    try:
        active_memories = memory_service.list_memories(
            user_id=config.long_term_memory_user_id,
        )
    except MemoryServiceError as exc:
        st.error(str(exc))
        return

    with st.expander(f"Active memories ({len(active_memories)})"):
        if not active_memories:
            st.caption("No active long-term memories.")
        for memory in active_memories:
            st.markdown(memory.content)
            st.caption(
                f"{memory.memory_type.value} | {memory.scope.value} | "
                f"confidence {memory.confidence:.2f} | importance {memory.importance:.2f}"
            )
            if st.button(
                "Forget",
                key=f"forget-memory-{memory.memory_id}",
                use_container_width=True,
            ):
                try:
                    memory_service.forget_memory(
                        user_id=config.long_term_memory_user_id,
                        memory_id=memory.memory_id,
                    )
                    st.rerun()
                except MemoryServiceError as exc:
                    st.error(str(exc))

    confirm_project = st.checkbox(
        "Confirm project-memory deletion",
        key="confirm-project-memory-deletion",
    )
    if st.button(
        "Clear project memories",
        disabled=not confirm_project,
        use_container_width=True,
    ):
        try:
            memory_service.clear_scope(
                user_id=config.long_term_memory_user_id,
                project_id=config.long_term_memory_project_id,
            )
            st.rerun()
        except MemoryServiceError as exc:
            st.error(str(exc))

    confirm_all = st.checkbox(
        "Confirm all-memory deletion",
        key="confirm-all-memory-deletion",
    )
    if st.button(
        "Delete all memories",
        disabled=not confirm_all,
        type="secondary",
        use_container_width=True,
    ):
        try:
            memory_service.clear_scope(user_id=config.long_term_memory_user_id)
            st.rerun()
        except MemoryServiceError as exc:
            st.error(str(exc))

    with st.expander("Memory audit trail"):
        try:
            events = memory_service.list_events(user_id=config.long_term_memory_user_id)
            st.json(
                [
                    {
                        "event_id": event.event_id,
                        "memory_id": event.memory_id,
                        "event": event.event_type,
                        "project_id": event.project_id,
                        "created_at": event.created_at,
                        "details": event.details,
                    }
                    for event in events
                ]
            )
        except MemoryServiceError as exc:
            st.error(str(exc))


def _handle_memory_command(
    user_goal: str,
    memory_service: MemoryService | None,
    config: AppConfig,
) -> str | None:
    normalized = " ".join(user_goal.lower().split())
    is_inspect = any(
        phrase in normalized
        for phrase in ("what do you remember", "show my memories", "list my memories")
    )
    is_remember = normalized.startswith("remember ")
    is_forget = normalized.startswith("forget ") or normalized in {
        "delete all memories",
        "forget everything",
    }
    if not (is_inspect or is_remember or is_forget):
        return None
    if memory_service is None:
        return "Long-term memory is unavailable because the local database could not be opened."
    if not memory_service.enabled:
        return (
            "Long-term memory is OFF. I did not read or write persistent memory. "
            "Conversation history still works independently."
        )

    try:
        if is_inspect:
            memories = memory_service.list_memories(user_id=config.long_term_memory_user_id)
            if not memories:
                return "I don't have any active long-term memories stored for you."
            lines = ["These are the active long-term memories actually stored for you:"]
            lines.extend(
                f"- [{memory.memory_type.value}/{memory.scope.value}] {memory.content}"
                for memory in memories
            )
            return "\n".join(lines)

        if is_remember:
            extraction, results = memory_service.extract_and_remember(
                user_goal,
                user_id=config.long_term_memory_user_id,
                project_id=config.long_term_memory_project_id,
            )
            if not extraction.candidates:
                return "I did not find a durable fact to store. Use the Memory Center for manual entry."
            accepted = [result for result in results if result.action in {"created", "superseded"}]
            duplicates = [result for result in results if result.action == "duplicate"]
            rejected = [result.reason for result in results if result.action == "rejected"]
            if accepted:
                return f"Stored {len(accepted)} long-term memory record(s) through validation and policy checks."
            if duplicates:
                return "That information is already stored as an active memory."
            return f"I did not store that memory: {rejected[0] if rejected else 'policy rejected it.'}"

        if normalized in {"delete all memories", "forget everything"}:
            count = memory_service.clear_scope(user_id=config.long_term_memory_user_id)
            return f"Deleted {count} long-term memory record(s). The audit trail retains event metadata, not the deleted content."
        if "everything about this project" in normalized:
            count = memory_service.clear_scope(
                user_id=config.long_term_memory_user_id,
                project_id=config.long_term_memory_project_id,
            )
            return f"Deleted {count} memory record(s) for project {config.long_term_memory_project_id}."

        forget_query = re.sub(r"^forget(?: that)?\s+", "", user_goal, flags=re.IGNORECASE)
        forgotten = memory_service.forget(
            forget_query,
            user_id=config.long_term_memory_user_id,
            project_id=config.long_term_memory_project_id,
        )
        if not forgotten:
            return "I found no active stored memory matching that request. Nothing was deleted."
        return "Deleted these stored memories:\n" + "\n".join(
            f"- {memory.content}" for memory in forgotten
        )
    except MemoryServiceError as exc:
        return f"The memory operation failed safely: {exc}"


def _build_memory_metrics(retrieval, context) -> dict:
    if retrieval is None or context is None:
        return {}
    injected_ids = set(context.included_ids)
    return {
        "candidate_count": retrieval.metrics.candidate_count,
        "retrieved_count": retrieval.metrics.retrieved_count,
        "injected_count": len(context.included_ids),
        "retrieval_seconds": retrieval.metrics.total_seconds,
        "ranking_seconds": retrieval.metrics.ranking_seconds,
        "context_characters": context.character_count,
        "context_tokens": context.approximate_tokens,
        "retrieved_memories": [
            {**ranked.to_debug_dict(), "injected": ranked.memory.memory_id in injected_ids}
            for ranked in retrieval.memories
        ],
    }


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
