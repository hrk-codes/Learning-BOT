import logging
import json
import re
import sqlite3
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from agent.agent import run_agent
from agent.planned_agent import run_planned_agent
from approval.models import ApprovalStatus, parse_timestamp
from approval.policy import ApprovalPolicy
from approval.repository import ApprovalRepositoryError, SQLiteApprovalRepository
from approval.risk_engine import RiskEngine
from approval.service import ApprovalService, ApprovalServiceError
from config import AppConfig, get_config
from graph.runtime import GraphRunResult, LangGraphPlannedAgent
from llm.groq_client import GroqClientError
from memory.chat_memory import ChatMemory
from memory.models import MemoryCandidate, MemoryScope, MemorySource, MemoryType
from memory.repository import MemoryRepositoryError, SQLiteMemoryRepository
from memory.service import MemoryService, MemoryServiceError
from multi_agent.runtime import MultiAgentRunResult, MultiAgentRuntime
from planner.models import PlanState, TaskStatus
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
    st.set_page_config(page_title="Stage 10 Multi-Agent System", page_icon="AI", layout="centered")
    st.title("Stage 10 Multi-Agent System")

    chat_memory = ChatMemory(
        history_path=config.history_path,
        recent_message_limit=config.recent_message_limit,
    )
    tool_registry = build_default_registry()
    approval_service = _build_approval_service(config, tool_registry)
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
    if "side_effect_permission_enabled" not in st.session_state:
        st.session_state.side_effect_permission_enabled = (
            config.side_effect_permission_enabled
        )
    if "langgraph_enabled" not in st.session_state:
        st.session_state.langgraph_enabled = config.langgraph_enabled
    if "multi_agent_enabled" not in st.session_state:
        st.session_state.multi_agent_enabled = config.multi_agent_enabled

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

        st.header("LangGraph Runtime")
        st.toggle(
            "Use LangGraph for complex goals",
            key="langgraph_enabled",
            help=(
                "Uses a stateful graph for planned workflows. Simple requests keep "
                "the direct Stage 6 agent path; the Stage 8 runtime remains available "
                "as a comparison fallback."
            ),
        )
        st.caption("Local SQLite checkpoints preserve graph state across restarts.")

        st.header("Specialized Team")
        st.toggle(
            "Use manager-led multi-agent workflow",
            key="multi_agent_enabled",
            help=(
                "Uses a Stage 10 LangGraph team: manager, researcher, writer, and "
                "reviewer. Simple requests finish through the manager without extra "
                "specialist calls."
            ),
        )
        st.caption(
            f"Up to {config.multi_agent_max_delegations} delegations, "
            f"{config.multi_agent_max_agent_retries} retry per agent, and "
            f"{config.multi_agent_max_review_revisions} review revision."
        )

        st.header("Human Approval")
        st.toggle(
            "Side-effect capability permission",
            key="side_effect_permission_enabled",
            help=(
                "Permission allows this local session to propose side-effecting mock "
                "tools. Every specific consequential action still requires approval."
            ),
        )
        st.caption(
            f"Per-action approval expires after {config.approval_timeout_seconds} seconds."
        )
        st.caption("Stage 8 side-effect tools are simulations; no real email or deletion occurs.")

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
                help=(
                    f"{tool.description} Permission: {tool.permission}. "
                    f"Risk: {tool.risk_level.value}. Side effect: {tool.side_effect.value}."
                ),
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

    conversation_context = chat_memory.get_recent_history(st.session_state.messages)
    tool_manager = _build_tool_manager(tool_registry)
    graph_runtime = _build_graph_runtime(
        config=config,
        conversation_context=conversation_context,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        tool_manager=tool_manager,
        rag_pipeline=rag_pipeline,
        rag_top_k=rag_top_k,
        rag_min_score=rag_min_score,
        long_term_memory_context=None,
        memory_service=memory_service,
        approval_service=approval_service,
    )
    try:
        waiting_graph = graph_runtime.find_waiting_run(
            user_id=config.long_term_memory_user_id
        )
    except (OSError, sqlite3.DatabaseError, ValueError) as exc:
        st.error(f"LangGraph checkpoint state is unavailable: {exc}")
        waiting_graph = None
    if waiting_graph is not None:
        _handle_waiting_graph(
            graph_runtime=graph_runtime,
            graph_run=waiting_graph,
            approval_service=approval_service,
            config=config,
            chat_memory=chat_memory,
        )
        return

    if approval_service is not None:
        try:
            waiting_state = approval_service.find_waiting_workflow(
                user_id=config.long_term_memory_user_id
            )
        except ApprovalServiceError as exc:
            st.error(f"Approval state is unavailable. Consequential actions are blocked: {exc}")
            waiting_state = None
        if waiting_state is not None:
            _handle_waiting_workflow(
                state=waiting_state,
                approval_service=approval_service,
                config=config,
                chat_memory=chat_memory,
                memory_service=memory_service,
                conversation_context=conversation_context,
                tool_manager=tool_manager,
                rag_pipeline=rag_pipeline,
                rag_top_k=rag_top_k,
                rag_min_score=rag_min_score,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return

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
    memory_metrics = _build_memory_metrics(retrieval, memory_context)
    memory_metrics["extraction_candidate_count"] = extraction_count
    planning_decision = PlanningNeedDetector().detect(user_goal)
    use_planner = bool(
        st.session_state.planner_enabled and planning_decision.needs_planning
    )
    agent_state = None
    plan_state = None
    graph_run = None
    multi_agent_run = None

    with st.chat_message("assistant"):
        status_placeholder = st.empty()

        try:
            if st.session_state.multi_agent_enabled:
                status_placeholder.info("Manager is coordinating the Stage 10 specialist team...")
                multi_agent_runtime = _build_multi_agent_runtime(
                    config=config,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tool_manager=tool_manager,
                    rag_pipeline=rag_pipeline,
                    rag_top_k=rag_top_k,
                    rag_min_score=rag_min_score,
                )
                multi_agent_run = multi_agent_runtime.start(
                    goal=user_goal,
                    user_id=config.long_term_memory_user_id,
                    conversation_context=conversation_context,
                    memory_context=(memory_context.payload if memory_context else None),
                    knowledge_base=rag_pipeline.describe_for_agent(),
                )
                final_answer = multi_agent_run.final_answer
            elif use_planner:
                if st.session_state.langgraph_enabled:
                    status_placeholder.info("Running the Stage 9 stateful execution graph...")
                    graph_runtime = _build_graph_runtime(
                        config=config,
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
                        memory_service=memory_service,
                        approval_service=approval_service,
                    )
                    graph_run = graph_runtime.start(
                        goal=user_goal,
                        conversation_context=conversation_context,
                        memory_context=(memory_context.payload if memory_context else None),
                        knowledge_base=rag_pipeline.describe_for_agent(),
                    )
                    plan_state = graph_run.plan_state
                else:
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
                        approval_service=approval_service,
                        approval_user_id=config.long_term_memory_user_id,
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
        except (GroqClientError, PlannerError, ApprovalServiceError, ValueError) as exc:
            st.error(str(exc))
            return

        status_placeholder.empty()
        st.markdown(final_answer)

        if multi_agent_run is not None:
            _render_multi_agent_execution(multi_agent_run)
        elif plan_state is not None:
            _render_plan_execution(plan_state, planning_decision)
            if graph_run is not None:
                _render_graph_execution(graph_run)
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

        if plan_state is not None and plan_state.status.value == "waiting_for_approval":
            if approval_service is None:
                st.error("Approval service is unavailable. The action remains blocked.")
            elif graph_run is not None:
                _render_graph_approval_panel(
                    graph_runtime=graph_runtime,
                    graph_run=graph_run,
                    approval_service=approval_service,
                    config=config,
                )
            else:
                _render_approval_panel(plan_state, approval_service, config)

    if plan_state is not None and plan_state.status.value == "waiting_for_approval":
        return
    chat_memory.add_message(st.session_state.messages, "assistant", final_answer)
    chat_memory.save_history(st.session_state.messages)


def _build_approval_service(config: AppConfig, tool_registry) -> ApprovalService | None:
    try:
        repository = SQLiteApprovalRepository(config.approval_db_path)
        return ApprovalService(
            repository=repository,
            tool_lookup=tool_registry.get_tool,
            risk_engine=RiskEngine(),
            policy=ApprovalPolicy(
                confirmation_timeout_seconds=config.approval_timeout_seconds
            ),
        )
    except ApprovalRepositoryError as exc:
        st.error(
            "Approval storage is unavailable. All consequential actions are blocked: "
            f"{exc}"
        )
        return None


def _build_tool_manager(tool_registry) -> ToolManager:
    permissions = {"safe", "read_only_external"}
    if st.session_state.side_effect_permission_enabled:
        permissions.add("side_effecting")
    return ToolManager(
        registry=tool_registry,
        enabled_tools=set(st.session_state.enabled_tools),
        authorized_permissions=permissions,
    )


def _build_graph_runtime(
    *,
    config: AppConfig,
    conversation_context: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    tool_manager: ToolManager,
    rag_pipeline: RagPipeline,
    rag_top_k: int,
    rag_min_score: float,
    long_term_memory_context: dict | None,
    memory_service: MemoryService | None,
    approval_service: ApprovalService | None,
) -> LangGraphPlannedAgent:
    """Build a fresh graph definition around live services for this Streamlit run.

    The definition is rebuilt safely on a rerun, while graph state is restored from
    SQLite through the stable thread ID. Runtime services themselves never enter the
    checkpoint because connections and callable objects are not durable state.
    """

    return LangGraphPlannedAgent(
        config=config,
        conversation_context=conversation_context,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        tool_manager=tool_manager,
        rag_pipeline=rag_pipeline,
        rag_top_k=rag_top_k,
        rag_min_score=rag_min_score,
        long_term_memory_context=long_term_memory_context,
        memory_search_fn=_build_planner_memory_search(memory_service, config),
        approval_service=approval_service,
        approval_user_id=config.long_term_memory_user_id,
    )


def _build_multi_agent_runtime(
    *,
    config: AppConfig,
    model: str,
    temperature: float,
    max_tokens: int,
    tool_manager: ToolManager,
    rag_pipeline: RagPipeline,
    rag_top_k: int,
    rag_min_score: float,
) -> MultiAgentRuntime:
    """Construct Stage 10 around existing capabilities without broadening access.

    The runtime receives the established RAG pipeline and ToolManager. Role-level
    contracts decide which of those capabilities a specialist can actually use.
    """

    return MultiAgentRuntime(
        config=config,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        tool_manager=tool_manager,
        rag_pipeline=rag_pipeline,
        rag_top_k=rag_top_k,
        rag_min_score=rag_min_score,
    )


def _handle_waiting_graph(
    *,
    graph_runtime: LangGraphPlannedAgent,
    graph_run: GraphRunResult,
    approval_service: ApprovalService | None,
    config: AppConfig,
    chat_memory: ChatMemory,
) -> None:
    """Render or resume the exact interrupted graph thread after human review."""

    if approval_service is None:
        st.error("Approval storage is unavailable. The graph remains safely paused.")
        return
    should_resume = (
        st.session_state.pop("stage9_resume_thread_id", None) == graph_run.thread_id
    )
    waiting_task = next(
        (
            task
            for task in graph_run.plan_state.tasks
            if task.status == TaskStatus.WAITING_FOR_APPROVAL
        ),
        None,
    )
    if waiting_task and waiting_task.approval_id:
        try:
            request = approval_service.get_approval(waiting_task.approval_id)
            should_resume = should_resume or request.status != ApprovalStatus.PENDING
        except ApprovalServiceError as exc:
            st.error(f"Approval state is unavailable. The graph remains paused: {exc}")
            return

    if should_resume:
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            status_placeholder.info("Resuming the saved LangGraph execution...")
            try:
                graph_run = graph_runtime.resume(graph_run.thread_id)
            except (GroqClientError, PlannerError, ApprovalServiceError, ValueError) as exc:
                st.error(f"The graph could not resume safely: {exc}")
                return
            status_placeholder.empty()
            st.markdown(graph_run.plan_state.final_answer)
            _render_plan_execution(
                graph_run.plan_state, PlanningNeedDetector().detect(graph_run.plan_state.goal)
            )
            _render_graph_execution(graph_run)
            if graph_run.interrupted:
                _render_graph_approval_panel(
                    graph_runtime=graph_runtime,
                    graph_run=graph_run,
                    approval_service=approval_service,
                    config=config,
                )
                return
        chat_memory.add_message(
            st.session_state.messages, "assistant", graph_run.plan_state.final_answer
        )
        chat_memory.save_history(st.session_state.messages)
        return

    _render_plan_execution(
        graph_run.plan_state, PlanningNeedDetector().detect(graph_run.plan_state.goal)
    )
    _render_graph_execution(graph_run)
    _render_graph_approval_panel(
        graph_runtime=graph_runtime,
        graph_run=graph_run,
        approval_service=approval_service,
        config=config,
    )


def _render_graph_approval_panel(
    *,
    graph_runtime: LangGraphPlannedAgent,
    graph_run: GraphRunResult,
    approval_service: ApprovalService,
    config: AppConfig,
) -> None:
    def queue_resume() -> None:
        st.session_state.stage9_resume_thread_id = graph_run.thread_id

    _render_approval_panel(
        graph_run.plan_state,
        approval_service,
        config,
        on_decision=queue_resume,
        workflow_persist=lambda state: graph_runtime.update_plan_state(
            graph_run.thread_id, state
        ),
    )


def _render_graph_execution(graph_run: GraphRunResult) -> None:
    """Show workflow metadata without revealing prompts, document text, or reasoning."""

    trace = list(graph_run.values.get("node_trace", []))
    with st.expander("LangGraph Execution", expanded=True):
        st.write(
            {
                "graph_definition": "planner -> router -> execute -> approval/retry -> evaluate -> replan/end",
                "run_id": graph_run.run_id,
                "thread_id": graph_run.thread_id,
                "status": graph_run.plan_state.status.value,
                "next_nodes": list(graph_run.next_nodes),
                "interrupted": graph_run.interrupted,
                "node_transitions": len(trace),
                "graph_retries": graph_run.values.get("graph_retry_count", 0),
                "checkpointing": "SQLite local development saver",
            }
        )
        if trace:
            st.dataframe(trace, use_container_width=True, hide_index=True)


def _render_multi_agent_execution(run: MultiAgentRunResult) -> None:
    """Expose observable workflow facts without exposing prompts or hidden reasoning."""

    values = run.values
    trace = list(values.get("node_trace", []))
    results = list(values.get("agent_results", []))
    with st.expander("Stage 10 Team Execution", expanded=True):
        st.write(
            {
                "run_id": run.run_id,
                "thread_id": run.thread_id,
                "status": values.get("status"),
                "delegations": values.get("delegation_count", 0),
                "review_revisions": values.get("revision_count", 0),
                "agent_attempts": values.get("agent_attempts", {}),
                "specialist_calls": len(results),
            }
        )
        if trace:
            st.caption("Execution trace")
            st.dataframe(trace, use_container_width=True, hide_index=True)
        if results:
            st.caption("Specialist result metadata")
            st.dataframe(
                [
                    {
                        "agent": item.get("agent_name"),
                        "task": item.get("task_id"),
                        "status": item.get("status"),
                        "duration_seconds": item.get("duration_seconds"),
                        "rag_used": item.get("metadata", {}).get("rag_used"),
                        "tools_used": item.get("metadata", {}).get("tools_used"),
                        "retry_count": item.get("retry_count"),
                    }
                    for item in results
                ],
                use_container_width=True,
                hide_index=True,
            )


def _handle_waiting_workflow(
    *,
    state: PlanState,
    approval_service: ApprovalService,
    config: AppConfig,
    chat_memory: ChatMemory,
    memory_service: MemoryService | None,
    conversation_context: list[dict[str, str]],
    tool_manager: ToolManager,
    rag_pipeline: RagPipeline,
    rag_top_k: int,
    rag_min_score: float,
    model: str,
    temperature: float,
    max_tokens: int,
) -> None:
    task = next(
        (
            item
            for item in state.tasks
            if item.status.value == "waiting_for_approval"
        ),
        None,
    )
    if task is None or not task.approval_id:
        st.error("The persisted workflow has no verifiable approval task. It remains blocked.")
        return
    try:
        request = approval_service.get_approval(task.approval_id)
    except ApprovalServiceError as exc:
        st.error(f"Approval cannot be verified. The action remains blocked: {exc}")
        return

    if request.status == ApprovalStatus.PENDING:
        _render_plan_execution(state, PlanningNeedDetector().detect(state.goal))
        _render_approval_panel(state, approval_service, config)
        return

    _resume_waiting_workflow(
        state=state,
        approval_service=approval_service,
        config=config,
        chat_memory=chat_memory,
        memory_service=memory_service,
        conversation_context=conversation_context,
        tool_manager=tool_manager,
        rag_pipeline=rag_pipeline,
        rag_top_k=rag_top_k,
        rag_min_score=rag_min_score,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _resume_waiting_workflow(
    *,
    state: PlanState,
    approval_service: ApprovalService,
    config: AppConfig,
    chat_memory: ChatMemory,
    memory_service: MemoryService | None,
    conversation_context: list[dict[str, str]],
    tool_manager: ToolManager,
    rag_pipeline: RagPipeline,
    rag_top_k: int,
    rag_min_score: float,
    model: str,
    temperature: float,
    max_tokens: int,
) -> None:
    memory_payload = None
    if memory_service is not None and memory_service.enabled:
        try:
            retrieval = memory_service.search(
                state.goal,
                user_id=config.long_term_memory_user_id,
                project_id=config.long_term_memory_project_id,
            )
            memory_payload = memory_service.build_context(retrieval).payload
        except MemoryServiceError as exc:
            st.warning(f"Long-term memory is unavailable while resuming: {exc}")

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        try:
            resumed = run_planned_agent(
                config=config,
                goal=state.goal,
                conversation_context=conversation_context,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tool_manager=tool_manager,
                rag_pipeline=rag_pipeline,
                rag_top_k=rag_top_k,
                rag_min_score=rag_min_score,
                long_term_memory_context=memory_payload,
                memory_search_fn=_build_planner_memory_search(
                    memory_service, config
                ),
                status_callback=lambda current: _update_plan_status(
                    status_placeholder, current
                ),
                approval_service=approval_service,
                approval_user_id=config.long_term_memory_user_id,
                existing_state=state,
            )
        except (GroqClientError, PlannerError, ApprovalServiceError, ValueError) as exc:
            st.error(f"The workflow could not resume safely: {exc}")
            return
        status_placeholder.empty()
        st.markdown(resumed.final_answer)
        _render_plan_execution(resumed, PlanningNeedDetector().detect(resumed.goal))
        if resumed.status.value == "waiting_for_approval":
            _render_approval_panel(resumed, approval_service, config)
            return

    chat_memory.add_message(
        st.session_state.messages, "assistant", resumed.final_answer
    )
    chat_memory.save_history(st.session_state.messages)


def _render_approval_panel(
    state: PlanState,
    approval_service: ApprovalService,
    config: AppConfig,
    *,
    on_decision: Callable[[], None] | None = None,
    workflow_persist: Callable[[PlanState], None] | None = None,
) -> None:
    task = next(
        (
            item
            for item in state.tasks
            if item.status.value == "waiting_for_approval"
        ),
        None,
    )
    if (
        task is None
        or task.action_id is None
        or task.action_version is None
        or task.approval_id is None
    ):
        st.error("The action proposal is incomplete. Execution remains blocked.")
        return
    try:
        proposal = approval_service.get_action(task.action_id, task.action_version)
        request = approval_service.get_approval(task.approval_id)
    except ApprovalServiceError as exc:
        st.error(f"The approval panel cannot verify this action: {exc}")
        return

    with st.container(border=True):
        st.subheader("Action Approval")
        st.warning(
            f"{proposal.risk_level.value.upper()} risk: {proposal.risk_reason}"
        )
        st.write(
            {
                "purpose": proposal.purpose,
                "tool": proposal.tool_name,
                "side_effect": proposal.side_effect.value,
                "action_id": proposal.action_id,
                "version": proposal.version,
                "status": request.status.value,
            }
        )
        st.markdown(f"**{proposal.preview.get('title', 'Action preview')}**")
        st.caption(proposal.preview.get("impact", ""))
        for label, value in proposal.preview.get("fields", {}).items():
            st.markdown(f"**{label}**")
            if isinstance(value, list):
                st.write(value)
            else:
                st.write(str(value))

        remaining = max(
            0,
            int(
                (
                    parse_timestamp(request.expires_at)
                    - datetime.now(timezone.utc)
                ).total_seconds()
            ),
        )
        st.caption(f"Approval expires in {remaining // 60}:{remaining % 60:02d}")

        if request.status == ApprovalStatus.PENDING:
            approve_col, deny_col, cancel_col = st.columns(3)
            if approve_col.button(
                "Approve",
                type="primary",
                key=f"approve-{request.approval_id}",
                use_container_width=True,
            ):
                try:
                    approval_service.approve(
                        request.approval_id,
                        user_id=config.long_term_memory_user_id,
                        expected_version=proposal.version,
                    )
                    if on_decision is not None:
                        on_decision()
                    st.rerun()
                except ApprovalServiceError as exc:
                    st.error(f"Approval was rejected: {exc}")
            if deny_col.button(
                "Deny",
                key=f"deny-{request.approval_id}",
                use_container_width=True,
            ):
                try:
                    approval_service.deny(
                        request.approval_id,
                        user_id=config.long_term_memory_user_id,
                        expected_version=proposal.version,
                    )
                    if on_decision is not None:
                        on_decision()
                    st.rerun()
                except ApprovalServiceError as exc:
                    st.error(f"Denial was rejected: {exc}")
            if cancel_col.button(
                "Cancel",
                key=f"cancel-{request.approval_id}",
                use_container_width=True,
            ):
                try:
                    approval_service.cancel(
                        request.approval_id,
                        user_id=config.long_term_memory_user_id,
                        expected_version=proposal.version,
                    )
                    if on_decision is not None:
                        on_decision()
                    st.rerun()
                except ApprovalServiceError as exc:
                    st.error(f"Cancellation was rejected: {exc}")

            _render_action_editor(
                state=state,
                task=task,
                proposal=proposal,
                request=request,
                approval_service=approval_service,
                config=config,
                workflow_persist=workflow_persist,
            )

        with st.expander("Approval audit trail"):
            st.json(
                [
                    {
                        "event": event.event_type,
                        "action_version": event.action_version,
                        "approval_id": event.approval_id,
                        "created_at": event.created_at,
                        "metadata": event.metadata,
                    }
                    for event in approval_service.list_audit(proposal.action_id)
                ]
            )


def _render_action_editor(
    *,
    state: PlanState,
    task,
    proposal,
    request,
    approval_service: ApprovalService,
    config: AppConfig,
    workflow_persist: Callable[[PlanState], None] | None = None,
) -> None:
    with st.expander("Edit action before approval"):
        with st.form(f"edit-action-{request.approval_id}"):
            if proposal.tool_name == "email.send_mock":
                edited_arguments = {
                    "to": st.text_input("To", value=proposal.arguments["to"]),
                    "subject": st.text_input(
                        "Subject", value=proposal.arguments["subject"]
                    ),
                    "body": st.text_area(
                        "Body", value=proposal.arguments["body"], height=180
                    ),
                }
            elif proposal.tool_name == "files.delete_mock":
                paths = st.text_area(
                    "Mock paths, one per line",
                    value="\n".join(proposal.arguments["paths"]),
                    height=140,
                )
                edited_arguments = {
                    "paths": [line.strip() for line in paths.splitlines() if line.strip()]
                }
            else:
                raw_arguments = st.text_area(
                    "Arguments JSON",
                    value=json.dumps(proposal.arguments, indent=2),
                    height=180,
                )
                try:
                    edited_arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    edited_arguments = None
            submitted = st.form_submit_button(
                "Create revised proposal", use_container_width=True
            )
        if submitted:
            if not isinstance(edited_arguments, dict):
                st.error("Edited arguments must be a valid JSON object.")
                return
            try:
                new_proposal, new_request = approval_service.edit(
                    request.approval_id,
                    user_id=config.long_term_memory_user_id,
                    expected_version=proposal.version,
                    arguments=edited_arguments,
                )
                task.tool_arguments = dict(new_proposal.arguments)
                task.action_version = new_proposal.version
                task.approval_id = new_request.approval_id
                state.metrics.actions_edited += 1
                state.record_event(
                    "ACTION EDITED",
                    task_id=task.task_id,
                    metadata={
                        "action_id": new_proposal.action_id,
                        "action_version": new_proposal.version,
                    },
                )
                if workflow_persist is not None:
                    workflow_persist(state)
                else:
                    approval_service.save_workflow(
                        state, user_id=config.long_term_memory_user_id
                    )
                st.rerun()
            except Exception as exc:
                st.error(f"The revised proposal was rejected: {exc}")


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
