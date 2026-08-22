from __future__ import annotations

import logging
import threading
import time
from datetime import timezone
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from platform_api.config import PlatformSettings
from platform_api.models import AgentRun, Document, RunStatus, UsageRecord, utc_now
from platform_api.run_service import append_event, claim_next_run
from platform_api.runtime_adapter import Stage10RuntimeAdapter, build_workspace_rag
from config import get_config


logger = logging.getLogger("platform.worker")


class RunCancellationRequested(Exception):
    pass


class WorkerService:
    def __init__(self, settings: PlatformSettings, session_factory: sessionmaker[Session]) -> None:
        self.settings = settings
        self.session_factory = session_factory
        self.runtime = Stage10RuntimeAdapter(settings)

    def process_once(self) -> bool:
        with self.session_factory() as db:
            document = self._claim_document(db)
        if document is not None:
            self._index_document(document.id)
            return True
        with self.session_factory() as db:
            run = claim_next_run(db)
        if run is None:
            return False
        self._execute_run(run.id)
        return True

    def _execute_run(self, run_id: str) -> None:
        started = time.perf_counter()
        try:
            with self.session_factory() as db:
                run = db.get(AgentRun, run_id)
                if run is None:
                    return
                append_event(db, run, "run.started", "Worker started the agent workflow.", node="manager", status="running")
                db.commit()
                user_id, workspace_id, goal = run.user_id, run.workspace_id, run.goal

            with self.session_factory() as db:
                def persist_live_event(event: dict) -> None:
                    if time.perf_counter() - started > self.settings.max_run_seconds:
                        raise TimeoutError("The run exceeded its maximum execution time.")
                    self._persist_trace_event(run_id, event)

                result = self.runtime.execute(
                    db,
                    user_id=user_id,
                    workspace_id=workspace_id,
                    goal=goal,
                    on_trace_event=persist_live_event,
                )

            with self.session_factory() as db:
                run = db.get(AgentRun, run_id)
                if run is None:
                    return
                if run.cancellation_requested:
                    run.status = RunStatus.CANCELLED.value
                    run.error_code = "cancelled_by_user"
                    append_event(db, run, "run.cancelled", "Run stopped after its current execution boundary.", status="cancelled")
                else:
                    self._record_metrics(db, run, result.metrics)
                    if run.llm_calls > self.settings.max_llm_calls or (run.input_tokens + run.output_tokens) > self.settings.max_run_tokens:
                        raise RuntimeError("The run exceeded its configured LLM budget.")
                    run.status = RunStatus.COMPLETED.value
                    run.result_text = result.answer
                    append_event(db, run, "run.completed", "Agent workflow completed.", node="finalize", status="completed")
                run.progress = 1.0
                run.current_node = None
                run.completed_at = utc_now()
                run.execution_seconds = time.perf_counter() - started
                db.commit()
        except RunCancellationRequested:
            with self.session_factory() as db:
                run = db.get(AgentRun, run_id)
                if run is None:
                    return
                run.status = RunStatus.CANCELLED.value
                run.error_code = "cancelled_by_user"
                run.error_message = None
                run.progress = 1.0
                run.current_node = None
                run.completed_at = utc_now()
                run.execution_seconds = time.perf_counter() - started
                append_event(db, run, "run.cancelled", "Run stopped at an execution boundary.", status="cancelled")
                db.commit()
        except Exception as exc:
            logger.exception("Agent run failed run_id=%s error_type=%s", run_id, type(exc).__name__)
            with self.session_factory() as db:
                run = db.get(AgentRun, run_id)
                if run is None:
                    return
                run.status = RunStatus.FAILED.value
                run.error_code = type(exc).__name__
                run.error_message = _public_failure(exc)
                run.progress = 1.0
                run.current_node = None
                run.completed_at = utc_now()
                run.execution_seconds = time.perf_counter() - started
                append_event(db, run, "run.failed", run.error_message, status="failed")
                db.commit()

    def _persist_trace_event(self, run_id: str, event: dict) -> None:
        with self.session_factory() as db:
            run = db.get(AgentRun, run_id)
            if run is None:
                return
            if run.cancellation_requested:
                raise RunCancellationRequested()
            details = dict(event.get("details") or {})
            tools = details.get("tools_used") or []
            run.tool_calls += len(tools)
            run.current_node = str(event.get("next_node") or event.get("node") or "agent")
            run.progress = min(0.9, run.progress + 0.1)
            append_event(
                db,
                run,
                "agent.node",
                f"{event.get('agent_name', 'agent')} completed {event.get('node', 'step')}.",
                node=str(event.get("node") or "agent"),
                status=str(event.get("status") or "completed"),
                public_data={
                    "agent": event.get("agent_name"),
                    "next_node": event.get("next_node"),
                    "duration_seconds": event.get("duration_seconds"),
                    "tools_used": tools,
                },
            )
            db.commit()

    def _record_metrics(self, db: Session, run: AgentRun, metrics) -> None:
        for metric in metrics:
            usage = metric.provider_usage
            input_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
            output_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)))
            run.llm_calls += 1
            run.input_tokens += input_tokens
            run.output_tokens += output_tokens
            db.add(
                UsageRecord(
                    run_id=run.id,
                    user_id=run.user_id,
                    model=metric.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_tokens=int(usage.get("cached_tokens", 0)),
                    elapsed_seconds=metric.total_seconds,
                )
            )

    def _claim_document(self, db: Session) -> Document | None:
        query = select(Document).where(Document.status == "queued").order_by(Document.created_at).limit(1)
        if db.bind is not None and db.bind.dialect.name == "postgresql":
            query = query.with_for_update(skip_locked=True)
        document = db.scalar(query)
        if document is None:
            return None
        claimed = db.execute(
            update(Document).where(Document.id == document.id, Document.status == "queued").values(status="processing")
        )
        if claimed.rowcount != 1:
            db.rollback()
            return None
        db.commit()
        return db.get(Document, document.id)

    def _index_document(self, document_id: str) -> None:
        try:
            with self.session_factory() as db:
                document = db.get(Document, document_id)
                if document is None:
                    return
                object_path = self.settings.object_store_path / document.object_key
                content = object_path.read_bytes()
                pipeline = build_workspace_rag(self.settings, document.workspace_id, get_config())
                result = pipeline.index_pdf(
                    document.filename,
                    content,
                    version=document.version,
                    user_id=document.user_id,
                    document_id=document.id,
                )
                document.status = "indexed"
                document.chunk_count = result.chunk_count
                document.indexed_at = utc_now()
                db.commit()
        except Exception as exc:
            logger.exception("Document indexing failed document_id=%s", document_id)
            with self.session_factory() as db:
                document = db.get(Document, document_id)
                if document:
                    document.status = "failed"
                    db.commit()


class WorkerThread:
    def __init__(self, worker: WorkerService) -> None:
        self.worker = worker
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="platform-worker", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            if not self.worker.process_once():
                self.stop_event.wait(self.worker.settings.worker_poll_seconds)


def _public_failure(exc: Exception) -> str:
    message = str(exc)
    if "GROQ_API_KEY" in message:
        return "The AI provider is not configured. Ask an administrator to check the server environment."
    if "429" in message or "rate limit" in message.lower():
        return "The AI provider is temporarily rate limited. Retry this run later."
    return "The agent workflow could not complete. Review the trace and server request ID."
