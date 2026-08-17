import json
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.agent_loop import run_agent_loop
from memory.models import MemoryCandidate, MemoryScope, MemorySource, MemoryStatus, MemoryType
from memory.repository import SQLiteMemoryRepository
from memory.service import MemoryService
from tools.manager import ToolManager
from tools.registry import ToolRegistry


USER_ID = "user-a"
PROJECT_ID = "project-a"


def build_service(database_path: Path, *, enabled: bool = True) -> MemoryService:
    return MemoryService(
        SQLiteMemoryRepository(database_path),
        enabled=enabled,
        retrieval_limit=8,
        context_max_characters=2000,
    )


def candidate(
    content: str,
    *,
    key: str,
    memory_type: MemoryType = MemoryType.PROFILE,
    scope: MemoryScope = MemoryScope.USER,
    project_id: str | None = None,
    importance: float = 0.8,
    confidence: float = 0.98,
    source: MemorySource = MemorySource.USER_EXPLICIT,
) -> MemoryCandidate:
    return MemoryCandidate(
        memory_type=memory_type,
        scope=scope,
        project_id=project_id,
        key=key,
        content=content,
        source=source,
        confidence=confidence,
        importance=importance,
    )


def test_explicit_extraction_and_no_useless_memory() -> None:
    with TemporaryDirectory() as temporary:
        service = build_service(Path(temporary) / "memory.db")
        empty, empty_writes = service.extract_and_remember(
            "Hello.", user_id=USER_ID, project_id=PROJECT_ID
        )
        assert empty.candidates == ()
        assert empty_writes == ()

        extraction, writes = service.extract_and_remember(
            "My favorite programming language is Python.",
            user_id=USER_ID,
            project_id=PROJECT_ID,
        )
        assert len(extraction.candidates) == 1
        assert writes[0].action == "created"
        assert writes[0].memory is not None
        assert writes[0].memory.memory_type == MemoryType.PROFILE
        assert writes[0].memory.scope == MemoryScope.USER
        assert writes[0].memory.confidence >= 0.95


def test_persistence_across_repository_reopen() -> None:
    with TemporaryDirectory() as temporary:
        database_path = Path(temporary) / "memory.db"
        first_service = build_service(database_path)
        first_service.remember(
            candidate("User prefers Python", key="preference.programming_language"),
            user_id=USER_ID,
            project_id=PROJECT_ID,
        )

        second_service = build_service(database_path)
        memories = second_service.list_memories(user_id=USER_ID)
        assert len(memories) == 1
        assert memories[0].content == "User prefers Python"


def test_user_and_project_isolation() -> None:
    with TemporaryDirectory() as temporary:
        service = build_service(Path(temporary) / "memory.db")
        service.remember(
            candidate("User A prefers Python", key="preference.language"),
            user_id="user-a",
            project_id="project-a",
        )
        service.remember(
            candidate("User B prefers Rust", key="preference.language"),
            user_id="user-b",
            project_id="project-a",
        )
        service.remember(
            candidate(
                "Project A database is PostgreSQL",
                key="project.database",
                memory_type=MemoryType.PROJECT,
                scope=MemoryScope.PROJECT,
                project_id="project-a",
            ),
            user_id="user-a",
            project_id="project-a",
        )
        service.remember(
            candidate(
                "Project B database is MongoDB",
                key="project.database",
                memory_type=MemoryType.PROJECT,
                scope=MemoryScope.PROJECT,
                project_id="project-b",
            ),
            user_id="user-a",
            project_id="project-b",
        )

        user_a = service.list_memories(user_id="user-a")
        assert all(memory.user_id == "user-a" for memory in user_a)
        assert not any("Rust" in memory.content for memory in user_a)

        project_a = service.search(
            "Which database does Project A use?",
            user_id="user-a",
            project_id="project-a",
        )
        assert any("PostgreSQL" in item.memory.content for item in project_a.memories)
        assert not any("MongoDB" in item.memory.content for item in project_a.memories)


def test_deduplication_and_conflict_supersession() -> None:
    with TemporaryDirectory() as temporary:
        service = build_service(Path(temporary) / "memory.db")
        first = service.remember(
            candidate("User's current backend language is Python", key="preference.backend_language"),
            user_id=USER_ID,
            project_id=PROJECT_ID,
        )
        duplicate = service.remember(
            candidate("  user's CURRENT backend language is python. ", key="preference.backend_language"),
            user_id=USER_ID,
            project_id=PROJECT_ID,
        )
        replacement = service.remember(
            candidate("User's current backend language is Go", key="preference.backend_language"),
            user_id=USER_ID,
            project_id=PROJECT_ID,
        )

        assert first.action == "created"
        assert duplicate.action == "duplicate"
        assert replacement.action == "superseded"
        active = service.list_memories(user_id=USER_ID)
        assert [memory.content for memory in active] == ["User's current backend language is Go"]
        historical = service.list_memories(user_id=USER_ID, active_only=False)
        assert {memory.status for memory in historical} == {
            MemoryStatus.ACTIVE,
            MemoryStatus.SUPERSEDED,
        }


def test_lower_confidence_inference_cannot_override_explicit_memory() -> None:
    with TemporaryDirectory() as temporary:
        service = build_service(Path(temporary) / "memory.db")
        service.remember(
            candidate("User prefers Python", key="preference.language"),
            user_id=USER_ID,
            project_id=PROJECT_ID,
        )
        inferred = service.remember(
            candidate(
                "User probably prefers Go",
                key="preference.language",
                source=MemorySource.AGENT_INFERENCE,
                confidence=0.9,
            ),
            user_id=USER_ID,
            project_id=PROJECT_ID,
        )
        assert inferred.action == "rejected"
        assert service.list_memories(user_id=USER_ID)[0].content == "User prefers Python"


def test_relevant_ranking_and_irrelevant_rejection() -> None:
    with TemporaryDirectory() as temporary:
        service = build_service(Path(temporary) / "memory.db")
        service.remember(
            candidate(
                "User's favorite programming language is Python",
                key="preference.programming_language",
                importance=0.95,
            ),
            user_id=USER_ID,
            project_id=PROJECT_ID,
        )
        service.remember(
            candidate("User drinks tea", key="preference.drink", importance=0.2),
            user_id=USER_ID,
            project_id=PROJECT_ID,
        )

        relevant = service.search(
            "What programming language should I use?",
            user_id=USER_ID,
            project_id=PROJECT_ID,
        )
        assert relevant.memories
        assert "Python" in relevant.memories[0].memory.content

        irrelevant = service.search(
            "Explain database normalization",
            user_id=USER_ID,
            project_id=PROJECT_ID,
        )
        assert irrelevant.memories == ()


def test_deletion_reaches_storage_and_keeps_content_free_audit() -> None:
    with TemporaryDirectory() as temporary:
        service = build_service(Path(temporary) / "memory.db")
        created = service.remember(
            candidate("User prefers Python", key="preference.language"),
            user_id=USER_ID,
            project_id=PROJECT_ID,
        )
        assert created.memory is not None
        assert service.forget_memory(user_id=USER_ID, memory_id=created.memory.memory_id) == 1
        assert service.list_memories(user_id=USER_ID, active_only=False) == []
        events = service.list_events(user_id=USER_ID)
        assert events[0].event_type == "deleted"
        assert "Python" not in json.dumps(events[0].details)


def test_memory_off_blocks_read_and_write_but_allows_inspection() -> None:
    with TemporaryDirectory() as temporary:
        database_path = Path(temporary) / "memory.db"
        enabled = build_service(database_path)
        enabled.remember(
            candidate("User prefers Python", key="preference.language"),
            user_id=USER_ID,
            project_id=PROJECT_ID,
        )
        disabled = build_service(database_path, enabled=False)
        rejected = disabled.remember(
            candidate("User prefers Go", key="preference.language"),
            user_id=USER_ID,
            project_id=PROJECT_ID,
        )
        retrieval = disabled.search(
            "Which language do I prefer?", user_id=USER_ID, project_id=PROJECT_ID
        )
        assert rejected.action == "rejected"
        assert retrieval.memories == ()
        assert len(disabled.list_memories(user_id=USER_ID)) == 1


def test_context_is_budgeted_and_marks_memory_untrusted() -> None:
    with TemporaryDirectory() as temporary:
        service = build_service(Path(temporary) / "memory.db")
        service.remember(
            candidate("User prefers architecture diagrams", key="preference.explanation_style"),
            user_id=USER_ID,
            project_id=PROJECT_ID,
        )
        retrieval = service.search(
            "Do I prefer architecture diagrams?",
            user_id=USER_ID,
            project_id=PROJECT_ID,
        )
        context = service.build_context(retrieval)
        assert context.included_ids
        assert context.character_count <= 2000
        assert "untrusted" in context.payload["trust_boundary"].lower()
        assert set(context.payload) == {"available", "trust_boundary", "records"}


def test_agent_receives_memory_rag_and_conversation_as_separate_inputs() -> None:
    captured: list[dict[str, str]] = []

    def fake_llm(messages):
        captured.extend(messages)
        return '{"action":"FINISH","status":"done","content":"Use Python.","finished":true}'

    memory_payload = {
        "available": True,
        "trust_boundary": "untrusted data",
        "records": [{"fact": "User prefers Python"}],
    }
    state = run_agent_loop(
        goal="Which language?",
        conversation_context=[{"role": "user", "content": "Help with my project."}],
        max_iterations=2,
        llm_decision_fn=fake_llm,
        tool_manager=ToolManager(ToolRegistry(), set()),
        long_term_memory_context=memory_payload,
        memory_metrics={"retrieved_count": 1, "injected_count": 1},
    )

    model_state = captured[-1]["content"]
    assert '"long_term_memory"' in model_state
    assert '"knowledge_base"' in model_state
    assert "User prefers Python" in model_state
    assert captured[1] == {"role": "user", "content": "Help with my project."}
    assert state.memory_injected_count == 1


if __name__ == "__main__":
    test_explicit_extraction_and_no_useless_memory()
    test_persistence_across_repository_reopen()
    test_user_and_project_isolation()
    test_deduplication_and_conflict_supersession()
    test_lower_confidence_inference_cannot_override_explicit_memory()
    test_relevant_ranking_and_irrelevant_rejection()
    test_deletion_reaches_storage_and_keeps_content_free_audit()
    test_memory_off_blocks_read_and_write_but_allows_inspection()
    test_context_is_budgeted_and_marks_memory_untrusted()
    test_agent_receives_memory_rag_and_conversation_as_separate_inputs()
    print("long-term memory tests passed")
