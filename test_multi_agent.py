from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

from langgraph.checkpoint.sqlite import SqliteSaver

from multi_agent.agents.base import AgentConfig
from multi_agent.agents.manager import ManagerAgent
from multi_agent.agents.researcher import ResearcherAgent
from multi_agent.agents.reviewer import ReviewerAgent
from multi_agent.agents.writer import WriterAgent
from multi_agent.graph import build_multi_agent_graph
from multi_agent.nodes import MultiAgentDependencies
from prompts.manager_prompt import MANAGER_SYSTEM_PROMPT
from prompts.researcher_prompt import RESEARCHER_SYSTEM_PROMPT
from prompts.reviewer_prompt import REVIEWER_SYSTEM_PROMPT
from prompts.writer_prompt import WRITER_SYSTEM_PROMPT
from tools.manager import ToolManager
from tools.registry import ToolRegistry


def test_simple_goal_finishes_without_specialist_delegation(tmp_path: Path) -> None:
    graph = build_test_graph(tmp_path, FakeLLM())
    result = graph.invoke(initial_state("Explain HTTP headers."), graph_config("simple"))

    assert node_names(result) == ["manager", "finalize"]
    assert result["agent_results"] == []
    assert result["final_answer"] == "Final answer."


def test_research_goal_routes_to_researcher_then_manager(tmp_path: Path) -> None:
    graph = build_test_graph(tmp_path, FakeLLM(), rag=FakeRag())
    result = graph.invoke(initial_state("Research PostgreSQL vs MySQL.", knowledge=True), graph_config("research"))

    assert node_names(result) == ["manager", "researcher", "manager", "finalize"]
    research = result["research_result"]
    assert research["status"] == "completed"
    assert research["metadata"]["rag_used"] is True
    assert research["output"]["claims"][0]["source_ids"] == ["chunk-1"]


def test_research_and_writing_are_separate_specialist_steps(tmp_path: Path) -> None:
    graph = build_test_graph(tmp_path, FakeLLM(), rag=FakeRag())
    result = graph.invoke(
        initial_state("Research databases and write a report.", knowledge=True),
        graph_config("writing"),
    )

    assert node_names(result) == [
        "manager", "researcher", "manager", "writer", "manager", "finalize"
    ]
    assert result["draft_result"]["output"]["draft"] == "Grounded draft."
    assert result["draft_result"]["metadata"]["tools_used"] == []


def test_review_requires_a_writer_revision_before_finalization(tmp_path: Path) -> None:
    llm = FakeLLM(review_responses=[
        '{"status":"revision_required","issues":["Clarify the comparison."],"feedback":"Clarify the comparison."}',
        '{"status":"approved","issues":[],"feedback":""}',
    ])
    graph = build_test_graph(tmp_path, llm, rag=FakeRag())
    result = graph.invoke(
        initial_state("Research databases, write a report, and verify carefully.", knowledge=True),
        graph_config("review"),
    )

    assert node_names(result) == [
        "manager", "researcher", "manager", "writer", "manager", "reviewer",
        "manager", "writer", "manager", "reviewer", "manager", "finalize",
    ]
    assert result["revision_count"] == 1
    assert result["review_result"]["output"]["status"] == "approved"


def test_researcher_failure_is_retried_without_restarting_the_workflow(tmp_path: Path) -> None:
    llm = FakeLLM(research_responses=["not json", _research_json()])
    graph = build_test_graph(tmp_path, llm, rag=FakeRag(), output_repairs=0)
    result = graph.invoke(initial_state("Research a topic.", knowledge=True), graph_config("retry"))

    assert node_names(result) == [
        "manager", "researcher", "manager", "researcher", "manager", "finalize"
    ]
    assert result["agent_attempts"]["researcher"] == 2
    assert result["research_result"]["status"] == "completed"


def test_schema_failure_is_repaired_once_before_returning_a_result(tmp_path: Path) -> None:
    llm = FakeLLM(research_responses=["not json", _research_json()])
    graph = build_test_graph(tmp_path, llm, rag=FakeRag(), output_repairs=1)
    result = graph.invoke(initial_state("Research a topic.", knowledge=True), graph_config("repair"))

    assert result["research_result"]["status"] == "completed"
    assert result["agent_attempts"]["researcher"] == 1
    assert llm.calls_by_role["researcher"] == 2


def test_writer_receives_only_selected_style_memory(tmp_path: Path) -> None:
    llm = FakeLLM()
    graph = build_test_graph(tmp_path, llm, rag=FakeRag())
    state = initial_state("Research databases and write a report.", knowledge=True)
    state["memory_context"] = {
        "records": [
            {"type": "profile", "fact": "Prefer concise reports."},
            {"type": "procedural", "fact": "Use bullet points."},
            {"type": "project", "fact": "Internal secret that writer does not need."},
        ]
    }
    graph.invoke(state, graph_config("memory"))

    writer_payload = llm.payloads_by_role["writer"][0]
    assert writer_payload["style_preferences"] == [
        {"type": "profile", "fact": "Prefer concise reports."},
        {"type": "procedural", "fact": "Use bullet points."},
    ]
    assert "conversation_context" not in writer_payload


def test_researcher_has_only_scoped_search_tool_permission(tmp_path: Path) -> None:
    llm = FakeLLM()
    graph = build_test_graph(tmp_path, llm, rag=FakeRag(), tools={"search.web", "email.send_mock"})
    result = graph.invoke(initial_state("Research a topic.", knowledge=True), graph_config("permissions"))

    task = result["agent_results"][0]
    assert task["agent_name"] == "researcher"
    assert llm.payloads_by_role["researcher"][0]["task"]["allowed_tools"] == ["search.web"]


def test_review_loop_stops_at_the_configured_revision_limit(tmp_path: Path) -> None:
    llm = FakeLLM(review_responses=[
        '{"status":"revision_required","issues":["Still needs work."],"feedback":"Still needs work."}',
        '{"status":"revision_required","issues":["Still needs work."],"feedback":"Still needs work."}',
    ])
    graph = build_test_graph(tmp_path, llm, rag=FakeRag(), max_review_revisions=1)
    result = graph.invoke(
        initial_state(
            "Research, write a report, and review carefully.",
            knowledge=True,
            max_review_revisions=1,
        ),
        graph_config("bounded-review"),
    )

    assert result["revision_count"] == 1
    assert node_names(result)[-1] == "finalize"
    assert len(result["agent_results"]) == 5


def test_delegation_limit_prevents_circular_agent_work(tmp_path: Path) -> None:
    graph = build_test_graph(tmp_path, FakeLLM(), rag=FakeRag(), max_delegations=1)
    result = graph.invoke(
        initial_state("Research databases and write a report.", knowledge=True, max_delegations=1),
        graph_config("bounded-delegation"),
    )

    assert node_names(result) == ["manager", "researcher", "manager", "finalize"]
    assert result["delegation_count"] == 1


def build_test_graph(
    tmp_path: Path,
    llm: "FakeLLM",
    *,
    rag=None,
    tools: set[str] | None = None,
    max_delegations: int = 8,
    max_agent_retries: int = 1,
    max_review_revisions: int = 1,
    output_repairs: int = 1,
):
    def config(name: str, prompt: str, **kwargs) -> AgentConfig:
        return AgentConfig(
            name=name,
            system_prompt=prompt,
            model="test-model",
            temperature=0.0,
            max_tokens=256,
            timeout_seconds=1,
            max_retries=output_repairs,
            **kwargs,
        )

    manager = ManagerAgent(config("manager", MANAGER_SYSTEM_PROMPT), llm)
    researcher = ResearcherAgent(
        config("researcher", RESEARCHER_SYSTEM_PROMPT, allowed_tools=("search.web",), allow_rag=True), llm
    )
    writer = WriterAgent(config("writer", WRITER_SYSTEM_PROMPT, allow_memory=True), llm)
    reviewer = ReviewerAgent(config("reviewer", REVIEWER_SYSTEM_PROMPT), llm)
    registry = ToolRegistry()
    # The manager passes the researcher allow-list independently of registry contents.
    manager_tools = ToolManager(registry, enabled_tools=tools or set())
    saver = SqliteSaver(sqlite3.connect(tmp_path / "multi-agent.db", check_same_thread=False))
    return build_multi_agent_graph(
        MultiAgentDependencies(
            manager=manager,
            researcher=researcher,
            writer=writer,
            reviewer=reviewer,
            rag_pipeline=rag,
            rag_top_k=4,
            rag_min_score=0.25,
            tool_manager=manager_tools,
        ),
        saver,
    )


def initial_state(
    goal: str,
    *,
    knowledge: bool = False,
    max_delegations: int = 8,
    max_agent_retries: int = 1,
    max_review_revisions: int = 1,
) -> dict:
    return {
        "run_id": "run-test",
        "thread_id": "thread-test",
        "user_id": "user-one",
        "goal": goal,
        "conversation_context": [],
        "memory_context": None,
        "knowledge_base": {"available": knowledge, "documents": []},
        "researcher_tools": ["search.web"],
        "current_agent": None,
        "manager_decision": None,
        "current_task": None,
        "research_result": None,
        "draft_result": None,
        "review_result": None,
        "agent_attempts": {},
        "delegation_count": 0,
        "revision_count": 0,
        "max_delegations": max_delegations,
        "max_agent_retries": max_agent_retries,
        "max_review_revisions": max_review_revisions,
        "final_answer": "",
        "status": "running",
        "error": None,
        "started_at": "2026-08-21T00:00:00+00:00",
        "completed_at": None,
        "agent_results": [],
        "node_trace": [],
    }


def graph_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}


def node_names(result: dict) -> list[str]:
    return [event["node"] for event in result["node_trace"]]


class FakeRag:
    def retrieve(self, *_args, **_kwargs):
        chunk = SimpleNamespace(
            chunk_id="chunk-1",
            text="PostgreSQL and MySQL have distinct strengths.",
            score=0.91,
            metadata={"filename": "databases.pdf", "page_number": 2},
        )
        return SimpleNamespace(chunks=[chunk])


class FakeLLM:
    def __init__(self, *, research_responses: list[str] | None = None, review_responses: list[str] | None = None) -> None:
        self.research_responses = list(research_responses or [])
        self.review_responses = list(review_responses or [])
        self.calls_by_role: dict[str, int] = {}
        self.payloads_by_role: dict[str, list[dict]] = {}

    def __call__(self, messages: list[dict[str, str]], _timeout: int) -> str:
        system = messages[0]["content"]
        role = _role_for_prompt(system)
        self.calls_by_role[role] = self.calls_by_role.get(role, 0) + 1
        import json

        payload = json.loads(messages[-1]["content"].split("\n", 1)[-1])
        self.payloads_by_role.setdefault(role, []).append(payload)
        if role == "researcher":
            return self.research_responses.pop(0) if self.research_responses else _research_json()
        if role == "writer":
            return '{"draft":"Grounded draft."}'
        if role == "reviewer":
            return self.review_responses.pop(0) if self.review_responses else '{"status":"approved","issues":[],"feedback":""}'
        return '{"final_answer":"Final answer."}'


def _role_for_prompt(prompt: str) -> str:
    if "Researcher" in prompt:
        return "researcher"
    if "Writer" in prompt:
        return "writer"
    if "Reviewer" in prompt:
        return "reviewer"
    return "manager"


def _research_json() -> str:
    return (
        '{"claims":[{"claim":"The evidence supports a comparison.",'
        '"source_ids":["chunk-1"],"confidence":0.9}],'
        '"sources":[{"source_id":"chunk-1","label":"databases.pdf page 2"}],'
        '"gaps":[],"confidence":0.9}'
    )
