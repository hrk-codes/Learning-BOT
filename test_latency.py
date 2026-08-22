import json
from dataclasses import replace

from agent import planned_agent
from config import get_config
from latency.fast_path import classify_fast_path
from llm import groq_client
from planner.models import PlanState
from tools.manager import ToolManager
from tools.registry import ToolRegistry


def test_simple_question_uses_fast_path() -> None:
    result = classify_fast_path("Explain HTTP headers in simple terms.")
    assert result.eligible is True


def test_capability_requests_bypass_fast_path() -> None:
    for goal in (
        "Search the web for the latest Python news.",
        "Answer from my uploaded PDF document.",
        "Remember that I prefer concise answers.",
        "Calculate 123 times 456.",
        "Research two databases, compare them, and recommend one.",
    ):
        assert classify_fast_path(goal).eligible is False


class StreamingResponse:
    status_code = 200
    headers: dict[str, str] = {}

    def iter_lines(self, decode_unicode: bool = True):
        yield 'data: {"choices":[{"delta":{"content":"Hello"}}]}'
        yield 'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":8}}}'
        yield "data: [DONE]"

    def close(self) -> None:
        pass


def test_stream_metrics_capture_first_token_and_safe_usage() -> None:
    captured: list[groq_client.LatencyMetrics] = []
    original_post = groq_client.requests.post
    try:
        groq_client.requests.post = lambda *args, **kwargs: StreamingResponse()
        config = replace(get_config(), groq_api_key="test-key")
        answer = "".join(
            groq_client.stream_chat_completion(
                config=config,
                messages=[{"role": "user", "content": "not logged"}],
                model="openai/gpt-oss-20b",
                temperature=0.2,
                max_tokens=32,
                on_metrics=captured.append,
            )
        )
    finally:
        groq_client.requests.post = original_post

    assert answer == "Hello"
    assert len(captured) == 1
    metric = captured[0]
    assert metric.time_to_first_token_seconds is not None
    assert metric.provider_usage == {
        "prompt_tokens": 12,
        "completion_tokens": 1,
        "cached_tokens": 8,
    }
    assert "not logged" not in json.dumps(metric.public_dict())


def test_planning_uses_fast_model_and_final_synthesis_uses_final_model() -> None:
    calls: list[str] = []
    original_completion = planned_agent.complete_chat_completion

    def fake_completion(*, model: str, **kwargs) -> str:
        calls.append(model)
        system = kwargs["messages"][0]["content"]
        if "Write the final user-facing answer" in system:
            return "Final answer from the final model."
        return json.dumps(
            {
                "goal_satisfied": True,
                "reason": "Verified",
                "final_answer": "Fast provisional answer",
                "replan_needed": False,
                "missing": [],
            }
        )

    try:
        planned_agent.complete_chat_completion = fake_completion
        runtime = planned_agent.build_planning_runtime(
            config=get_config(),
            conversation_context=[],
            model="fast-model",
            final_model="final-model",
            temperature=0.2,
            max_tokens=64,
            tool_manager=ToolManager(ToolRegistry(), set()),
            rag_pipeline=None,
            rag_top_k=1,
            rag_min_score=0.0,
            long_term_memory_context=None,
            memory_search_fn=None,
        )
        result = runtime.evaluator.evaluate(PlanState(goal="Explain a concept", tasks=[]))
    finally:
        planned_agent.complete_chat_completion = original_completion

    assert calls == ["fast-model", "final-model"]
    assert result.final_answer == "Final answer from the final model."
