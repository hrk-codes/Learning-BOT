from agent.agent_loop import run_agent_loop
from agent.decision_schema import DecisionParseError, parse_agent_decision


def test_agent_finishes_quickly_for_simple_goal() -> None:
    def fake_llm(_messages):
        return """
        {
          "action": "FINISH",
          "status": "simple goal answered",
          "content": "Three Python interview topics are lists, dictionaries, and functions.",
          "finished": true
        }
        """

    state = run_agent_loop(
        goal="Give me three Python interview topics.",
        conversation_context=[],
        max_iterations=4,
        llm_decision_fn=fake_llm,
    )

    assert state.status == "completed"
    assert state.iteration_count == 1
    assert state.trace[0].action == "FINISH"
    assert "lists" in state.final_answer


def test_agent_can_take_multiple_steps_before_finish() -> None:
    decisions = iter(
        [
            '{"action":"ANALYZE","status":"requirements identified","content":"Need roadmap topics.","finished":false}',
            '{"action":"PLAN","status":"plan drafted","content":"Draft plan includes Python, APIs, LLMs, memory, and agents.","finished":false}',
            '{"action":"FINISH","status":"goal complete","content":"Final learning plan: Python, APIs, LLMs, memory, agents.","finished":true}',
        ]
    )

    state = run_agent_loop(
        goal="Create a structured learning plan for becoming an AI agent engineer.",
        conversation_context=[],
        max_iterations=4,
        llm_decision_fn=lambda _messages: next(decisions),
    )

    assert state.status == "completed"
    assert state.iteration_count == 3
    assert [step.action for step in state.trace] == ["ANALYZE", "PLAN", "FINISH"]


def test_max_iterations_stops_unfinished_agent() -> None:
    state = run_agent_loop(
        goal="Continue forever.",
        conversation_context=[],
        max_iterations=2,
        llm_decision_fn=lambda _messages: (
            '{"action":"CONTINUE","status":"not done","content":"Still working.","finished":false}'
        ),
    )

    assert state.status == "max_iterations"
    assert state.iteration_count == 2
    assert "maximum iteration limit" in state.final_answer


def test_malformed_decision_stops_safely() -> None:
    state = run_agent_loop(
        goal="Handle bad model output.",
        conversation_context=[],
        max_iterations=4,
        llm_decision_fn=lambda _messages: "this is not json",
    )

    assert state.status == "error"
    assert state.iteration_count == 1
    assert "decision step failed" in state.final_answer


def test_decision_schema_rejects_invalid_action() -> None:
    try:
        parse_agent_decision(
            '{"action":"TOOL_CALL","status":"too early","content":"No tools in Stage 3.","finished":false}'
        )
    except DecisionParseError as exc:
        assert "Unknown action" in str(exc)
    else:
        raise AssertionError("Expected DecisionParseError")


if __name__ == "__main__":
    test_agent_finishes_quickly_for_simple_goal()
    test_agent_can_take_multiple_steps_before_finish()
    test_max_iterations_stops_unfinished_agent()
    test_malformed_decision_stops_safely()
    test_decision_schema_rejects_invalid_action()
    print("agent tests passed")
