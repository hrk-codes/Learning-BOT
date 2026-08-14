from agent.agent_loop import run_agent_loop
from agent.decision_schema import DecisionParseError, parse_agent_decision
from tools.factory import build_default_registry
from tools.manager import ToolManager
from tools.registry import ToolRegistry
from tools.base import ToolDefinition, ToolResult


def empty_tool_manager() -> ToolManager:
    return ToolManager(registry=ToolRegistry(), enabled_tools=set())


def default_tool_manager(enabled_tools: set[str] | None = None) -> ToolManager:
    registry = build_default_registry()
    if enabled_tools is None:
        enabled_tools = {tool.name for tool in registry.list_tools()}
    return ToolManager(registry=registry, enabled_tools=enabled_tools)


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
        tool_manager=empty_tool_manager(),
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
        tool_manager=empty_tool_manager(),
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
        tool_manager=empty_tool_manager(),
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
        tool_manager=empty_tool_manager(),
    )

    assert state.status == "error"
    assert state.iteration_count == 1
    assert "decision step failed" in state.final_answer


def test_decision_schema_rejects_invalid_action() -> None:
    try:
        parse_agent_decision(
            '{"action":"SHELL","status":"dangerous","content":"No shell in Stage 4.","finished":false}'
        )
    except DecisionParseError as exc:
        assert "Unknown action" in str(exc)
    else:
        raise AssertionError("Expected DecisionParseError")


def test_calculator_tool_call_becomes_observation_then_finish() -> None:
    decisions = iter(
        [
            '{"action":"TOOL_CALL","status":"need exact math","content":"Calculate the product.","tool_name":"calculator.evaluate","tool_arguments":{"expression":"12345 * 678"},"finished":false}',
            '{"action":"FINISH","status":"answered from tool result","content":"12345 * 678 = 8,369,910.","finished":true}',
        ]
    )

    state = run_agent_loop(
        goal="What is 12345 * 678?",
        conversation_context=[],
        max_iterations=4,
        llm_decision_fn=lambda _messages: next(decisions),
        tool_manager=default_tool_manager({"calculator.evaluate"}),
    )

    assert state.status == "completed"
    assert state.tool_call_count == 1
    assert state.trace[0].tool_name == "calculator.evaluate"
    assert state.trace[0].tool_result["success"] is True
    assert state.trace[0].tool_result["result"]["result"] == 8369910


def test_disabled_tool_returns_error_observation() -> None:
    decisions = iter(
        [
            '{"action":"TOOL_CALL","status":"weather requested","content":"Try weather.","tool_name":"weather.get_current","tool_arguments":{"location":"Delhi"},"finished":false}',
            '{"action":"FINISH","status":"tool unavailable","content":"Weather is disabled, so I cannot check current weather.","finished":true}',
        ]
    )

    state = run_agent_loop(
        goal="What's the current weather in Delhi?",
        conversation_context=[],
        max_iterations=4,
        llm_decision_fn=lambda _messages: next(decisions),
        tool_manager=default_tool_manager({"calculator.evaluate"}),
    )

    assert state.status == "completed"
    assert state.trace[0].tool_result["success"] is False
    assert "disabled" in state.trace[0].tool_result["error"]


def test_invalid_tool_arguments_are_blocked() -> None:
    decisions = iter(
        [
            '{"action":"TOOL_CALL","status":"bad args","content":"Try calculator.","tool_name":"calculator.evaluate","tool_arguments":{"expr":"2+2"},"finished":false}',
            '{"action":"FINISH","status":"validation handled","content":"The calculator request was invalid because expression was missing.","finished":true}',
        ]
    )

    state = run_agent_loop(
        goal="Calculate 2+2 with invalid args.",
        conversation_context=[],
        max_iterations=4,
        llm_decision_fn=lambda _messages: next(decisions),
        tool_manager=default_tool_manager({"calculator.evaluate"}),
    )

    assert state.trace[0].tool_result["success"] is False
    assert "missing required field" in state.trace[0].tool_result["error"]


def test_tool_failure_becomes_observation_not_crash() -> None:
    def failing_tool(_arguments):
        return ToolResult(success=False, error="simulated outage")

    registry = ToolRegistry()
    registry.register_tool(
        ToolDefinition(
            name="demo.fail",
            description="Always fails for tests.",
            input_schema={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
            output_schema={"type": "object"},
            permission="safe",
            timeout_seconds=1,
            version="1.0",
            execute=failing_tool,
        )
    )
    decisions = iter(
        [
            '{"action":"TOOL_CALL","status":"call failing tool","content":"Call demo tool.","tool_name":"demo.fail","tool_arguments":{"value":"x"},"finished":false}',
            '{"action":"FINISH","status":"failure explained","content":"The tool failed with a simulated outage.","finished":true}',
        ]
    )

    state = run_agent_loop(
        goal="Show tool failure handling.",
        conversation_context=[],
        max_iterations=4,
        llm_decision_fn=lambda _messages: next(decisions),
        tool_manager=ToolManager(registry=registry, enabled_tools={"demo.fail"}),
    )

    assert state.status == "completed"
    assert state.trace[0].tool_result["success"] is False
    assert state.trace[0].tool_result["error"] == "simulated outage"


def test_multi_tool_sequence() -> None:
    registry = ToolRegistry()
    registry.register_tool(
        ToolDefinition(
            name="weather.get_current",
            description="Fake weather test tool.",
            input_schema={"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
            output_schema={"type": "object"},
            permission="read_only_external",
            timeout_seconds=1,
            version="test",
            execute=lambda _arguments: ToolResult(success=True, result={"temperature_c": 30}),
        )
    )
    registry.register_tool(build_default_registry().get_tool("calculator.evaluate"))
    decisions = iter(
        [
            '{"action":"TOOL_CALL","status":"get weather","content":"Need Delhi weather.","tool_name":"weather.get_current","tool_arguments":{"location":"Delhi"},"finished":false}',
            '{"action":"TOOL_CALL","status":"convert temperature","content":"Convert Celsius to Fahrenheit.","tool_name":"calculator.evaluate","tool_arguments":{"expression":"30 * 9 / 5 + 32"},"finished":false}',
            '{"action":"FINISH","status":"multi-tool answer complete","content":"Delhi weather was checked, then Celsius was converted to Fahrenheit.","finished":true}',
        ]
    )

    state = run_agent_loop(
        goal="Find current weather for Delhi and calculate the Fahrenheit equivalent.",
        conversation_context=[],
        max_iterations=5,
        llm_decision_fn=lambda _messages: next(decisions),
        tool_manager=ToolManager(registry=registry, enabled_tools={"weather.get_current", "calculator.evaluate"}),
    )

    assert state.status == "completed"
    assert [step.tool_name for step in state.trace if step.tool_name] == [
        "weather.get_current",
        "calculator.evaluate",
    ]


if __name__ == "__main__":
    test_agent_finishes_quickly_for_simple_goal()
    test_agent_can_take_multiple_steps_before_finish()
    test_max_iterations_stops_unfinished_agent()
    test_malformed_decision_stops_safely()
    test_decision_schema_rejects_invalid_action()
    test_calculator_tool_call_becomes_observation_then_finish()
    test_disabled_tool_returns_error_observation()
    test_invalid_tool_arguments_are_blocked()
    test_tool_failure_becomes_observation_not_crash()
    test_multi_tool_sequence()
    print("agent tests passed")
