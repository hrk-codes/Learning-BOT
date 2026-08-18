from tools.calculator.tool import build_calculator_tool
from tools.factory import build_default_registry
from tools.manager import ToolManager


def test_registry_lists_default_tools() -> None:
    registry = build_default_registry()
    names = {tool.name for tool in registry.list_tools()}
    assert names == {
        "calculator.evaluate",
        "weather.get_current",
        "search.web",
        "email.send_mock",
        "files.delete_mock",
    }


def test_calculator_rejects_unsafe_expression() -> None:
    tool = build_calculator_tool()
    result = tool.run({"expression": "__import__('os').system('dir')"})
    assert result.success is False
    assert "Only numeric arithmetic expressions" in result.error


def test_manager_blocks_disabled_tool() -> None:
    registry = build_default_registry()
    manager = ToolManager(registry=registry, enabled_tools={"calculator.evaluate"})
    result = manager.execute_tool("search.web", {"query": "AI news"})
    assert result.success is False
    assert "disabled" in result.error


if __name__ == "__main__":
    test_registry_lists_default_tools()
    test_calculator_rejects_unsafe_expression()
    test_manager_blocks_disabled_tool()
    print("tool tests passed")
