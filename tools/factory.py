from tools.calculator.tool import build_calculator_tool
from tools.registry import ToolRegistry
from tools.search.tool import build_search_tool
from tools.weather.tool import build_weather_tool


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_tool(build_calculator_tool())
    registry.register_tool(build_weather_tool())
    registry.register_tool(build_search_tool())
    return registry
