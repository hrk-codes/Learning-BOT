from tools.base import Permission, ToolDefinition, ToolResult
from tools.registry import ToolRegistry


class ToolManager:
    def __init__(self, registry: ToolRegistry, enabled_tools: set[str]) -> None:
        self.registry = registry
        self.enabled_tools = enabled_tools

    def list_available_tools(self) -> list[ToolDefinition]:
        return self.registry.list_tools()

    def list_active_tools(self) -> list[ToolDefinition]:
        return [tool for tool in self.registry.list_tools() if tool.name in self.enabled_tools]

    def get_active_tool_descriptions(self) -> list[dict]:
        return [tool.to_model_description() for tool in self.list_active_tools()]

    def execute_tool(self, name: str, arguments: dict) -> ToolResult:
        tool = self.registry.get_tool(name)
        if tool is None:
            return ToolResult(success=False, error=f"Unknown tool requested: {name}")
        if name not in self.enabled_tools:
            return ToolResult(success=False, error=f"Tool is disabled for this session: {name}")
        permission_error = self._check_permission(tool.permission)
        if permission_error:
            return ToolResult(success=False, error=permission_error, metadata={"tool": name})

        return tool.run(arguments)

    def _check_permission(self, permission: Permission) -> str | None:
        if permission in {"safe", "read_only_external"}:
            return None
        # The runtime models permissions but does not yet implement approval
        # workflows. Side-effecting tools remain blocked until a later stage can
        # ask the user for explicit confirmation.
        return f"Permission {permission!r} requires user confirmation, which is not enabled yet."
