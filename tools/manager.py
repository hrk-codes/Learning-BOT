from approval.models import ActionProposal, SideEffectType
from tools.base import Permission, ToolDefinition, ToolResult
from tools.registry import ToolRegistry


class ToolManager:
    def __init__(
        self,
        registry: ToolRegistry,
        enabled_tools: set[str],
        authorized_permissions: set[Permission] | None = None,
    ) -> None:
        self.registry = registry
        self.enabled_tools = enabled_tools
        self.authorized_permissions = authorized_permissions or {
            "safe",
            "read_only_external",
        }

    def list_available_tools(self) -> list[ToolDefinition]:
        return self.registry.list_tools()

    def list_active_tools(self) -> list[ToolDefinition]:
        return [
            tool
            for tool in self.registry.list_tools()
            if tool.name in self.enabled_tools
            and tool.permission in self.authorized_permissions
        ]

    def get_active_tool_descriptions(self) -> list[dict]:
        return [tool.to_model_description() for tool in self.list_active_tools()]

    def execute_tool(
        self,
        name: str,
        arguments: dict,
        *,
        approved_action: ActionProposal | None = None,
    ) -> ToolResult:
        tool = self.registry.get_tool(name)
        if tool is None:
            return ToolResult(success=False, error=f"Unknown tool requested: {name}")
        if name not in self.enabled_tools:
            return ToolResult(success=False, error=f"Tool is disabled for this session: {name}")
        permission_error = self._check_permission(tool.permission)
        if permission_error:
            return ToolResult(success=False, error=permission_error, metadata={"tool": name})

        requires_bound_approval = (
            tool.requires_confirmation
            or tool.permission == "side_effecting"
            or tool.side_effect
            not in {SideEffectType.NONE, SideEffectType.READ_ONLY}
        )
        if requires_bound_approval:
            if approved_action is None:
                return ToolResult(
                    success=False,
                    error=(
                        f"Tool {name} requires user confirmation for this specific action."
                    ),
                    metadata={"tool": name, "approval_required": True},
                )
            try:
                matches = tool.matches_approved_action(approved_action, arguments)
            except Exception as exc:
                return ToolResult(
                    success=False,
                    error=f"Approved action validation failed: {exc}",
                    metadata={"tool": name, "approval_mismatch": True},
                )
            if not matches:
                return ToolResult(
                    success=False,
                    error="The action no longer matches the exact version the user approved.",
                    metadata={"tool": name, "approval_mismatch": True},
                )

        return tool.run(arguments)

    def get_active_tool(self, name: str) -> ToolDefinition | None:
        return next((tool for tool in self.list_active_tools() if tool.name == name), None)

    def _check_permission(self, permission: Permission) -> str | None:
        if permission in self.authorized_permissions:
            return None
        # Approval is not permission. A reviewed action still cannot execute if
        # this user/session is not authorized for the tool's permission class.
        return (
            f"Permission {permission!r} requires user confirmation and is not "
            "authorized for this session."
        )
