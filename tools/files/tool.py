from typing import Any

from approval.models import RiskLevel, SideEffectType
from tools.base import ToolDefinition, ToolResult


def build_mock_file_delete_tool() -> ToolDefinition:
    return ToolDefinition(
        name="files.delete_mock",
        description=(
            "Simulate permanent file deletion for approval testing. It never deletes "
            "real files."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "Exact mock paths proposed for deletion.",
                }
            },
            "required": ["paths"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"simulated_deleted": {"type": "array"}},
        },
        permission="side_effecting",
        timeout_seconds=2,
        version="1.0",
        execute=_execute,
        risk_level=RiskLevel.HIGH,
        side_effect=SideEffectType.DESTRUCTIVE,
        supports_preview=True,
        requires_confirmation=True,
        preview_builder=_preview,
    )


def _preview(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": "Simulate permanent file deletion",
        "impact": (
            "The tool models a destructive action but Stage 8 will not remove real files."
        ),
        "fields": {
            "Action": "Permanent deletion simulation",
            "Files": arguments["paths"],
            "Count": len(arguments["paths"]),
        },
    }


def _execute(arguments: dict[str, Any]) -> ToolResult:
    return ToolResult(
        success=True,
        result={"simulated_deleted": arguments["paths"], "real_files_deleted": False},
        metadata={"mock": True},
    )
