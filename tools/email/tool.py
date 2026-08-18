import uuid
from typing import Any

from approval.models import RiskLevel, SideEffectType
from tools.base import ToolDefinition, ToolResult


def build_mock_email_tool() -> ToolDefinition:
    return ToolDefinition(
        name="email.send_mock",
        description=(
            "Send an email through the Stage 8 simulated outbox. This is a safe learning "
            "tool and does not contact a real recipient."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address."},
                "subject": {"type": "string", "description": "Email subject."},
                "body": {"type": "string", "description": "Complete email body."},
            },
            "required": ["to", "subject", "body"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "external_id": {"type": "string"},
                "delivery": {"type": "string"},
            },
        },
        permission="side_effecting",
        timeout_seconds=2,
        version="1.0",
        execute=_execute,
        risk_level=RiskLevel.HIGH,
        side_effect=SideEffectType.EXTERNAL_COMMUNICATION,
        supports_preview=True,
        requires_confirmation=True,
        preview_builder=_preview,
    )


def _preview(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": "Send simulated email",
        "impact": (
            "This creates an external-communication receipt in the safe mock outbox. "
            "No real email will be delivered."
        ),
        "fields": {
            "To": arguments["to"],
            "Subject": arguments["subject"],
            "Body": arguments["body"],
        },
    }


def _execute(arguments: dict[str, Any]) -> ToolResult:
    external_id = f"mock_msg_{uuid.uuid4().hex[:12]}"
    return ToolResult(
        success=True,
        result={
            "external_id": external_id,
            "delivery": "simulated",
            "to": arguments["to"],
            "subject": arguments["subject"],
        },
        metadata={"mock": True, "external_id": external_id},
    )
