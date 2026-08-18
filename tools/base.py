import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from approval.models import (
    ActionProposal,
    ActionStatus,
    RiskLevel,
    SideEffectType,
    calculate_action_digest,
)
from tools.schemas import SchemaValidationError, validate_object_schema


Permission = Literal["safe", "read_only_external", "side_effecting"]


@dataclass(frozen=True)
class ToolResult:
    success: bool
    result: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    permission: Permission
    timeout_seconds: int
    version: str
    execute: Callable[[dict[str, Any]], ToolResult]
    risk_level: RiskLevel = RiskLevel.LOW
    side_effect: SideEffectType = SideEffectType.NONE
    supports_preview: bool = False
    requires_confirmation: bool = False
    preview_builder: Callable[[dict[str, Any]], dict[str, Any]] | None = None

    def to_model_description(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "permission": self.permission,
            "version": self.version,
            "risk_level": self.risk_level.value,
            "side_effect": self.side_effect.value,
            "supports_preview": self.supports_preview,
            "requires_confirmation": self.requires_confirmation,
        }

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return validate_object_schema(arguments, self.input_schema)

    def build_preview(self, arguments: dict[str, Any]) -> dict[str, Any]:
        clean_arguments = self.validate_arguments(arguments)
        if self.preview_builder is not None:
            return self.preview_builder(clean_arguments)
        return {
            "title": self.description,
            "impact": f"Run {self.name} with the displayed arguments.",
            "arguments": clean_arguments,
        }

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        # Never trust model-generated arguments directly. The tool contract is
        # the runtime boundary that keeps "LLM requested it" separate from
        # "runtime authorized and executed it."
        try:
            clean_arguments = self.validate_arguments(arguments)
        except SchemaValidationError as exc:
            return ToolResult(
                success=False,
                error=f"Invalid arguments for {self.name}: {exc}",
                metadata={"tool": self.name, "validation_failed": True},
            )

        started_at = time.perf_counter()
        try:
            result = self.execute(clean_arguments)
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"{self.name} failed: {exc}",
                metadata={"tool": self.name, "exception": type(exc).__name__},
                elapsed_seconds=time.perf_counter() - started_at,
            )

        return ToolResult(
            success=result.success,
            result=result.result,
            error=result.error,
            metadata={**result.metadata, "tool": self.name},
            elapsed_seconds=time.perf_counter() - started_at,
        )

    def matches_approved_action(
        self, proposal: ActionProposal, arguments: dict[str, Any]
    ) -> bool:
        if proposal.status not in {ActionStatus.APPROVED, ActionStatus.EXECUTING}:
            return False
        if proposal.tool_name != self.name or proposal.tool_version != self.version:
            return False
        clean_arguments = self.validate_arguments(arguments)
        digest = calculate_action_digest(
            self.name, self.version, proposal.version, clean_arguments
        )
        return digest == proposal.argument_digest and clean_arguments == proposal.arguments
