import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

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

    def to_model_description(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "permission": self.permission,
            "version": self.version,
        }

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        # Never trust model-generated arguments directly. The tool contract is
        # the runtime boundary that keeps "LLM requested it" separate from
        # "runtime authorized and executed it."
        try:
            clean_arguments = validate_object_schema(arguments, self.input_schema)
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
