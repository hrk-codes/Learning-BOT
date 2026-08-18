import ast
import operator
from typing import Any

from approval.models import RiskLevel, SideEffectType
from tools.base import ToolDefinition, ToolResult


ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def build_calculator_tool() -> ToolDefinition:
    return ToolDefinition(
        name="calculator.evaluate",
        description="Evaluate a safe arithmetic expression. Use for math calculations, conversions, and numeric expressions.",
        input_schema={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Arithmetic expression such as 12345 * 678."}
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"result": {"type": "number"}, "expression": {"type": "string"}},
        },
        permission="safe",
        timeout_seconds=2,
        version="1.0",
        execute=_execute,
        risk_level=RiskLevel.LOW,
        side_effect=SideEffectType.NONE,
    )


def _execute(arguments: dict[str, Any]) -> ToolResult:
    expression = arguments["expression"]
    result = _safe_eval(expression)
    return ToolResult(
        success=True,
        result={"expression": expression, "result": result},
        metadata={"permission": "safe"},
    )


def _safe_eval(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:
        return ALLOWED_OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_OPERATORS:
        return ALLOWED_OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError("Only numeric arithmetic expressions are allowed.")
