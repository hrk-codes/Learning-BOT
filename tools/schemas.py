from typing import Any


class SchemaValidationError(Exception):
    """Raised when tool arguments do not match the declared input contract."""


def validate_object_schema(arguments: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    if schema.get("type") != "object":
        raise SchemaValidationError("Only object schemas are supported in Stage 4.")
    if not isinstance(arguments, dict):
        raise SchemaValidationError("arguments must be a JSON object.")

    properties = schema.get("properties", {})
    required = schema.get("required", [])
    additional_allowed = schema.get("additionalProperties", False)

    for field in required:
        if field not in arguments:
            raise SchemaValidationError(f"missing required field: {field}")

    clean: dict[str, Any] = {}
    for key, value in arguments.items():
        if key not in properties:
            if additional_allowed:
                clean[key] = value
                continue
            raise SchemaValidationError(f"unexpected field: {key}")

        clean[key] = _validate_value(key, value, properties[key])

    return clean


def _validate_value(key: str, value: Any, field_schema: dict[str, Any]) -> Any:
    expected_type = field_schema.get("type")
    if expected_type == "string":
        if not isinstance(value, str) or not value.strip():
            raise SchemaValidationError(f"{key} must be a non-empty string")
        return value.strip()
    if expected_type == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise SchemaValidationError(f"{key} must be a number")
        return value
    if expected_type == "boolean":
        if not isinstance(value, bool):
            raise SchemaValidationError(f"{key} must be true or false")
        return value
    if expected_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise SchemaValidationError(f"{key} must be an integer")
        return value

    raise SchemaValidationError(f"{key} uses unsupported schema type: {expected_type}")
