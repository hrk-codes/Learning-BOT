from typing import Any

import requests

from tools.base import ToolDefinition, ToolResult


def build_weather_tool() -> ToolDefinition:
    return ToolDefinition(
        name="weather.get_current",
        description="Get current weather for a city or place. Use only for present weather, not forecasts or historical weather.",
        input_schema={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City/place name, for example Delhi or Mumbai."}
            },
            "required": ["location"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "temperature_c": {"type": "number"},
                "wind_speed_kmh": {"type": "number"},
            },
        },
        permission="read_only_external",
        timeout_seconds=8,
        version="1.0",
        execute=_execute,
    )


def _execute(arguments: dict[str, Any]) -> ToolResult:
    location = arguments["location"]
    geo_response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": location, "count": 1, "language": "en", "format": "json"},
        timeout=8,
    )
    geo_response.raise_for_status()
    geo_data = geo_response.json()
    results = geo_data.get("results") or []
    if not results:
        return ToolResult(success=False, error=f"No weather location found for {location}.")

    place = results[0]
    weather_response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        },
        timeout=8,
    )
    weather_response.raise_for_status()
    current = weather_response.json().get("current", {})

    return ToolResult(
        success=True,
        result={
            "location": f"{place.get('name')}, {place.get('country')}",
            "temperature_c": current.get("temperature_2m"),
            "humidity_percent": current.get("relative_humidity_2m"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
        },
        metadata={"provider": "open-meteo", "permission": "read_only_external"},
    )
