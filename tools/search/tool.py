from typing import Any

import requests

from tools.base import ToolDefinition, ToolResult


def build_search_tool() -> ToolDefinition:
    return ToolDefinition(
        name="search.web",
        description="Search the web for current or factual information. Returns a small result summary, not full webpages.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query."}},
            "required": ["query"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}, "results": {"type": "string"}},
        },
        permission="read_only_external",
        timeout_seconds=8,
        version="1.0",
        execute=_execute,
    )


def _execute(arguments: dict[str, Any]) -> ToolResult:
    query = arguments["query"]
    response = requests.get(
        "https://api.duckduckgo.com/",
        params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
        timeout=8,
    )
    response.raise_for_status()
    data = response.json()

    results: list[dict[str, str]] = []
    if data.get("AbstractText"):
        results.append({"title": data.get("Heading", "DuckDuckGo result"), "snippet": data["AbstractText"][:500]})

    for topic in data.get("RelatedTopics", [])[:5]:
        if isinstance(topic, dict) and topic.get("Text"):
            results.append({"title": topic.get("FirstURL", "Related result"), "snippet": topic["Text"][:300]})

    if not results:
        return ToolResult(success=False, error=f"No concise search results found for {query}.")

    return ToolResult(
        success=True,
        result={"query": query, "results": results[:3]},
        metadata={"provider": "duckduckgo-instant-answer", "permission": "read_only_external"},
    )
