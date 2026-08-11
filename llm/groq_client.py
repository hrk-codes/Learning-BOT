import json
import logging
import time
from collections.abc import Generator

import requests

from config import AppConfig


logger = logging.getLogger(__name__)


class GroqClientError(Exception):
    """Raised when Groq cannot return a usable assistant response."""


def stream_chat_completion(
    config: AppConfig,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> Generator[str, None, None]:
    if not config.groq_api_key:
        raise GroqClientError(
            "Missing GROQ_API_KEY. Create a .env file or set the environment variable before starting the app."
        )

    headers = {
        "Authorization": f"Bearer {config.groq_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

    started_at = time.perf_counter()
    logger.info(
        "Starting Groq request model=%s message_count=%s max_tokens=%s temperature=%s",
        model,
        len(messages),
        max_tokens,
        temperature,
    )

    try:
        with requests.post(
            config.groq_api_url,
            headers=headers,
            json=payload,
            stream=True,
            timeout=config.request_timeout_seconds,
        ) as response:
            if response.status_code >= 400:
                raise GroqClientError(_format_http_error(response))

            received_text = False

            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue

                # Groq streams OpenAI-compatible Server-Sent Events: "data: {...}".
                if not raw_line.startswith("data: "):
                    continue

                data = raw_line.removeprefix("data: ").strip()
                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise GroqClientError("Groq returned a streaming chunk that was not valid JSON.") from exc

                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                if content:
                    received_text = True
                    yield content

            if not received_text:
                raise GroqClientError("Groq completed the request but did not return any text.")

            elapsed = time.perf_counter() - started_at
            logger.info("Groq request succeeded elapsed_seconds=%.2f", elapsed)

    except requests.Timeout as exc:
        logger.exception("Groq request timed out")
        raise GroqClientError("The Groq request timed out. Try again or lower max tokens.") from exc
    except requests.ConnectionError as exc:
        logger.exception("Groq connection failed")
        raise GroqClientError("Could not connect to Groq. Check your internet connection and try again.") from exc
    except requests.RequestException as exc:
        logger.exception("Groq request failed")
        raise GroqClientError("The Groq request failed before a response was completed.") from exc


def _format_http_error(response: requests.Response) -> str:
    status_code = response.status_code

    try:
        body = response.json()
    except ValueError:
        body = response.text

    logger.error("Groq HTTP error status_code=%s body=%s", status_code, body)

    if status_code == 401:
        return "Groq rejected the API key. Check GROQ_API_KEY and restart the app."
    if status_code == 429:
        return "Groq rate limit reached. Wait a moment and try again."
    if status_code == 400:
        return f"Groq rejected the request. Check the model, messages, and parameters. Details: {body}"

    return f"Groq returned HTTP {status_code}. Details: {body}"
