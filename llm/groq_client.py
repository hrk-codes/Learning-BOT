import json
import logging
import re
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager

import requests

from config import AppConfig


logger = logging.getLogger(__name__)
_REQUEST_GATE = threading.Lock()
_GROQ_KEY_PATTERN = re.compile(r"gsk_[A-Za-z0-9]+")


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
    if model.startswith("openai/gpt-oss-"):
        # Keep reasoning useful without letting it consume the decision budget.
        payload["reasoning_effort"] = "low"

    started_at = time.perf_counter()
    logger.info(
        "Starting Groq request model=%s message_count=%s max_tokens=%s temperature=%s",
        model,
        len(messages),
        max_tokens,
        temperature,
    )

    try:
        with _serialized_rate_limited_request(
            url=config.groq_api_url,
            headers=headers,
            payload=payload,
            timeout_seconds=config.request_timeout_seconds,
            max_retries=config.rate_limit_max_retries,
            max_wait_seconds=config.rate_limit_max_wait_seconds,
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


def complete_chat_completion(
    config: AppConfig,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Collect a streamed response when the caller needs one complete value.

    The Stage 3 agent needs a structured JSON decision before it can choose
    the next runtime action. Streaming is still useful at the HTTP layer, but
    the runtime should parse only after the full JSON text has arrived.
    """
    return "".join(
        stream_chat_completion(
            config=config,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    )


def _format_http_error(response: requests.Response) -> str:
    status_code = response.status_code

    try:
        body = response.json()
    except ValueError:
        body = response.text

    error_type, error_code, error_message = _error_details(body)
    logger.error(
        "Groq HTTP error status_code=%s error_type=%s error_code=%s",
        status_code,
        error_type,
        error_code,
    )

    if status_code == 401:
        return "Groq rejected the API key. Check GROQ_API_KEY and restart the app."
    if status_code == 429:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            return f"Groq rate limit reached. Try again after {retry_after} seconds."
        return "Groq rate limit reached. Reduce request frequency or token usage and try again."
    if status_code == 400:
        return (
            "Groq rejected the request. Check the model, messages, and parameters. "
            f"Details: {error_message}"
        )

    return f"Groq returned HTTP {status_code}. Details: {error_message}"


@contextmanager
def _serialized_rate_limited_request(
    url: str,
    headers: dict[str, str],
    payload: dict,
    timeout_seconds: int,
    max_retries: int,
    max_wait_seconds: float,
) -> Generator[requests.Response, None, None]:
    # Hold the gate until streaming completes so concurrent Streamlit sessions
    # cannot consume the same organization quota at the same time.
    with _REQUEST_GATE:
        response = _post_with_rate_limit_retry(
            url=url,
            headers=headers,
            payload=payload,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            max_wait_seconds=max_wait_seconds,
        )
        try:
            yield response
        finally:
            response.close()


def _post_with_rate_limit_retry(
    url: str,
    headers: dict[str, str],
    payload: dict,
    timeout_seconds: int,
    max_retries: int,
    max_wait_seconds: float,
) -> requests.Response:
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative.")
    if max_wait_seconds < 0:
        raise ValueError("max_wait_seconds must be non-negative.")

    cumulative_wait = 0.0
    for retry_count in range(max_retries + 1):
        request_started = time.perf_counter()
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            stream=True,
            timeout=timeout_seconds,
        )
        request_elapsed = time.perf_counter() - request_started
        logger.info(
            "Groq HTTP response status_code=%s attempt=%s/%s elapsed_seconds=%.2f",
            response.status_code,
            retry_count + 1,
            max_retries + 1,
            request_elapsed,
        )

        if response.status_code != 429 or retry_count == max_retries:
            return response

        retry_after = _parse_retry_after(response.headers.get("retry-after"))
        if retry_after is None:
            logger.warning("Groq rate limit response omitted a valid retry-after value")
            return response

        if cumulative_wait + retry_after > max_wait_seconds:
            logger.warning(
                "Groq rate limit wait budget exceeded retry_after_seconds=%.2f "
                "cumulative_wait_seconds=%.2f max_wait_seconds=%.2f",
                retry_after,
                cumulative_wait,
                max_wait_seconds,
            )
            return response

        logger.warning(
            "Groq rate limited request retry=%s/%s retry_after_seconds=%.2f "
            "cumulative_wait_seconds=%.2f",
            retry_count + 1,
            max_retries,
            retry_after,
            cumulative_wait + retry_after,
        )
        response.close()
        time.sleep(retry_after)
        cumulative_wait += retry_after

    raise RuntimeError("Rate-limit retry loop ended unexpectedly.")


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def _error_details(body: object) -> tuple[str, str, str]:
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            error_type = str(error.get("type") or "unknown")
            error_code = str(error.get("code") or "unknown")
            message = str(error.get("message") or "No provider message returned.")
            message = _GROQ_KEY_PATTERN.sub("[REDACTED]", message)
            return error_type, error_code, message[:500]
    return "unknown", "unknown", "The provider returned an unreadable error response."
