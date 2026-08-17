import io
import logging
import threading
import time

import requests

from llm import groq_client


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        retry_after: str | None = None,
        body: object | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = {}
        if retry_after is not None:
            self.headers["retry-after"] = retry_after
        self.body = body or {"error": {"message": "test error"}}
        self.text = "test error"
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def json(self) -> object:
        return self.body


def test_rate_limit_retry_respects_retry_after() -> None:
    limited = FakeResponse(429, "1.5")
    success = FakeResponse(200)
    responses = iter([limited, success])
    sleeps: list[float] = []

    original_post = groq_client.requests.post
    original_sleep = groq_client.time.sleep
    try:
        groq_client.requests.post = lambda *args, **kwargs: next(responses)
        groq_client.time.sleep = sleeps.append
        result = groq_client._post_with_rate_limit_retry(
            url="https://example.invalid/chat",
            headers={"Authorization": "Bearer test"},
            payload={"stream": True},
            timeout_seconds=10,
            max_retries=2,
            max_wait_seconds=60,
        )
    finally:
        groq_client.requests.post = original_post
        groq_client.time.sleep = original_sleep

    assert result is success
    assert limited.closed is True
    assert sleeps == [1.5]


def test_rate_limit_retry_stops_after_configured_attempts() -> None:
    responses = [FakeResponse(429, "1"), FakeResponse(429, "2"), FakeResponse(429, "3")]
    response_iterator = iter(responses)
    sleeps: list[float] = []

    original_post = groq_client.requests.post
    original_sleep = groq_client.time.sleep
    try:
        groq_client.requests.post = lambda *args, **kwargs: next(response_iterator)
        groq_client.time.sleep = sleeps.append
        result = groq_client._post_with_rate_limit_retry(
            url="https://example.invalid/chat",
            headers={"Authorization": "Bearer test"},
            payload={"stream": True},
            timeout_seconds=10,
            max_retries=2,
            max_wait_seconds=60,
        )
    finally:
        groq_client.requests.post = original_post
        groq_client.time.sleep = original_sleep

    assert result is responses[2]
    assert responses[0].closed is True
    assert responses[1].closed is True
    assert responses[2].closed is False
    assert sleeps == [1.0, 2.0]


def test_rate_limit_wait_budget_is_enforced() -> None:
    limited = FakeResponse(429, "61")
    sleeps: list[float] = []

    original_post = groq_client.requests.post
    original_sleep = groq_client.time.sleep
    try:
        groq_client.requests.post = lambda *args, **kwargs: limited
        groq_client.time.sleep = sleeps.append
        result = groq_client._post_with_rate_limit_retry(
            url="https://example.invalid/chat",
            headers={},
            payload={},
            timeout_seconds=10,
            max_retries=2,
            max_wait_seconds=60,
        )
    finally:
        groq_client.requests.post = original_post
        groq_client.time.sleep = original_sleep

    assert result is limited
    assert sleeps == []


def test_invalid_retry_after_values_return_immediately() -> None:
    original_post = groq_client.requests.post
    original_sleep = groq_client.time.sleep
    try:
        for value in (None, "invalid", "-1"):
            limited = FakeResponse(429, value)
            sleeps: list[float] = []
            groq_client.requests.post = lambda *args, response=limited, **kwargs: response
            groq_client.time.sleep = sleeps.append
            result = groq_client._post_with_rate_limit_retry(
                url="https://example.invalid/chat",
                headers={},
                payload={},
                timeout_seconds=10,
                max_retries=2,
                max_wait_seconds=60,
            )
            assert result is limited
            assert sleeps == []
    finally:
        groq_client.requests.post = original_post
        groq_client.time.sleep = original_sleep


def test_non_rate_limit_responses_are_not_retried() -> None:
    original_post = groq_client.requests.post
    try:
        for status_code in (400, 401, 500):
            calls = 0
            response = FakeResponse(status_code)

            def fake_post(*args, **kwargs):
                nonlocal calls
                calls += 1
                return response

            groq_client.requests.post = fake_post
            result = groq_client._post_with_rate_limit_retry(
                url="https://example.invalid/chat",
                headers={},
                payload={},
                timeout_seconds=10,
                max_retries=2,
                max_wait_seconds=60,
            )
            assert result is response
            assert calls == 1
    finally:
        groq_client.requests.post = original_post


def test_transport_errors_are_not_retried() -> None:
    original_post = groq_client.requests.post
    try:
        for error in (requests.Timeout(), requests.ConnectionError()):
            calls = 0

            def fake_post(*args, **kwargs):
                nonlocal calls
                calls += 1
                raise error

            groq_client.requests.post = fake_post
            try:
                groq_client._post_with_rate_limit_retry(
                    url="https://example.invalid/chat",
                    headers={},
                    payload={},
                    timeout_seconds=10,
                    max_retries=2,
                    max_wait_seconds=60,
                )
            except type(error):
                pass
            else:
                raise AssertionError("Expected transport error")
            assert calls == 1
    finally:
        groq_client.requests.post = original_post


def test_request_gate_serializes_concurrent_streams() -> None:
    first_entered = threading.Event()
    release_first = threading.Event()
    post_calls: list[int] = []
    errors: list[Exception] = []

    original_post = groq_client.requests.post
    try:
        def fake_post(*args, **kwargs):
            post_calls.append(len(post_calls) + 1)
            return FakeResponse(200)

        groq_client.requests.post = fake_post

        def first_worker() -> None:
            try:
                with groq_client._serialized_rate_limited_request(
                    "https://example.invalid/chat", {}, {}, 10, 0, 0
                ):
                    first_entered.set()
                    release_first.wait(timeout=2)
            except Exception as exc:
                errors.append(exc)

        def second_worker() -> None:
            try:
                first_entered.wait(timeout=2)
                with groq_client._serialized_rate_limited_request(
                    "https://example.invalid/chat", {}, {}, 10, 0, 0
                ):
                    pass
            except Exception as exc:
                errors.append(exc)

        first_thread = threading.Thread(target=first_worker)
        second_thread = threading.Thread(target=second_worker)
        first_thread.start()
        assert first_entered.wait(timeout=2)
        second_thread.start()
        time.sleep(0.05)
        assert post_calls == [1]
        release_first.set()
        first_thread.join(timeout=2)
        second_thread.join(timeout=2)
    finally:
        release_first.set()
        groq_client.requests.post = original_post

    assert errors == []
    assert post_calls == [1, 2]


def test_error_output_and_logs_redact_api_keys() -> None:
    exposed_key = "gsk_thisMustNeverAppear"
    response = FakeResponse(
        400,
        body={
            "error": {
                "message": f"Invalid credential {exposed_key}",
                "type": "invalid_request_error",
                "code": "bad_request",
            }
        },
    )
    log_output = io.StringIO()
    handler = logging.StreamHandler(log_output)
    groq_client.logger.addHandler(handler)
    try:
        message = groq_client._format_http_error(response)
    finally:
        groq_client.logger.removeHandler(handler)

    assert exposed_key not in message
    assert exposed_key not in log_output.getvalue()
    assert "[REDACTED]" in message


if __name__ == "__main__":
    test_rate_limit_retry_respects_retry_after()
    test_rate_limit_retry_stops_after_configured_attempts()
    test_rate_limit_wait_budget_is_enforced()
    test_invalid_retry_after_values_return_immediately()
    test_non_rate_limit_responses_are_not_retried()
    test_transport_errors_are_not_retried()
    test_request_gate_serializes_concurrent_streams()
    test_error_output_and_logs_redact_api_keys()
    print("llm tests passed")
