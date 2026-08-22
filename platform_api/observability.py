from __future__ import annotations

import json
import hashlib
import logging
import threading
import time
import uuid
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from platform_api.config import get_platform_settings
from platform_api.errors import ApiError, error_payload


logger = logging.getLogger("platform.request")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request.state.request_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex}"
        started = time.perf_counter()
        response = None
        if request.url.path.startswith("/api/"):
            authorization = request.headers.get("authorization", "")
            actor = (
                hashlib.sha256(authorization.encode("utf-8")).hexdigest()[:20]
                if authorization
                else (request.client.host if request.client else "unknown")
            )
            try:
                rate_limiter.check(
                    f"api:{actor}", get_platform_settings().request_rate_limit
                )
            except ApiError as exc:
                response = JSONResponse(
                    status_code=exc.status_code,
                    content=error_payload(request, exc.code, exc.message, exc.details),
                )
        if response is None:
            response = await call_next(request)
        elapsed = time.perf_counter() - started
        response.headers["x-request-id"] = request.state.request_id
        response.headers["x-content-type-options"] = "nosniff"
        response.headers["x-frame-options"] = "DENY"
        response.headers["referrer-policy"] = "no-referrer"
        logger.info(
            json.dumps(
                {
                    "event": "http_request",
                    "request_id": request.state.request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "elapsed_ms": round(elapsed * 1000, 2),
                }
            )
        )
        return response


class InMemoryRateLimiter:
    """Single-process safety gate. Production can replace this with Redis unchanged at the router boundary."""

    def __init__(self) -> None:
        self._entries: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, limit: int, window_seconds: int = 60) -> None:
        now = time.monotonic()
        with self._lock:
            entries = self._entries[key]
            while entries and entries[0] <= now - window_seconds:
                entries.popleft()
            if len(entries) >= limit:
                retry_after = max(1, int(window_seconds - (now - entries[0])))
                raise ApiError(
                    429,
                    "rate_limit_exceeded",
                    f"Too many requests. Retry in approximately {retry_after} seconds.",
                    {"retry_after_seconds": retry_after},
                )
            entries.append(now)


rate_limiter = InMemoryRateLimiter()
