from __future__ import annotations

import signal
import time

from platform_api.config import get_platform_settings
from platform_api.database import build_session_factory, initialize_development_schema
from platform_api.observability import configure_logging
from platform_api.worker_service import WorkerService


def main() -> None:
    settings = get_platform_settings()
    configure_logging(settings.log_level)
    initialize_development_schema(settings)
    worker = WorkerService(settings, build_session_factory(settings))
    stopping = False

    def stop(*_args) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    while not stopping:
        if not worker.process_once():
            time.sleep(settings.worker_poll_seconds)


if __name__ == "__main__":
    main()
