from __future__ import annotations

import os
from pathlib import Path


os.environ["PLATFORM_ENV"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///platform_data/test-platform.db"
os.environ["PLATFORM_JWT_SECRET"] = "test-secret-that-is-long-enough-for-isolated-tests"
os.environ["AUTO_CREATE_SCHEMA"] = "true"
os.environ["EMBEDDED_WORKER"] = "false"

from fastapi.testclient import TestClient
from sqlalchemy import delete

from platform_api.database import Base, build_engine, build_session_factory
from platform_api.main import app, settings
from platform_api.models import AgentRun, AuditEvent, RunEvent, ToolConfig, User, Workspace
from platform_api.runtime_adapter import RuntimeResult
from platform_api.worker_service import WorkerService


def setup_function() -> None:
    engine = build_engine(settings.database_url)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


def register(client: TestClient, email: str) -> dict:
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "production-pass-123", "display_name": "Platform User"},
    )
    assert response.status_code == 201, response.text
    return response.json()


def auth(session: dict) -> dict[str, str]:
    return {"Authorization": f"Bearer {session['access_token']}"}


def test_auth_workspace_and_user_isolation() -> None:
    with TestClient(app) as client:
        first = register(client, "first@example.com")
        second = register(client, "second@example.com")
        created = client.post(
            "/api/v1/runs",
            headers={**auth(first), "Idempotency-Key": "first-run"},
            json={"workspace_id": first["workspace"]["id"], "goal": "Write a deployment checklist", "mode": "auto"},
        )
        assert created.status_code == 202
        hidden = client.get(f"/api/v1/runs/{created.json()['id']}", headers=auth(second))
        assert hidden.status_code == 404
        assert hidden.json()["error"]["code"] == "run_not_found"


def test_run_idempotency_and_cancel_lifecycle() -> None:
    with TestClient(app) as client:
        session = register(client, "idempotent@example.com")
        headers = {**auth(session), "Idempotency-Key": "stable-request-1"}
        payload = {"workspace_id": session["workspace"]["id"], "goal": "Prepare a concise launch brief", "mode": "write"}
        first = client.post("/api/v1/runs", headers=headers, json=payload)
        replay = client.post("/api/v1/runs", headers=headers, json=payload)
        assert first.status_code == 202
        assert replay.status_code == 200
        assert replay.json()["id"] == first.json()["id"]
        cancelled = client.post(f"/api/v1/runs/{first.json()['id']}/cancel", headers=auth(session))
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"


def test_worker_completes_claimed_run_without_live_llm() -> None:
    class FakeRuntime:
        def execute(self, _db, **kwargs):
            kwargs["on_trace_event"](
                {
                    "node": "writer",
                    "agent_name": "writer",
                    "status": "completed",
                    "next_node": "reviewer",
                    "duration_seconds": 0.01,
                    "details": {"tools_used": []},
                }
            )
            return RuntimeResult(
                answer="A bounded, reviewed result.",
                trace=[],
                metrics=[],
            )

    with TestClient(app) as client:
        session = register(client, "worker@example.com")
        created = client.post(
            "/api/v1/runs",
            headers=auth(session),
            json={"workspace_id": session["workspace"]["id"], "goal": "Build a release brief", "mode": "auto"},
        )
        run_id = created.json()["id"]
        worker = WorkerService(settings, build_session_factory(settings))
        worker.runtime = FakeRuntime()
        assert worker.process_once() is True
        completed = client.get(f"/api/v1/runs/{run_id}", headers=auth(session))
        assert completed.status_code == 200
        assert completed.json()["status"] == "completed"
        assert completed.json()["result_text"] == "A bounded, reviewed result."
        assert [event["event_type"] for event in completed.json()["events"]] == [
            "run.queued",
            "run.started",
            "agent.node",
            "run.completed",
        ]


def test_memory_and_tool_settings_are_workspace_scoped() -> None:
    with TestClient(app) as client:
        session = register(client, "resources@example.com")
        workspace = session["workspace"]["id"]
        memory = client.post(
            "/api/v1/memories",
            headers=auth(session),
            json={"workspace_id": workspace, "memory_type": "preference", "content": "Prefer concise release notes."},
        )
        assert memory.status_code == 201
        tools = client.get(f"/api/v1/tools?workspace_id={workspace}", headers=auth(session))
        assert tools.status_code == 200
        assert any(item["enabled"] for item in tools.json())
        tool_name = tools.json()[0]["name"]
        changed = client.patch(
            f"/api/v1/tools/{tool_name}?workspace_id={workspace}",
            headers=auth(session),
            json={"enabled": False},
        )
        assert changed.status_code == 200
        assert changed.json()["enabled"] is False


def test_errors_use_stable_envelope() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/runs/not-real", headers={"Authorization": "Bearer invalid"})
        assert response.status_code == 401
        assert set(response.json()["error"]) >= {"code", "message", "request_id"}
