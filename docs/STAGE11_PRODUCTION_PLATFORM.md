# Stage 11: Production AI Agent Platform

Stage 11 does not make the model more intelligent. It makes the Stage 10 intelligence
operable by multiple authenticated users through explicit service boundaries, durable
state, background execution, and observable lifecycle contracts.

## Architecture

```text
Browser / React operations workspace
        |
        | HTTPS + JSON + bearer identity
        v
FastAPI /api/v1
  |-- authentication + authorization
  |-- validation + idempotency + rate limits
  |-- run/document lifecycle APIs
  |-- SSE progress + health + metrics
        |
        v
PostgreSQL durable application state
        |
        | oldest queued work, transactional claim
        v
Background worker
  |-- Stage 10 MultiAgentRuntime adapter
  |-- user-scoped memory, RAG, and tool configuration
  |-- bounded model calls and execution trace
        |
        +--> Groq API
        +--> object storage abstraction (local volume in development)
```

The API returns `202 Accepted` only after the run is committed. The request does not
remain open while Manager, Researcher, Writer, and Reviewer execute. A worker claims
queued work, and the UI observes progress through the run record and an authenticated
SSE stream.

## Why These Choices

### FastAPI

FastAPI provides typed request validation, dependency-based identity, generated local
OpenAPI documentation, and native streaming responses without hiding HTTP fundamentals.
The security flow follows the official OAuth2/JWT pattern, with Argon2 password hashes
through `pwdlib` and short-lived bearer tokens.

### PostgreSQL and SQLAlchemy

PostgreSQL owns users, workspaces, runs, events, usage, document metadata, memories,
approvals, tool settings, and audit events. SQLAlchemy sessions define transaction
boundaries. SQLite remains a zero-cost local-development adapter; production mode rejects
a non-PostgreSQL URL.

### Database-backed queue before Redis

Stage 11 already requires PostgreSQL. A transactionally claimed queue is enough for the
current workload and keeps the first production slice understandable. Redis would become
preferable when queue throughput, delayed scheduling, distributed rate limiting, or
worker fan-out measurements prove PostgreSQL polling is the bottleneck. It is not added
merely because production stacks often contain it.

### Object storage abstraction

The database stores document metadata and an object key, not PDF bytes. Local development
maps object keys to `platform_data/objects`; the same boundary can later target S3 or an
S3-compatible service without changing the document API.

## Security Boundaries

- Passwords are Argon2 hashes; plaintext passwords are never stored.
- JWT identity comes from `sub`; endpoint bodies never choose a `user_id`.
- Every workspace, run, memory, document, tool setting, and approval query includes the
  authenticated owner.
- CORS uses configured origins rather than `*`.
- Requests return a request ID and stable error envelope.
- Logs contain routes, status, timing, model usage, and identifiers, but not API keys,
  prompts, memories, document content, or generated answers.
- Side-effecting tools remain excluded from unattended worker permissions. Approval is a
  one-action authority boundary, not permission to enable a tool globally.

## Run Lifecycle

```text
queued -> running -> completed
                  -> failed
queued/running -> cancelled
running -> waiting_for_approval -> running (extension boundary)
```

Each transition creates a `run_events` record. Final outputs and public trace metadata
are durable. Provider call metrics are stored separately in `usage_records` so reliability
and cost can be measured without retaining prompt content.

## Local Development

Terminal 1:

```powershell
.\.venv\Scripts\python.exe -m uvicorn platform_api.main:app --reload --port 8000
```

Terminal 2:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The API documentation is available only outside production
at `http://localhost:8000/api/docs`.

## Docker Topology

```powershell
docker compose up --build
```

Open `http://localhost:8080`. Compose runs PostgreSQL, migration-first FastAPI, a separate
worker, and the Nginx-served frontend. Replace `PLATFORM_JWT_SECRET` and database
credentials before any public deployment.

## Verification Walkthrough

1. Create an operator account and confirm the workspace loads.
2. Launch a mission. The first API response should be a queued run, not an LLM answer.
3. Open **Runs** and watch `queued -> running -> completed` plus manager/specialist events.
4. Refresh the browser. The run and result should remain because they live in the database.
5. Add a preference in **Memory**, launch a writing mission, and inspect the resulting run.
6. Upload a PDF in **Knowledge** and watch `queued -> processing -> indexed`.
7. Disable a capability in **Toolbox** and confirm future workers do not expose it.
8. Inspect **Approvals** to verify side effects are never silently authorized.
9. Inspect **Insights**, `/health/ready`, and `/metrics` for operational state.
10. Register a second account and verify it cannot retrieve the first account's run ID.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
cd frontend
npm run build
```

The Stage 11 tests cover identity, cross-user isolation, idempotency, cancellation,
worker completion with a fake runtime, resource scoping, and consistent errors. Existing
Stage 1-10 tests remain the regression suite for the intelligence layer.

## Production Extensions

- Managed PostgreSQL backups and restore drills.
- S3-compatible object storage with signed upload/download URLs.
- Redis only after measured need for distributed rate limits or queue throughput.
- OpenTelemetry exporter and a hosted metrics/log backend.
- Email verification, password reset, token revocation, and external identity providers.
- A durable approval resume path for side-effecting multi-agent tasks.
- CI deployment gates, container scanning, secret manager integration, and rollback drills.

## Primary References

- [FastAPI OAuth2 with JWT](https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/)
- [SQLAlchemy Session basics](https://docs.sqlalchemy.org/en/20/orm/session_basics.html)
- [Vite guide](https://vite.dev/guide/)
- [Docker Compose startup order](https://docs.docker.com/compose/how-tos/startup-order/)
