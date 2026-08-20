# Stage 10: Manager-Led Multi-Agent System

## Purpose

Stage 9 made one complex workflow explicit. Stage 10 answers a different question: when is one
general-purpose decision-maker carrying too many independent responsibilities?

The answer is not "whenever we can make more LLM calls." A multi-agent system is justified
only when specialization provides enough quality, clarity, or safety value to pay for extra
latency, tokens, coordination, and failure modes.

This first implementation is a small, sequential team:

```text
USER -> MANAGER -> RESEARCHER -> MANAGER -> WRITER -> MANAGER -> REVIEWER
                                                               ^              |
                                                               +--- revise ----+
```

The manager is the only routing authority. Specialists do not freely chat with each other,
and no specialist silently takes over another specialist's responsibility.

## Roles And Contracts

| Role | Owns | Does not own |
|---|---|---|
| Manager | Delegation, limits, recovery choice, final synthesis | Raw RAG retrieval, tool execution, draft review |
| Researcher | Evidence gathering, RAG, approved read-only search, confidence/gaps | Final polished answer, side effects |
| Writer | Drafting from structured research and selected style preferences | Tools, raw RAG, universal memory access |
| Reviewer | Independent pass/revise/research-needed decision | Rewriting, tool execution, routing |

The important code is in [contracts.py](../multi_agent/agents/contracts.py) and
[results.py](../multi_agent/agents/results.py). Every delegated task has an ID, assigned role,
goal, expected output, constraints, scoped tool list, and RAG permission. Every specialist
returns an `AgentResult` with task ID, status, duration, retries, sources, output, and a safe
error when needed.

Research is validated so every claim identifies one or more declared `source_id` values. A
claim cannot cite a source that was never supplied. The writer receives that normalized artifact
instead of raw vector records; the reviewer receives the draft plus the same evidence and
returns a typed review status.

If JSON or schema validation fails, an agent gets one bounded repair prompt. It never silently
continues with arbitrary prose. A remaining failure becomes a failed `AgentResult`; the manager
can retry within its limit or finish with an explicit limitation.

## Context Boundaries

The graph state in [state.py](../multi_agent/state.py) stores compact workflow artifacts, not a
global prompt dump.

```text
Manager:    goal + compact result statuses + limits
Researcher: delegated task + selected RAG/search evidence
Writer:     delegated task + structured research + profile/procedural style memory
Reviewer:   delegated task + draft + structured research + criteria
```

Conversation history, unselected memory records, raw documents, tool internals, and hidden
reasoning do not automatically cross those boundaries. The writer receives only `profile` and
`procedural` memory records because those can affect requested style.

## Existing Capability Reuse

```text
Stage 5 RAG       -> Researcher calls the existing RagPipeline
Stage 4 tools     -> Researcher can receive only active search.web access
Stage 6 memory    -> Writer receives a selected memory subset from the existing service
Stage 8 approval  -> No Stage 10 specialist gets a side-effecting tool path
Stage 9 LangGraph -> One new graph coordinates this team with SQLite checkpoints
```

Stage 8 remains authoritative for consequential actions. This team does not expose mock email,
file deletion, or other side-effecting tools. Adding one later must still pass action proposal,
risk policy, human approval, version recheck, and idempotent execution.

## Topology And Review Loop

[graph.py](../multi_agent/graph.py) is intentionally one `StateGraph`, not nested subgraphs:

```text
START -> manager
manager -> researcher | writer | reviewer | finalize
researcher -> manager
writer -> manager
reviewer -> manager
finalize -> END
```

The manager's only valid actions are `delegate_research`, `delegate_writing`,
`delegate_review`, `revise`, and `finish`. A reviewer returns `approved`,
`revision_required`, or `research_required`; the manager chooses a writer revision, additional
research, or completion. Review revisions and total delegations are bounded, preventing loops
such as manager -> researcher -> manager forever.

## Static, Dynamic, And Parallel Work

V1 is mostly structured/static: research before writing, writing before review. It is dynamic
where useful: simple goals skip specialists, research-only goals skip writer/reviewer, feedback
can request a revision or new evidence, and failures get a limited retry.

Independent research tasks could later fan out in parallel. That can reduce latency from
`A + B + C` toward `max(A, B, C)`, but it also raises provider concurrency, rate-limit pressure,
cost, artifact merge complexity, and disagreement handling. Sequential V1 keeps the learning
flow inspectable.

## Limits And Observability

`.env.example` exposes the operational boundaries:

```text
MULTI_AGENT_MAX_DELEGATIONS=8
MULTI_AGENT_MAX_AGENT_RETRIES=1
MULTI_AGENT_MAX_REVIEW_REVISIONS=1
MULTI_AGENT_TIMEOUT_SECONDS=60
MULTI_AGENT_OUTPUT_REPAIR_ATTEMPTS=1
```

Each specialist call gets the configured HTTP timeout. The UI trace shows only safe metadata:
run ID, node, agent, task, duration, status, RAG/tool usage, and retries. It does not expose
prompts, API keys, raw private document text, or hidden chain-of-thought.

## Verification Guide

1. Run `streamlit run app.py` and enable **Use manager-led multi-agent workflow**.
2. Ask `Explain JSON in two sentences.` Expected trace: `manager -> finalize` with no specialist calls.
3. Index a PDF, then ask `Research the deployment options from my indexed document.` Expected:
   `manager -> researcher -> manager -> finalize` and `rag_used: true` in researcher metadata.
4. Ask `Research the deployment options, write a concise report, and verify it carefully.`
   Expected: researcher, writer, reviewer, manager, then finalizer. A revision can follow a
   non-approved review.
5. Open **Stage 10 Team Execution** and inspect the route, task IDs, status, duration, and
   RAG/tool metadata.
6. Disable the Stage 10 toggle and confirm the direct/Stage 9 workflow still works.
7. Run tests:

```powershell
.venv\Scripts\python.exe -m pytest -q test_multi_agent.py
.venv\Scripts\python.exe -m pytest -q
```

## When Not To Use It

Use one well-designed agent for low-risk, simple, latency-sensitive work where research,
writing, and independent review are not distinct valuable jobs. A poorly designed multi-agent
team can be slower, more expensive, and less reliable than one constrained agent. Future steps
may add per-role models, parallel researchers, evaluation metrics, conflict resolution, and
hierarchical subgraphs only when measurements prove they solve a real problem.
