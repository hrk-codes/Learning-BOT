# Stage 9: LangGraph Stateful Orchestration

Stage 9 introduces LangGraph as the orchestration layer around the capabilities built in
Stages 1 through 8. It does not replace Groq calls, prompts, RAG, long-term memory, tools,
planning, execution, or approval policy. It makes the transitions between them explicit.

## Why This Stage Exists

The custom Stage 7 and Stage 8 runtime already expressed a real workflow:

```text
while plan is running:
  schedule ready work
  execute a task
  retry transient failure
  pause for approval
  evaluate the goal
  replan or finish
```

That Python was not wrong. It taught the underlying control problem. As branches, loops,
retries, durable pauses, and recovery paths grow, however, the workflow topology becomes
hard to see inside one procedural loop. LangGraph represents the same control flow as a
stateful graph.

```text
START
  -> planner
  -> task_router
  -> execute_task
       -> approval interrupt
       -> retry_task -> execute_task
       -> task_router
  -> evaluate
       -> replan -> task_router
       -> finalize -> END
```

## Mapping The Existing System

| Existing custom architecture | LangGraph abstraction | Stage 9 implementation |
|---|---|---|
| `PlanState` plus workflow facts | Graph state | `graph/state.py` `GraphAgentState` |
| Planner, scheduler, executor, evaluator | Nodes | `graph/nodes.py` adapters |
| `if` / task-status checks | Conditional edges | `graph/routing.py` |
| Calling the next subsystem | Edge | `graph/graph.py` |
| Retry/replan loops | Cyclic graph paths | `retry_task -> execute_task`, `replan -> task_router` |
| Stage 8 pause/resume | Dynamic interrupt | `approval` node uses `interrupt()` |
| Approval SQLite plan persistence | Graph checkpointing | LangGraph `SqliteSaver` |
| One workflow execution | Graph run / thread | `run_id` and stable `thread_id` |

LangGraph is a coordinator. `MemoryService`, `RagPipeline`, `ToolManager`, `ApprovalService`,
`Planner`, and `TaskExecutor` remain their own services with their existing contracts.

## Graph, Node, State, And Edge

A graph is connected execution: nodes do work and edges describe where control goes next.

```text
Node = station where work happens
Edge = road to the next station
State = the shared, serializable travel record
```

A node is not necessarily an LLM call. The Stage 9 nodes are coherent orchestration units:

| Node | Purpose | Reads | Writes | Failure / next path |
|---|---|---|---|---|
| `planner` | Create and validate a bounded task DAG | goal, selected memory, knowledge-base description | serialized plan | safe failure -> finalize |
| `task_router` | Ask the Stage 7 scheduler what can run | plan state | next task ID | execute or evaluate |
| `execute_task` | Run one existing task | plan, task ID | result, task/plan updates | approval, retry, or router |
| `retry_task` | Requeue a retryable safe task | last result, retry budget | task ready state | execute task |
| `approval` | Pause for a Stage 8 action decision | action/approval IDs | approval metadata | interrupt or execute |
| `evaluate` | Check whether the user goal is satisfied | plan outputs | evaluation | replan or finalize |
| `replan` | Add a validated replacement plan revision | failed/evaluated plan | revised plan | router |
| `finalize` | Produce the terminal lifecycle result | plan/evaluation | final answer and status | END |

The topology is defined once in `graph/graph.py`; a particular run is a separate instance
with its own state and thread ID.

## State Discipline

Graph state answers: **what is happening during this one workflow?** It contains the goal,
small selected conversation/memory context, serialized `PlanState`, current task ID, result
facts, approval reference, trace metadata, and final result.

It does not contain database connections, live tool functions, model clients, whole PDFs,
the vector store, complete memory database, API keys, prompts, or chain-of-thought. Those
are runtime services or external stores. The graph state keeps references and selected data
that nodes require to coordinate safely and that a checkpointer can serialize.

```text
Graph state        = one workflow's current execution facts
Conversation memory = recent messages supplied as context
Long-term memory   = user/project facts in MemoryService SQLite
RAG                = relevant document evidence from the vector store
```

`node_trace` uses an append reducer. Each node adds one structured lifecycle event instead
of replacing previous events. Other fields intentionally replace their previous value, such
as the latest serialized plan or next task ID. This gives deterministic update semantics
without adding merge complexity where it is not needed.

## Routing, Loops, And Errors

`graph/routing.py` contains the conditional decisions. This avoids burying unrelated
business rules inside every node.

```text
execute_task
  waiting_for_approval -> approval
  transient safe failure -> retry_task -> execute_task
  any other outcome -> task_router

evaluate
  goal satisfied -> finalize -> END
  bounded replan required -> replan -> task_router
  otherwise -> finalize -> END
```

An exception is a technical event. A workflow failure is a routing decision: retry only a
classified transient and non-consequential failure; replan when the evaluated goal is still
recoverable; stop safely for permanent configuration/authentication problems. Side-effecting
tools never enter the graph retry loop. Stage 8 approval, version binding, permissions, and
idempotency receipts remain authoritative.

## Human Approval And Interrupts

The existing `ApprovalService` prepares the frozen action proposal. The graph does not
reimplement risk policy. When a task is waiting for approval, the `approval` node calls
LangGraph's `interrupt()` with a JSON-safe review payload.

```text
execute_task
  -> Stage 8 proposal and risk check
  -> graph checkpoint
  -> interrupt / Streamlit approval panel
  -> human approve, deny, cancel, or edit
  -> Command(resume=...)
  -> approval node rechecks durable service state
  -> execute_task
```

LangGraph restarts the interrupted node from its beginning after a resume. That is why the
node reads the approval service again instead of trusting an in-memory callback. The UI edits
the Stage 8 proposal through the existing versioned service, synchronizes the serialized plan
into the checkpoint, then resumes the same graph thread. A human-approved action is still
executed only through the existing exact-payload/idempotency boundary.

## Checkpointing And Thread IDs

`graph/checkpoints.py` uses LangGraph's current `SqliteSaver`. A checkpoint is written after
graph steps. `thread_id` is the durable pointer for one graph run, while `run_id` identifies
that application-level execution for observability.

```text
thread stage9_a
  planner checkpoint
  task checkpoint
  approval interrupt checkpoint
  application restart
  same thread ID -> restore state -> resume
```

SQLite is intentionally a local development solution. It works for this single-process
Streamlit project and supports the Stage 9 restart-recovery demonstration. Production
multi-worker systems should use a durable shared store such as a supported Postgres saver,
with encryption, authentication, and operational controls.

## Graph Definition Versus Graph Run

```text
Graph definition
  The reusable blueprint in graph/graph.py

Graph run
  One goal, one state, one run_id, one thread_id, and its checkpoints
```

Two users or two requests must use distinct thread IDs. Reusing a thread ID intentionally
continues that execution; creating a new one starts isolated state.

## Streamlit Developer View

Complex goals with **Use LangGraph for complex goals** enabled show a `LangGraph Execution`
expander. It reports run/thread IDs, current status, next node, interruption state, retries,
and per-node duration. It deliberately excludes model reasoning, prompts, secret values, and
retrieved document content.

## Verification

Run the focused Stage 9 tests:

```powershell
.venv\Scripts\python.exe -m pytest -q test_langgraph.py
```

They verify linear state flow, conditional routing, a bounded retry loop, permanent failure
termination, interrupt/resume after a new graph instance opens the same SQLite checkpoint,
and isolated threads.

Run the application:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

Then test these visible paths:

1. Ask `Research JSON and XML, compare their tradeoffs for a small web API, and recommend one.`
   Confirm `LangGraph Execution` shows planner, router, task nodes, evaluation, and finalize.
2. Ask a simple question such as `What is an API?` Confirm it still uses the direct path.
3. Ask `Send a simulated project update to john@example.com with subject Weekly Update and body The build is ready.`
   Confirm the graph pauses, displays the existing Stage 8 preview, and shows `approval` as the
   next node.
4. Restart Streamlit while the approval is pending. Confirm the same proposal and graph thread
   return, then approve it. The graph resumes rather than restarting planning.
5. Inspect the trace and approval audit. Confirm no real email is sent because the tool remains
   a safe simulation.

## LangGraph, LangChain, And Other Choices

LangGraph provides state, nodes, edges, loops, interrupts, and persistence. LangChain is a
different layer that provides broad model, prompt, tool, and retriever integrations. This
project does not add LangChain because its direct Groq HTTP client and existing services are
already clear and sufficient.

| Option | Best at | Use it when | Not the primary fit here because |
|---|---|---|---|
| Plain Python | Small linear workflows | One LLM call or one simple tool call | topology is already growing beyond a short function |
| LangGraph | Stateful agent orchestration | branching, loops, retries, approvals, durable runs | it does not replace domain services |
| Temporal | Reliable distributed business workflows | workers, timers, compensation, long-running production jobs | unnecessary infrastructure for this local app |
| Airflow | Scheduled batch data pipelines | time-based ETL and dependency DAGs | not designed for interactive agent interruptions |
| Prefect/Dagster | Data and asset orchestration | observable data workflows | a different execution model from conversational agents |
| General state-machine library | Deterministic finite states | smaller non-LLM state machines | lacks LangGraph's agent-focused persistence conventions |

Do not use LangGraph for `user -> one LLM call -> answer`, or for one clean tool call with no
meaningful branching. It becomes valuable when state, conditional routing, cycles, retries,
human interaction, and durable execution are actual requirements.

## Production Evolution And Limits

```text
9A simple graph -> 9B routing -> 9C retry loops -> 9D approval interrupt
-> 9E SQLite checkpoints -> 9F persistent shared store -> 9G safe parallel work
-> 9H tracing/evaluation -> production orchestration
```

LangGraph does not make complexity disappear. Poor node boundaries, oversized state, unsafe
side effects, too many loops, expensive LLM calls, low-quality retrieval evidence, and
distributed operations still require engineering judgment. The Stage 9 rule is the same as
every previous stage: use the framework to make real workflow complexity visible and testable,
not to hide it.
