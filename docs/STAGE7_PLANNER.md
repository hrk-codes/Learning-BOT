# Stage 7 Planner, Dependency Graph, and Executor

## Objective

Stage 7 adds deliberate planning for complex goals without converting Learning-BOT into a
multi-agent or distributed workflow system. Simple questions continue through the Stage 6
reactive agent. Complex goals receive a bounded, machine-readable execution plan.

```text
Goal
-> planning-need detector
-> planner JSON
-> plan validator
-> dependency graph
-> ready-task scheduler
-> capability executor
-> task result and plan state
-> goal evaluator
-> finish or bounded replan
```

## Why Planning Is Conditional

Planning costs an additional LLM call, tokens, parsing, validation, and latency. A direct
question such as `What is JSON?` does not benefit from task decomposition. A request to
research two options, calculate implications, compare evidence, and recommend one does.

`PlanningNeedDetector` is deterministic and cheap. It scores signals such as comparison,
research, explicit sequences, multi-source synthesis, multiple outcomes, and long multi-part
requirements. The UI reports the score and reasons so routing can be inspected.

This detector is intentionally conservative. A later evaluation set can replace its initial
rules with a measured classifier if real requests show enough false positives or negatives.

## Planner Versus Executor

The planner emits intent:

```json
{
  "id": "compare_options",
  "description": "Compare the collected database evidence",
  "capability": "llm",
  "dependencies": ["research_postgres", "research_mysql"],
  "inputs": ["postgres_notes", "mysql_notes"],
  "output_key": "comparison",
  "priority": 0,
  "required": true,
  "max_retries": 0
}
```

The executor owns action. It invokes only four existing capability boundaries:

| Capability | Runtime boundary |
|---|---|
| `llm` | One task-specific Groq completion |
| `tool` | Existing `ToolManager` schema, enabled-tool, and permission checks |
| `rag` | Existing `RagPipeline.retrieve()` evidence path |
| `memory` | Existing scoped `MemoryService.search()` plus context builder |

The planner cannot run Python, SQL, shell commands, arbitrary HTTP, or hidden tools. A plan
that names a tool is not authorization to execute it.

## Plan And Task State

`PlanState` is request-lifetime workflow state. It is deliberately separate from Stage 6
long-term user memory.

```text
PlanState
  plan_id, goal, version, revision, status
  tasks and active_task_id
  named outputs
  assumptions
  lifecycle events
  timing and call metrics
  goal evaluation and final answer
```

Every task moves through explicit states:

```text
PENDING -> READY -> RUNNING -> COMPLETED
                      |
                      +-> FAILED

PENDING/READY -> CANCELLED
dependency failure -> BLOCKED
```

`READY` remains separate from `RUNNING`. V1 executes sequentially, but the scheduler can
identify several independent ready tasks without changing their lifecycle meaning.

## Dependencies And Inputs

Dependencies define order. Inputs define dataflow.

```text
dependencies = ["retrieve_policy"]
inputs       = ["policy_evidence"]
```

A task may wait for another task without consuming its output, but an input produced by
another task must name that producer as a dependency. Output keys are unique, which prevents
silent overwrites and makes execution context testable.

## Validation Before Execution

`PlanValidator` rejects:

- empty or oversized plans;
- malformed or duplicate task IDs;
- unknown or self dependencies;
- dependency cycles;
- unavailable RAG, memory, or tool capabilities;
- missing tool names;
- duplicate or missing output keys;
- input keys not produced by dependencies;
- retry budgets above the configured maximum.

The planner receives one bounded repair attempt by default. Repair receives validator issues,
not permission to weaken validation.

## Scheduling And Failure Propagation

A pending task becomes ready only when:

```text
all dependencies completed
AND required inputs exist
AND the capability remains available
```

When a dependency fails or is cancelled, downstream tasks become `BLOCKED`. They are not
marked `FAILED` because their own executor never ran. Independent tasks may still complete,
preserving useful partial work for evaluation or replanning.

## Retry Policy

Retries happen only for explicitly transient results such as a timeout or temporary
connection failure. Invalid arguments, denied permissions, missing inputs, authentication
problems, and normal tool errors are permanent for that attempt and are not retried blindly.

The effective retry count is the smaller of:

```text
task.max_retries
PLANNER_MAX_TASK_RETRIES
```

Every attempt also consumes `PLANNER_MAX_EXECUTION_STEPS`, preventing retries and revisions
from creating an unbounded loop.

## Replanning

Replanning is triggered only when goal evaluation identifies missing work or a failure makes
the current route unusable. It is capped by `PLANNER_MAX_REVISIONS`.

Completed tasks and outputs are immutable. A revision retires unfinished historical tasks and
adds new uniquely identified tasks. The full revised graph is validated again before running.
History remains visible in the execution panel.

## Goal Evaluation

The runtime does not equate completed checkboxes with success. When no task is ready, a
dedicated evaluator receives the goal, safe task summaries, and named outputs.

Python applies a hard rule:

```text
required incomplete task -> plan cannot be completed
```

Even when every required task completed, the evaluator must positively confirm that the
original goal is satisfied and produce a grounded final answer. Otherwise the runtime revises
or fails clearly when its revision budget is exhausted.

## Security Boundaries

- Tool arguments remain untrusted and pass through schemas.
- Side-effecting tools remain blocked without explicit user confirmation.
- Memory and RAG content are data, not instructions.
- Planner output cannot introduce unknown capabilities.
- Logs record IDs, statuses, attempts, latency, and counts, not keys or full private context.
- Visible events expose lifecycle metadata, not hidden chain-of-thought.

## Configuration

```text
PLANNER_ENABLED=true
PLANNER_TEMPERATURE=0.1
PLANNER_MIN_OUTPUT_TOKENS=640
PLANNER_MAX_TASKS=5
PLANNER_MAX_REVISIONS=2
PLANNER_MAX_EXECUTION_STEPS=12
PLANNER_MAX_TASK_RETRIES=1
PLANNER_MAX_REPAIR_ATTEMPTS=1
```

Low planner temperature favors stable JSON. The minimum output budget gives the model room for
the plan contract even when the answer slider is set low.

## Step-By-Step Verification

### 1. Confirm The Direct Path

Ask:

```text
What is an HTTP request?
```

Open `Agent Execution`. Expected: `mode` is `direct`; no execution plan is created.

### 2. Confirm Complex Decomposition

Ask:

```text
Research PostgreSQL and MySQL, compare their tradeoffs for a small AI product,
and recommend one with clear reasons.
```

Open `Execution Plan`. Expected: planned mode, multiple task IDs, explicit dependencies,
named outputs, completed progress, a separate goal evaluation, and a final answer.

### 3. Confirm Independent Readiness

Use a goal that independently researches two subjects before comparison. In lifecycle events,
both research tasks may become READY before V1 executes them one at a time.

### 4. Confirm Capability Validation

Disable web search, then ask for a multi-source current-web comparison. Expected: the planner
must repair the plan using available capabilities or stop before executing an unknown/disabled
tool.

### 5. Confirm RAG, Memory, Tool, And LLM Composition

1. Index `documents/sample/employee-handbook.pdf`.
2. Store a related personal preference through Memory Center.
3. Ask for a recommendation that uses the preference, handbook policy, and an exact numeric
   calculation.
4. Inspect task capabilities, output keys, RAG sources, memory metrics, and calculator result.

### 6. Run Automated Tests

```powershell
python -m pytest -q test_planner.py
python -m pytest -q
```

The Stage 7 suite covers routing, dependencies, cycles, capability validation, plan repair,
memory context, transient retry, side-effect permissions, blocked descendants, goal evaluation,
replanning, cancellation, and combined capability execution.

## Deliberate V1 Limits

- one local Streamlit process;
- sequential task execution;
- no durable workflow resumption after process failure;
- no multi-agent delegation;
- no distributed queue, Redis scheduler, or Kubernetes;
- no arbitrary code execution;
- no automatic approval for side effects.

These are not missing because the technologies are unpopular. They are deferred because Stage 7
does not yet have a measured problem that requires their operational cost.
