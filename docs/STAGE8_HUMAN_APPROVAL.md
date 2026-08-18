# Stage 8 Human-in-the-Loop and Safe Execution

## Objective

Stage 8 makes the Stage 7 planner interruptible at consequential actions. The model may
propose work, but Python classifies risk, freezes the exact action, asks the user, rechecks
permission, and executes only the approved version.

```text
LLM proposal
-> tool contract and argument validation
-> deterministic risk assessment
-> action proposal
-> approval gate when required
-> approve, edit, deny, cancel, or expire
-> permission and action-version recheck
-> tool execution
-> execution receipt and audit event
-> plan update, continue, replan, or finish
```

Human-in-the-loop is therefore not an `Are you sure?` dialog. A human becomes an explicit
decision-maker at a persisted workflow boundary.

## The Stage 7 Limitation

Stage 7 could validate and execute a complex plan, but it did not distinguish content
generation from a real-world consequence:

```text
Draft an email = produce text, no external side effect
Send an email  = communicate externally, consequential

List files     = read-only
Delete files   = destructive
```

Stage 8 adds this distinction without rebuilding the planner, memory, RAG, tools, or agent
runtime.

## Authority Boundaries

The layers answer different questions:

| Layer | Question |
|---|---|
| Authentication | Who is this user? |
| Authorization/permission | May this user or session invoke this capability? |
| Risk policy | Does this proposed invocation require human review? |
| Approval | Does this user authorize this exact action and version now? |
| Argument validation | Does the payload satisfy the tool contract? |
| Executor | Perform the already validated and authorized action. |

Approval never grants permission. If side-effect permission is disabled after approval, the
tool still cannot execute. This local learning app uses a configured user ID and a session
permission toggle; production would replace those with real authentication and server-side
authorization.

## Risk Model

Risk uses four levels:

```text
LOW -> MEDIUM -> HIGH -> CRITICAL
```

Side effects are explicit:

```text
NONE
READ_ONLY
REVERSIBLE_WRITE
IRREVERSIBLE_WRITE
EXTERNAL_COMMUNICATION
FINANCIAL
DESTRUCTIVE
```

`RiskEngine` starts with the tool contract and then inspects arguments. An external message
to one target and one to many targets do not have identical impact. The current deterministic
rules escalate large recipient or destructive target scopes and always classify financial
actions as critical.

The LLM never gets authority to label its own proposal safe. Deterministic rules are easy to
test and audit. A production hybrid could combine contract metadata, policy rules, resource
ownership, data sensitivity, amount limits, and an advisory model classifier, while keeping
runtime policy authoritative.

## Tool Contract Extension

Every `ToolDefinition` now exposes:

```text
risk_level
side_effect
supports_preview
requires_confirmation
```

Current examples:

| Tool | Side effect | Risk | Approval |
|---|---|---|---|
| `calculator.evaluate` | none | low | no |
| `weather.get_current` | read-only | low | no |
| `search.web` | read-only | low | no |
| LLM email drafting | none | low | no |
| `email.send_mock` | external communication | high | per action |
| `files.delete_mock` | destructive | high | per action |

The two consequential tools are safe simulations. They produce useful previews and receipts,
but send no email and delete no files.

## Action Proposal and Preview

An `ActionProposal` is not an executed action. It stores the plan/task identity, user,
tool and tool version, validated arguments, purpose, risk result, preview, action version,
expiry, argument digest, idempotency key, and status.

The preview is built from the structured arguments that will be executed. For email it shows
recipient, subject, and body. For deletion it shows exact mock paths and count. This is a
dry run: prepare and inspect the consequence without causing it.

Showing only `Tool: email.send_mock` would be inadequate because the user could not judge the
recipient or content. A trustworthy preview and the executor share the same proposal payload.

## Version-Locked Approval

Approval binds to:

```text
action ID + action version + tool name + tool version + canonical arguments
```

Those values produce a SHA-256 argument digest. Execution recalculates the digest and checks
the exact payload. More importantly, `TaskRunner` executes arguments from the approved,
durable proposal rather than mutable planner state. This closes a time-of-check/time-of-use
gap.

Editing creates version `N + 1`, marks the old request `edited`, reassesses risk, rebuilds the
preview, and creates a new pending approval. Approval for version 1 cannot authorize version
2. A changed tool contract also invalidates approval.

## Approval Lifecycle

```text
PENDING -> APPROVED -> EXECUTING -> COMPLETED
   |           |                       |
   |           +-> EXPIRED             +-> receipt
   +-> EDITED -> new version
   +-> DENIED
   +-> CANCELLED
   +-> EXPIRED
```

An approval has a configurable timeout. Expired approvals fail closed, including an approval
that expired after the click but before execution began.

Denial and cancellation are terminal for that proposal. The task enters a matching state,
downstream dependencies are blocked, and goal evaluation may request a bounded replan. The
replanner is forbidden from silently proposing the same denied/cancelled tool again in that
workflow.

## Plan Pause and Resume

`WAITING_FOR_APPROVAL` is neither success nor failure. When the executor encounters the gate:

```text
task -> WAITING_FOR_APPROVAL
plan -> WAITING_FOR_APPROVAL
serialize full PlanState to SQLite
return control to Streamlit
```

SQLite stores plans, action versions, approval requests, receipts, and audit events. On a UI
rerun or app restart, the app finds the latest waiting workflow for the configured user. A
pending decision renders the approval panel. A terminal or approved decision resumes the
same plan through `PlanningRuntime.resume()`.

No in-memory callback or hidden model continuation is trusted to reconstruct the action.

## Permission Recheck and Fail-Closed Behavior

The final tool boundary checks all of these again:

```text
tool exists and is enabled
session has the permission class
action status is approved/executing
action user matches
action and approval versions match
tool version matches
arguments satisfy schema
arguments match the approved digest and payload
```

If approval storage, user identity, permission, action version, tool version, or digest cannot
be verified, the consequential action does not execute. Approval uncertainty defaults to no
side effect.

## Idempotency and Receipts

Each action version gets one logical idempotency key:

```text
act_<id>:v<version>
```

For this single-process app, a process-wide execution gate covers receipt lookup, tool call,
and receipt write. A repeated local execution returns the existing successful receipt rather
than invoking the tool again. A failed receipt is not retried automatically because an
external timeout can leave the outcome ambiguous.

An `ExecutionReceipt` records action/version, tool, status, execution time, optional external
ID, and safe metadata. This distinguishes:

```text
approved != execution started != execution succeeded
```

Production providers should also receive and enforce the same idempotency key. Distributed
workers would require a transactional database lock or distributed idempotency store; the
process-local lock is intentionally not presented as that solution.

## Audit and Observability

The audit trail records proposal, risk, request, grant, denial, edit, cancel, expiry, start,
completion, and failure events. Events identify user, plan, task, action, version, approval,
time, and safe metadata. Arguments and preview content are not copied into audit metadata,
reducing unnecessary sensitive-data duplication.

Plan metrics expose risk-assessment time, approval requests, grants, denials, expirations,
edits, approval wait time, execution time, and tool calls. In production these support rates
and latency distributions.

```text
total workflow time = agent time + tool time + human decision time
```

Approving every read-only action would add friction and create approval fatigue. Stage 8 uses
automatic low-risk reads and meaningful per-action review for side effects.

## Alternatives and When They Fit

| Choice | Best fit |
|---|---|
| No approval | Deterministic low-risk and read-only actions. |
| Per-action approval | Consequential action where exact arguments matter; Stage 8 default. |
| Per-tool approval | Short trusted sessions with narrow reversible tools; not safe as a permanent destructive grant. |
| Per-risk-class approval | Mature policy systems with measured classification quality. |
| Batch approval | Reviewable homogeneous actions with clear combined impact and rollback. |
| Static risk rules | Small auditable systems and initial production versions. |
| LLM-assisted risk | Advisory analysis of nuanced content, never sole authorization. |
| Action summary | Simple structured side effects. |
| Dry run | Mutations where affected targets can be computed before commit. |
| Full simulation | High-impact domains with a faithful sandbox or transaction model. |
| Pause/resume | Human latency may outlive one request; Stage 8 behavior. |
| Cancel | The action is no longer wanted. |
| Replan | The goal remains valid but the denied path is unavailable. |

## Files and Responsibilities

```text
approval/models.py       Risk, action, approval, receipt, audit contracts and digest
approval/risk_engine.py  Deterministic tool-plus-argument risk classification
approval/policy.py       Runtime decision for whether approval is required
approval/service.py      Approval lifecycle, versions, expiry, receipts, persistence
approval/repository.py   SQLite actions, approvals, receipts, audit, workflow state
approval/audit.py        Secret-minimizing audit event writer
approval/gate.py         Executor-facing gate result
tools/base.py            Extended side-effect and preview contract
tools/manager.py         Permission and exact-action recheck before tool execution
executor/task_runner.py  Gate request, frozen execution, idempotency, receipts
executor/executor.py     Approval-aware task transitions
planner/runtime.py       Durable workflow pause and resume
planner/serialization.py Full PlanState persistence contract
app.py                   Preview and approve/edit/deny/cancel interface
test_human_approval.py   Stage 8 boundary and lifecycle tests
```

## Verification Walkthrough

Run the app:

```powershell
cd "C:\Users\hrkgh\Agent learn\BOT 1"
.venv\Scripts\python.exe -m streamlit run app.py
```

Then verify:

1. Ask `What is 25 * 17?` and confirm calculator completes with no approval.
2. Ask `Draft an email to John about the project update.` Confirm text is drafted and no
   simulated send action runs.
3. Ask `Send a simulated project update to john@example.com with subject Weekly Update and
   body The build is ready.` Confirm the plan pauses at `WAITING_FOR_APPROVAL` with attempts 0.
4. Inspect To, Subject, Body, risk, side effect, version, expiration, and audit events.
5. Edit the recipient or body. Confirm version increments and a new approval is required.
6. Approve the latest version. Confirm permission is rechecked, the mock executes, the task
   completes, and an execution receipt ID appears.
7. Repeat with another proposal and choose Deny. Confirm no receipt and no execution.
8. Repeat and choose Cancel. Confirm the action becomes cancelled.
9. Set `APPROVAL_TIMEOUT_SECONDS=5`, restart, wait, and confirm expiry blocks execution.
10. Disable `Side-effect capability permission` before approving/resuming and confirm the
    approved action still fails authorization.
11. Restart Streamlit while a proposal is pending and confirm the same workflow and action
    version reappear.

Run automated verification:

```powershell
python -m pytest -q test_human_approval.py
python -m pytest -q
```

## Current Limitations

Stage 8 is intentionally local and single-process. The email and deletion tools are mocks.
The configured user ID is not real authentication. SQLite writes are durable but are not a
distributed workflow engine. The process lock cannot coordinate multiple app processes. Risk
rules are illustrative rather than universal, and document access control remains separate
from side-effect approval.

A production evolution would add authenticated identity, server-enforced RBAC/ABAC, resource
ownership checks, provider-supported idempotency, transactional workers, encrypted sensitive
payloads, policy versioning, alerting, and evaluation data for approval quality and fatigue.

The invariant does not change:

> The agent proposes. Runtime policy decides what requires review. The human authorizes one
> exact consequential action. The executor performs only that validated version.
