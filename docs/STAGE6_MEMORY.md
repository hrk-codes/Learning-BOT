# Stage 6: Long-Term Memory

Stage 6 adds a controlled memory service to the Stage 5 agent. It does not replace
conversation history, RAG, tools, or per-request agent state.

## The Five Different Inputs

| System part | What it contains | Lifetime | Authority |
|---|---|---|---|
| Conversation history | Recent user/assistant messages | Current chat | Context only |
| Long-term memory | Selected user and project facts | Across restarts | Untrusted user data |
| RAG | Evidence from indexed documents | Until document deletion | External evidence |
| Tools | Actions and fresh external results | One tool call | Permission-controlled observation |
| Agent state | Goal, trace, counters, observations | One request | Runtime orchestration |

Stage 2's `history.json` answers "what was said recently?" It cannot reliably classify,
scope, rank, update, audit, or selectively delete durable facts. Stage 6 stores those
facts as records in SQLite and retrieves only records relevant to the current request.

## Architecture

```text
User message
  |-- recent chat -------------------------------> agent context
  |-- explicit fact -> extractor -> candidate
                          -> policy/validation
                          -> deduplication
                          -> conflict resolution
                          -> SQLite + audit event

Current request -> scoped SQL candidates -> deterministic ranker
                -> context budget -> untrusted long-term-memory section
                -> agent

Indexed PDF -> RAG retrieval --------------------> separate knowledge section
Tool request -> permissioned tool ----------------> separate observation
```

The read and write pipelines are deliberately separate. Reading memory does not grant
the model permission to write it. The model never executes SQL and the UI never edits
database tables directly.

## Files And Responsibilities

```text
memory/models.py          Typed candidates, records, events, results, and metrics
memory/repository.py      SQLite schema, transactions, scoped queries, and audit writes
memory/policy.py          Validation, confidence caps, secret checks, and normalization
memory/extractor.py       Conservative explicit-statement extraction
memory/ranker.py          Query relevance, scope, recency, importance, and confidence
memory/context_builder.py Narrow, budgeted, model-facing context
memory/service.py         Remember, search, update, forget, list, and clear behavior
memory/chat_memory.py     Stage 2 recent conversation history; still independent
agent/agent_loop.py       Injects memory, RAG, conversation, tools, and state separately
app.py                    Chat UI, Memory Center, controls, and debug metrics
```

## Typed Memory Model

Supported memory types are:

- `profile`: identity, durable goals, and stable preferences.
- `semantic`: user-specific facts and preferences.
- `episodic`: useful past events.
- `procedural`: preferred ways of working.
- `project`: facts tied to one project.

The model also defines working, conversation, task, and global scope concepts. Stage 6
V1 policy permits persistent `user` and `project` scopes only. This keeps the access
boundary understandable and avoids accidentally enabling global cross-user reads.

Every record includes:

```text
memory_id, user_id, project_id, type, scope, key, content,
source, confidence, importance, status,
created_at, updated_at, valid_from, valid_until
```

The stable `key` represents the concept being stored. For example,
`preference.backend_language` lets a new explicit Go preference supersede an old Python
preference while preserving the old record as historical.

## Write Pipeline

The V1 extractor recognizes conservative explicit forms such as:

```text
My name is ...
My favorite programming language is ...
My long-term goal is ...
I prefer ...
I use ... for backend
I'm moving my backend projects to ...
I am building ...
Remember that ...
```

It returns typed `MemoryCandidate` objects, not prose to be parsed later. Greetings and
ordinary questions produce no candidates. The Memory Center provides deterministic
manual insertion for facts outside the small rule set.

Rule-based extraction is used first because it is predictable, cheap, offline, and easy
to inspect. An LLM extractor becomes useful for varied language and richer
classification, but its JSON candidates must enter the same validation pipeline. A
hybrid extractor is the natural next step: rules for explicit high-value patterns and
an LLM proposal for harder cases.

The policy then:

1. Requires user ownership and valid project scope.
2. Rejects empty/oversized content, credentials, and stored prompt-override text.
3. Clamps confidence and importance to `0..1`.
4. Caps model inference below explicit user authority.
5. Normalizes content and keys.
6. Detects exact normalized duplicates.
7. Resolves same-key conflicts in one transaction.
8. Writes the active record and content-free event metadata.

An explicit statement can supersede an older fact. A lower-confidence model inference
cannot override an explicit user statement. Superseded records remain historical;
explicit deletion removes record content from storage while preserving only audit event
metadata.

## Read And Ranking Pipeline

SQLite first filters by `user_id`, active status, allowed user/project scope, project ID,
type when requested, and validity time. This is the security boundary as well as a
performance filter.

The ranker then requires lexical query relevance. Scope, importance, confidence, and
recency can rank relevant candidates, but cannot make an unrelated memory relevant.

```text
score = 45% lexical relevance
      + 15% scope match
      + 15% importance
      + 15% confidence
      + 10% recency
```

This is a teaching formula, not a universal ranking law. Keeping it in `ranker.py`
allows later replacement with semantic similarity or a learned reranker.

The context builder selects ranked records until `LONG_TERM_MEMORY_CONTEXT_MAX_CHARS`
is reached. It exposes only model-relevant fields and labels the entire section as
untrusted application data. RAG remains under `knowledge_base`; memory remains under
`long_term_memory`; recent chat remains normal message history.

## Storage And Retrieval Alternatives

| Choice | Good for | Limitation | Use when |
|---|---|---|---|
| JSON | Learning serialization and one chat | Weak querying/concurrency/migrations | Stage 2 |
| SQLite | Typed local persistence without a server | One-machine write scaling | Stage 6 V1 |
| PostgreSQL | Multi-user transactions and deployment | Operational infrastructure | Hosted production |
| Redis | Hot cache and short-lived coordination | Not the source of truth by default | Proven latency need |

| Retrieval | Advantage | Limitation |
|---|---|---|
| SQL filters | Exact scope and access control | Misses paraphrases |
| Keywords | Cheap request relevance | Vocabulary mismatch |
| Vectors | Semantic paraphrase matching | Model/index lifecycle and extra latency |
| Hybrid | Strong filters plus semantics | More tuning and observability |

V1 intentionally uses structured SQL plus lexical ranking. The interface already returns
ranked memory objects, so semantic retrieval can be added without changing the agent.
The next version can reuse the Stage 5 embedding provider but must keep memory vectors
in a separate collection/index and delete them with their SQLite records.

The intended evolution is:

```text
history.json
  -> SQLite + structured retrieval
  -> PostgreSQL + memory embeddings + hybrid ranking
  -> PostgreSQL + vector index + Redis hot-memory cache + background consolidation
```

## Privacy And User Control

- Every repository read and mutation requires `user_id`.
- Project retrieval admits only user memories plus the selected project.
- Memory OFF prevents long-term retrieval and new writes; recent chat still works.
- Users can inspect records, forget one, clear a project, or delete all memory.
- Broad UI deletion requires confirmation.
- Explicit chat deletion commands delete from SQLite, not only the display.
- Logs include IDs, counts, types, status, and latency, but not memory content.
- Stored content never overrides system policy, authorization, or tool permissions.

## Observability

`Memory Debug` shows extraction count, accepted/rejected writes, SQL latency, ranking
latency, candidate count, selected records, scores, context characters, approximate
tokens, and whether each record was injected. The audit view shows creation,
supersession, expiration, and deletion events without private deleted content.

When an answer is wrong, inspect in this order:

```text
extraction -> validation -> SQLite record -> scoped candidates
-> lexical relevance -> ranking -> context budget -> agent answer
```

Do not tune the prompt when the correct memory never entered the context.

## Evaluation

Run:

```powershell
python -m pytest -q
```

`test_long_term_memory.py` covers explicit extraction, ignored greetings, persistence,
user/project isolation, normalized deduplication, controlled contradiction handling,
source authority, relevant and irrelevant retrieval, physical deletion, content-free
audit metadata, Memory OFF, context budgeting, and separate agent inputs.

The practical evaluation question is: "Did the right stored memory enter the context?"
The debug view answers that without revealing private model reasoning.
