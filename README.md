# Learning-BOT: AI Systems Engineering, Stages 1-6

Learning-BOT is one continuously evolving Streamlit project built to understand AI
systems from first principles. It began as one raw HTTP request to a Groq LLM and now
includes conversation history, a bounded agent runtime, controlled tools, document RAG,
and governed long-term memory.

The goal is not to collect popular frameworks. Each stage adds one capability only after
the limitation of the previous stage becomes clear.

## Learning Journey

| Stage | Capability added | Problem it solves | Main engineering lesson |
|---|---|---|---|
| 1 | Groq chat assistant | How does an application call an LLM? | An LLM call is an HTTP request with headers, JSON, messages, parameters, and streamed events. |
| 2 | Conversation memory | How does a chat continue across messages and restarts? | Stored history and model context are related but different. |
| 3 | Agent runtime | How can the model pursue a goal through controlled steps? | The LLM decides; Python owns state, validation, routing, and stopping. |
| 4 | Tool calling | How can the agent perform actions or obtain fresh data? | A model requests tools, but a permissioned runtime executes them. |
| 5 | RAG knowledge | How can the agent answer from private/reference documents? | Retrieval supplies evidence; it does not train the model. |
| 6 | Long-term memory | How can the agent retain useful user/project facts safely? | Memory is a governed data system, not a larger transcript. |

## Current Architecture

```text
                                      USER
                                        |
                                        v
                                STREAMLIT CHAT UI
                                        |
                 +----------------------+----------------------+
                 |                      |                      |
                 v                      v                      v
        RECENT CONVERSATION      LONG-TERM MEMORY          RAG DOCUMENTS
          history.json                SQLite              vectors + metadata
                 |                      |                      |
                 +----------------------+----------------------+
                                        |
                                        v
                                  AGENT RUNTIME
                           observe -> decide -> route
                                        |
                    +-------------------+-------------------+
                    |                   |                   |
                    v                   v                   v
              INTERNAL STEP      RETRIEVE_KNOWLEDGE      TOOL_CALL
                                                            |
                                                            v
                                                    TOOL REGISTRY/MANAGER
                    |                   |                   |
                    +-------------------+-------------------+
                                        |
                                        v
                                GROQ CHAT COMPLETIONS
                                        |
                                        v
                              FINAL ANSWER + DEBUG TRACE
```

## The Boundaries That Matter

| Component | Question it answers | Storage/lifetime |
|---|---|---|
| Conversation history | What was said recently? | JSON plus Streamlit session state |
| Long-term memory | What durable user/project facts are relevant now? | SQLite across restarts |
| RAG | What do indexed reference documents say? | Saved PDFs plus vector index |
| Tools | What action or fresh lookup should be performed? | One controlled execution |
| Agent state | What is happening during this request? | One bounded agent run |
| LLM | What should be generated or selected next? | Stateless API call unless context is supplied |

Keeping these inputs separate makes behavior easier to secure, test, and debug.

---

## Stage 1: AI Chat Assistant

### Goal

Accept a question in Streamlit, call Groq through raw HTTP/JSON, and display the streamed
response.

### Why This Stage Exists

Before adding memory or agents, the project needed a clear understanding of the smallest
LLM application:

```text
User -> UI -> Python -> HTTP/JSON -> Groq API -> LLM -> response -> UI
```

### Request Flow

```text
1. Read GROQ_API_KEY from the environment.
2. Build Authorization and Content-Type headers.
3. Build a JSON body containing model, messages, temperature, max_tokens, and stream.
4. POST to https://api.groq.com/openai/v1/chat/completions.
5. Read Server-Sent Event lines from the response.
6. Parse each JSON chunk and yield generated text.
7. Display text incrementally in Streamlit.
```

### Core Concepts Learned

- API keys belong in `.env`, never in committed source code.
- `Authorization: Bearer <key>` proves which Groq account is making the request.
- The system prompt defines behavior and boundaries; it does not guarantee factual truth.
- `user`, `assistant`, and `system` roles give the model conversation structure.
- Temperature controls variation, not intelligence or factual accuracy.
- `max_tokens` limits generated output, not the complete prompt size.
- Streaming improves perceived responsiveness but adds incremental parsing failure modes.
- Timeouts, HTTP status handling, malformed JSON handling, and secret-safe logs are part
  of the API boundary.

### Main Files Today

```text
config.py                 Environment variables and model defaults
llm/groq_client.py        Raw HTTP request, SSE parsing, retries, and errors
prompts/system_prompt.py  Original chat behavior prompt
app.py                    Streamlit interface
```

These files have evolved, but they still contain the Stage 1 foundation.

### How To Verify The Concept

1. Start the app and ask `Explain HTTP headers in three sentences.`
2. Change temperature and compare response variation.
3. Reduce max tokens and observe output truncation.
4. Temporarily remove the key and verify the UI reports a missing credential safely.

### Limitation Revealed

Every request is independent unless previous messages are sent again. That leads to
Stage 2.

---

## Stage 2: Conversation Memory

### Goal

Keep a conversation coherent across Streamlit reruns and application restarts.

### What Was Added

```text
User message
-> append to session messages
-> persist to memory/history.json
-> select latest N messages
-> send selected context to the LLM
-> save assistant response
```

### Three Different Ideas

```text
Session state      = keeps the UI alive during Streamlit reruns
Persistent history = stores messages across application restarts
Context            = selected messages sent to the model for this request
```

Saving 1,000 messages does not mean all 1,000 should enter every LLM request. Storage
capacity and context-window capacity are different.

```text
history.json            = library
recent selected messages = books placed on the desk
LLM                     = reader using only what is on the desk
```

### Main File

`memory/chat_memory.py` validates message roles/content, loads history, saves history,
selects the recent message window, and clears the conversation.

### Production-Minded Lessons

- Corrupted JSON must degrade to a clear warning instead of crashing the UI.
- Invalid records should not enter model context.
- Context must be deliberately selected rather than copied blindly.
- A conversation transcript is useful continuity, but it is not structured long-term
  knowledge about the user.

### How To Verify The Concept

1. Say `My name is HRK.`
2. Ask `What is my name?`
3. Restart Streamlit and ask again.
4. Inspect the recent conversation section and `memory/history.json`.
5. Clear chat history and confirm that the transcript disappears.

### Limitation Revealed

The assistant can remember messages, but it still handles every request as one direct
generation. It cannot organize multi-step work. That leads to Stage 3.

---

## Stage 3: AI Agent Runtime

### Goal

Turn direct chat generation into a bounded runtime that can observe a goal, choose a
structured action, update state, and continue until finished.

### Agent Loop

```text
Goal
-> observe current state
-> ask LLM for one structured decision
-> parse and validate JSON
-> route the allowed action
-> update AgentState
-> repeat or finish
```

Stage 3 actions:

```text
ANALYZE
PLAN
CONTINUE
FINISH
```

### Important Mental Model

The loop is not the intelligence. The loop is orchestration.

```text
LLM    -> proposes the next decision
Python -> validates the contract, updates state, enforces limits, and stops execution
```

The runtime never accepts arbitrary model text as an executable instruction. Decisions
must satisfy `AgentDecision`, and malformed JSON stops safely.

### Main Files

```text
agent/agent.py             Application-to-runtime entry point
agent/agent_loop.py        Observe, decide, route, update, terminate
agent/agent_state.py       Goal, status, counters, observations, results, trace
agent/decision_schema.py   Allowed structured decision contract
prompts/agent_prompt.py    Model-facing action rules
```

### Safety And Reliability

- Maximum iterations prevent an accidental infinite loop.
- Unknown actions are rejected.
- `FINISH` must explicitly mark the run finished.
- Visible trace content records safe execution metadata, not hidden chain-of-thought.
- Partial work is returned when the iteration budget is exhausted.

### How To Verify The Concept

Ask:

```text
Create an AI engineering learning plan, check whether it includes Python,
APIs, LLMs, memory, and agents, then improve it if something is missing.
```

Open `Agent Execution`. A useful run may show multiple actions before `FINISH`, along
with iteration and LLM-call counts.

### Limitation Revealed

The agent can reason through steps, but it cannot perform trusted calculations or obtain
fresh external observations. That leads to Stage 4.

---

## Stage 4: Tool Calling

### Goal

Give the agent controlled capabilities without giving the model unrestricted Python,
shell, network, or database access.

### Tool Flow

```text
LLM TOOL_CALL decision
-> action router
-> tool manager checks whether the tool is active
-> input schema validation
-> permission boundary
-> trusted executor
-> structured ToolResult
-> result becomes an agent observation
-> LLM decides whether to continue or finish
```

Important:

```text
Tool request != tool execution
Tool result  != final answer
```

The model requests a capability. Python decides whether it exists, is enabled, receives
valid arguments, and may execute. The result returns to the loop as an observation.

### Toolbox

```text
calculator.evaluate   Safe arithmetic expression evaluation
weather.get_current   Current weather lookup
search.web            Focused web search result
```

### Main Files

```text
tools/base.py          ToolDefinition and ToolResult contracts
tools/schemas.py       Input validation helpers
tools/registry.py      Catalog of available tools
tools/manager.py       Active-tool and permission boundary
tools/factory.py       Default toolbox assembly
agent/action_router.py Executes approved actions and returns observations
```

### Production-Minded Lessons

- A registry separates discovery from execution.
- The active toolbox should contain only user-enabled capabilities.
- Arguments generated by an LLM are untrusted input.
- Tool failures should become structured observations, not crash the agent loop.
- Tool results should include success, result/error, metadata, and elapsed time.

### How To Verify The Concept

1. Ask `What is 12345 * 678?` and inspect the calculator tool call.
2. Ask for current weather and inspect the weather observation.
3. Disable a tool in the sidebar and confirm it cannot execute.
4. Ask for weather plus a Celsius-to-Fahrenheit conversion and inspect the multi-tool
   sequence.

### Limitation Revealed

Tools provide actions and fresh lookups, but they do not make private PDFs available to
the model. That leads to Stage 5.

---

## Stage 5: RAG Knowledge Layer

### Goal

Let the existing agent retrieve evidence from indexed reference documents without
fine-tuning the model or creating a separate PDF chatbot.

### RAG Is Two Pipelines

Ingestion happens when a PDF is added:

```text
PDF
-> validate and save source
-> parse pages
-> clean text
-> create overlapping chunks
-> attach filename/page/version metadata
-> embed chunks once
-> store vectors
```

Retrieval happens for a relevant request:

```text
Agent chooses RETRIEVE_KNOWLEDGE
-> focused query
-> query embedding
-> cosine similarity
-> top-k plus minimum-score filtering
-> grounded evidence observation
-> agent answer
-> source references
```

### What RAG Does Not Do

- It does not retrain or fine-tune Groq's model.
- It does not guarantee an answer exists in the documents.
- It does not make every chunk relevant.
- It does not turn document text into trusted system instructions.

Document chunks are untrusted evidence. They remain separate from system rules, user
memory, and tool permissions.

### Current Implementation

```text
pypdf                                      Text-based PDF extraction
sentence-transformers/all-MiniLM-L6-v2    Local embeddings
JSON vector store                          Inspectable learning index
cosine similarity                          Semantic ranking
1,200-character chunks                     Default window
200-character overlap                      Boundary protection
```

### Main Files

```text
rag/ingestion/                 Loading, parsing, cleaning, chunking
rag/embeddings/embedder.py     Text-to-vector interface
rag/storage/vector_store.py    Documents, chunks, vectors, cosine search
rag/retrieval/retriever.py     Query embedding and ranked retrieval
rag/context/context_builder.py Grounded model observation and sources
rag/pipeline.py                Indexing, retrieval, re-index, deletion lifecycle
docs/STAGE5_RAG.md             Deeper architecture and tradeoffs
```

### How To Verify The Concept

1. Upload `documents/sample/employee-handbook.pdf` in the Knowledge Base.
2. Ask:

   ```text
   According to the employee handbook, how many unused leave days
   can I take into next year?
   ```

3. Confirm `RETRIEVE_KNOWLEDGE` appears in the trace.
4. Confirm the answer says `10` and includes `employee-handbook.pdf - page 1`.
5. Open `RAG Debug` and inspect query, chunk ID, score, filename, page, and latency.
6. Ask an unsupported document question and verify the agent does not invent evidence.

### Debugging Principle

```text
PDF parsing -> chunks -> embeddings -> retrieved evidence -> model context -> answer
```

If the correct chunk was never retrieved, changing the final prompt is not the first fix.

### Limitation Revealed

RAG knows reference documents, not durable user preferences, changing project facts, or
personal goals. That leads to Stage 6.

---

## Stage 6: Long-Term Memory

### Goal

Maintain a controlled, scoped, inspectable, persistent, and retrievable model of useful
user and project information.

Stage 6 does not replace Stage 2. Recent conversation and long-term memory solve
different problems.

### Memory Write Pipeline

```text
Explicit user statement
-> conservative extractor
-> typed MemoryCandidate
-> validation
-> memory policy
-> normalized deduplication
-> conflict resolution
-> SQLite transaction
-> audit event
```

The runtime remains authoritative. Neither the LLM nor Streamlit writes SQL directly.

### Typed Memory Record

```text
memory_id
user_id + project_id
memory type + scope
stable concept key
content + normalized content
source
confidence + importance
status
created_at + updated_at
valid_from + valid_until
```

Supported types include profile, semantic, episodic, procedural, and project memory.
Stage 6 V1 persistently enables user and project scopes.

### Source And Confidence

```text
"I prefer Python."                 -> explicit, high confidence
"The user probably prefers Python" -> inference, lower authority
```

Source-aware confidence rules prevent a weak inference from silently replacing an
explicit statement.

### Duplicate And Conflict Handling

Exact normalized duplicates do not create another active row.

When two facts share a stable key but disagree:

```text
Old: User's backend language is Python.
New: User's backend language is Go.

Python -> superseded historical record
Go     -> new active record
```

Updates preserve useful history. Explicit deletion physically removes memory content
while retaining content-free audit event metadata.

### Memory Read Pipeline

```text
Current request
-> user/project SQL scope filter
-> active and valid candidates
-> lexical relevance requirement
-> deterministic ranking
-> configurable result limit
-> context-character budget
-> narrow untrusted memory payload
-> agent
```

V1 ranking:

```text
45% lexical relevance
15% scope match
15% importance
15% confidence
10% recency
```

Scope and importance can rank relevant memories, but cannot make an unrelated memory
relevant by themselves.

### User Controls

- Memory ON/OFF controls both long-term reads and new writes.
- Active memories can be inspected.
- A single memory can be forgotten.
- Project memories can be cleared without deleting user-profile memory.
- All memory can be deleted after confirmation.
- Audit events show created, superseded, expired, and deleted operations.

### Observability

`Memory Debug` exposes candidate count, accepted/rejected writes, database latency,
ranking latency, scores, retrieved records, injected records, context characters, and
approximate tokens. It does not expose hidden model reasoning.

### Main Files

```text
memory/models.py          Typed contracts and metrics
memory/repository.py      SQLite schema, transactions, scoped queries, audit
memory/policy.py          Validation, confidence authority, secret protection
memory/extractor.py       Conservative explicit-fact extraction
memory/ranker.py          Query-dependent deterministic scoring
memory/context_builder.py Budgeted model-facing payload
memory/service.py         Remember, search, list, update, forget, clear
docs/STAGE6_MEMORY.md     Deeper architecture, privacy, and evolution
```

### Real Failure Solved During The Build

Windows tests initially failed to remove a temporary SQLite database because its file
handle remained open.

Root cause:

```text
sqlite3 connection context manager
-> commits or rolls back
-> does not close the connection handle
```

The repository now owns an explicit transaction context and always closes connections
in `finally`. This fixed test cleanup and prevents Streamlit reruns from accumulating
database handles.

### How To Verify The Concept

1. Ask `Hello.` and confirm no long-term memory is created.
2. Ask `Remember that my favorite programming language is Python.`
3. Ask `What do you remember about me?` and verify only real SQLite records appear.
4. Restart Streamlit and confirm the record persists.
5. Ask for a programming-language recommendation and inspect `Memory Debug`.
6. Store a Python backend fact, then state that you moved to Go. Verify supersession.
7. Forget the preference and verify it disappears from active storage.
8. Turn Memory OFF and verify no long-term records are read or written.

### Current Limitation And Next Evolution

Stage 6 V1 deliberately uses structured SQL filters plus lexical ranking. The interface
allows this progression without changing the agent contract:

```text
SQLite structured memory
-> memory embeddings in an isolated index
-> hybrid structured + semantic ranking
-> PostgreSQL for hosted multi-user deployment
-> Redis only for proven hot-memory latency needs
```

---

## Project Structure After Stage 6

```text
app.py                         Streamlit UI and workflow coordination
config.py                      Environment and application configuration

agent/                         Bounded agent state, decisions, loop, routing
llm/groq_client.py             Raw Groq HTTP/SSE client and rate-limit handling
prompts/                       System, agent, RAG grounding prompts

memory/chat_memory.py          Stage 2 conversation history
memory/models.py               Stage 6 typed long-term memory
memory/repository.py           SQLite persistence and audit events
memory/service.py              Long-term memory business behavior
memory/extractor.py            Explicit memory candidate extraction
memory/policy.py               Validation and write authority
memory/ranker.py               Structured relevance ranking
memory/context_builder.py      Budgeted memory context

tools/                         Tool contracts, registry, manager, implementations
rag/                           PDF ingestion, embeddings, retrieval, context

documents/sample/              Controlled PDFs for testing
documents/raw/                 Uploaded PDFs, ignored by Git
vector_store/                  Generated RAG vectors, ignored by Git
docs/                          Stage 5 and Stage 6 deep documentation

test_memory.py                 Stage 2 tests
test_agent.py                  Stage 3/4/5/6 agent tests
test_tools.py                  Stage 4 tests
test_rag.py                    Stage 5 tests
test_long_term_memory.py       Stage 6 tests
test_llm.py                    Groq retry/gate/redaction tests
```

## Setup

```powershell
cd "C:\Users\hrkgh\Agent learn\BOT 1"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Add one fresh Groq key to `.env`:

```text
GROQ_API_KEY=your-fresh-key
```

Never commit `.env` and never paste real API keys into chat, logs, tests, screenshots, or
GitHub.

Important configuration groups are documented in `.env.example`:

```text
GROQ_*                Model, output, timeout, and bounded 429 retry settings
CHAT_HISTORY_*        Stage 2 persistence and context window
MAX_AGENT_ITERATIONS  Stage 3 loop limit
RAG_*                 Stage 5 documents, embeddings, chunks, ranking, context
LONG_TERM_MEMORY_*    Stage 6 database, identity, retrieval, and budget
```

## Run

```powershell
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

`localhost` means the Streamlit server is running on this computer. Port `8501` identifies
the local process. Other devices cannot reliably access it unless Streamlit is bound to a
network interface and firewall/network rules permit access. A hosted deployment would
replace localhost with a public or private domain.

## Groq Rate-Limit Behavior

The client uses one authorized key and a process-local request gate. Concurrent
Streamlit sessions enter the Groq request boundary one at a time.

For HTTP `429` only:

```text
read retry-after
-> validate delay
-> stay within cumulative wait budget
-> retry the same key at most the configured number of times
-> return a clear error when the budget is exhausted
```

Authentication errors, invalid requests, unavailable models, malformed responses,
timeouts, and connection failures are not disguised as rate-limit retries. Logs never
print API keys, prompts, or private document content.

## Run All Tests

```powershell
python -m pytest -q
```

Current verified result:

```text
40 passed
```

There is one harmless Pytest collection warning because the Stage 5 test embedding helper
is named `TestEmbedder` while defining a constructor. It does not prevent any test from
running.

## Debug From First Principles

When something fails, find the first incorrect boundary instead of changing the final
prompt immediately.

```text
API failure:
environment -> headers -> JSON payload -> HTTP status -> SSE/JSON parsing

Conversation failure:
saved messages -> validated messages -> recent selection -> model context

Agent failure:
observation -> raw decision -> JSON contract -> routed action -> state update -> stop rule

Tool failure:
active registry -> schema -> permission -> executor -> ToolResult -> next decision

RAG failure:
PDF parse -> chunks -> embeddings -> similarity -> selected evidence -> answer

Memory failure:
extraction -> validation -> policy -> SQLite -> scope filter -> ranking -> context budget
```

## Production Evolution

```text
Stage 1-2 local learning
    Raw HTTP + Streamlit + JSON

Stage 3-4 controlled execution
    Typed state + bounded loop + permissioned tools

Stage 5-6 knowledge and personalization
    RAG vectors + SQLite long-term memory

Hosted production
    Web API + authentication + PostgreSQL + background jobs + observability

Scale when evidence requires it
    pgvector/vector service + Redis cache + distributed rate limits + evaluations
```

The engineering rule remains the same across every stage: introduce technology because a
measured problem requires it, not because the technology is popular.

## Deep Documentation

- [Stage 5 RAG architecture](docs/STAGE5_RAG.md)
- [Stage 6 long-term memory architecture](docs/STAGE6_MEMORY.md)
