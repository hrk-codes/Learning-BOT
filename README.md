# Stage 6 Long-Term Memory Agent

A Streamlit AI agent built from first principles with Groq, recent conversation history,
controlled tools, local RAG, and scoped long-term user/project memory.

## Architecture

```text
User
  -> Streamlit UI
  -> recent conversation history (Stage 2 JSON)
  -> long-term memory service (Stage 6 SQLite)
  -> agent loop
       -> answer
       -> retrieve document knowledge (Stage 5 RAG)
       -> call a permitted tool (Stage 4)
  -> Groq LLM
```

The boundaries matter:

```text
Conversation -> recent messages from this chat
Memory       -> durable, scoped user/project facts
RAG          -> evidence from indexed reference documents
Tools        -> permissioned actions and fresh observations
Agent state  -> one request's goal, trace, and counters
```

## Memory Pipelines

Write:

```text
Explicit user statement -> typed candidate -> validation/policy
-> deduplication -> conflict resolution -> SQLite -> audit event
```

Read:

```text
Current request -> user/project SQL filter -> lexical relevance
-> confidence + importance + recency + scope ranking
-> context budget -> separate untrusted memory section -> agent
```

The V1 extractor is deliberately conservative and rule-based. It stores explicit forms
such as `My name is...`, `I prefer...`, `My long-term goal is...`, project statements,
and `Remember that...`. Ordinary greetings do not become durable memories. LLM-based
extraction and semantic vector memory remain isolated future upgrades behind the same
typed candidate and ranked-retrieval interfaces.

## Project Structure

```text
app.py                         Streamlit chat, Memory Center, RAG UI, debug views
config.py                      Environment and runtime configuration
agent/                         Bounded decisions, state, routing, and orchestration
memory/chat_memory.py          Stage 2 recent conversation JSON
memory/models.py               Typed Stage 6 memory contracts
memory/repository.py           SQLite schema, scoped queries, transactions, audit
memory/policy.py               Validation, source authority, secret protection
memory/extractor.py            Conservative explicit-fact extraction
memory/ranker.py               Deterministic query-dependent ranking
memory/context_builder.py      Budgeted model-facing memory context
memory/service.py              Remember, search, update, forget, inspect, clear
rag/                           Stage 5 document ingestion and retrieval
tools/                         Stage 4 registry, permissions, and tools
docs/STAGE5_RAG.md             RAG architecture and testing
docs/STAGE6_MEMORY.md          Memory architecture, tradeoffs, security, evolution
```

SQLite is part of Python's standard library, so Stage 6 adds no dependency. Generated
database files and WAL files are ignored by Git.

## Setup

```powershell
cd "C:\Users\hrkgh\Agent learn\BOT 1"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Put one fresh Groq key in `.env`:

```text
GROQ_API_KEY=your-fresh-key
```

Start the app:

```powershell
streamlit run app.py
```

Open `http://localhost:8501`.

## How To Check Stage 6

1. Confirm the page title is `Stage 6 Long-Term Memory Agent` and the sidebar has a
   `Memory Center` with an ON/OFF toggle, manual form, active records, deletion controls,
   and an audit trail.

2. Ask `Hello.` Open `Active memories`. Expected: no new long-term record.

3. Ask `Remember that my favorite programming language is Python.` Expected: a stored
   confirmation. Open `Active memories` and verify a profile/user record with high
   confidence and importance.

4. Ask `What do you remember about me?` Expected: only real active SQLite records are
   listed. This command works without Groq because it reads the memory service directly.

5. Stop Streamlit, start it again, and repeat step 4. Expected: the Python memory remains,
   proving persistence is independent from the current process/session.

6. With a valid Groq key, ask `What programming language should I use for my AI project?`
   Open `Memory Debug`. Expected: the Python record has a retrieval score and is marked
   `injected`; the agent can use it in the answer.

7. Ask `Explain database normalization.` Expected: the programming preference is not
   injected because scope/importance cannot make a lexically unrelated record relevant.

8. Ask `I use Python for my backend.` Then ask `I'm moving my backend projects to Go.`
   Expected: one active Go record, the Python record marked `superseded` in SQLite, and
   a supersession event in the audit trail.

9. Ask `Forget that I prefer Python.` Expected: matching content is deleted from SQLite;
   `What do you remember about me?` no longer lists it; the audit trail retains deletion
   metadata but not the deleted text.

10. Turn long-term memory OFF. Ask a personalized question and then state a new preference.
    Expected: no long-term records are retrieved or written. Recent chat history still
    works and the Memory Center can still inspect/delete user-owned data.

11. Add a project-scoped fact in the manual form. Verify its record shows `project` scope.
    Use `Clear project memories`; user-profile records should remain.

12. For RAG plus memory, index a sample PDF and ask for document advice tailored to your
    stored preference. Expected: `long_term_memory` and `knowledge_base` remain separate
    in behavior, with memory scores in `Memory Debug` and document chunks in `RAG Debug`.

13. For tool plus memory, enable `calculator.evaluate` and ask for a calculation formatted
    according to a stored explanation preference. Expected: memory informs the decision;
    the tool performs the arithmetic; neither impersonates the other.

Run the complete automated evaluation:

```powershell
python -m pytest -q
```

For the deeper model, ranking formula, alternatives, privacy boundary, and production
evolution, read [`docs/STAGE6_MEMORY.md`](docs/STAGE6_MEMORY.md).
