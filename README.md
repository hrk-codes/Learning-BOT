# Stage 3 AI Agent

Small Streamlit assistant that sends chat messages to Groq with raw HTTP/JSON, conversation memory, and a tool-free AI agent runtime.

## Where It Fits

User -> Chat UI -> Memory Layer -> Agent State -> Agent Runtime -> LLM Decision -> Internal Action -> State Update -> Finish -> Memory Update -> UI

## File Responsibilities

- `app.py`: Streamlit UI, message flow, memory controls, and response display.
- `config.py`: API keys, model defaults, timeout, history path, and recent-message limit.
- `agent/agent.py`: Connects the app and config to the agent runtime.
- `agent/agent_loop.py`: Observe, decide, act, update state, and terminate.
- `agent/agent_state.py`: Current goal, iteration count, status, observations, action results, and trace.
- `agent/decision_schema.py`: Structured JSON contract between the LLM and runtime.
- `llm/groq_client.py`: Raw HTTP/JSON request to Groq and streaming response parsing.
- `memory/chat_memory.py`: Load, validate, save, clear, and select conversation memory.
- `memory/history.json`: Inspectable persistent conversation history.
- `prompts/agent_prompt.py`: Agent decision instructions and allowed actions.
- `prompts/system_prompt.py`: Stage 2 chat prompt kept for comparison.

## Memory Model

Memory is stored information. Context is the selected information sent to the LLM for the current request.

This project uses two memory positions:

- Session memory: `st.session_state`, which keeps the current UI conversation alive while Streamlit is running.
- Persistent memory: `memory/history.json`, which keeps the conversation available after restarting the app.

The context builder does not send the entire history. It sends:

```text
system prompt
+
latest N user/assistant messages
```

This recent-message window is simple and transparent. It can lose older details in long conversations, but it teaches the core context-window problem before adding summarization or semantic retrieval.

Storage capacity is not the same as context-window capacity:

```text
history.json = library
selected recent messages = desk
LLM = person reading what is on the desk
```

## Agent Model

Memory answers: what should be retained?

Agent state answers: what is happening during this current execution?

The Stage 3 loop is:

```text
observe
decide with LLM
parse structured JSON decision
execute internal action
update state
repeat or finish
```

The loop is not the intelligence. The loop is the orchestration mechanism. The LLM decides the next action, and Python enforces the allowed actions, parse rules, state updates, and termination limits.

Allowed Stage 3 actions:

- `ANALYZE`
- `PLAN`
- `CONTINUE`
- `FINISH`

Stage 4 can add `TOOL_CALL` in the same decision schema, but Stage 3 deliberately has no external tools.

## Setup

1. Create a virtual environment.

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies.

   ```powershell
   pip install -r requirements.txt
   ```

3. Create `.env` from `.env.example` and add your key.

   ```powershell
   Copy-Item .env.example .env
   ```

   Then edit `.env`:

   ```text
   GROQ_API_KEY=your-real-key
   ```

4. Run the app.

   ```powershell
   streamlit run app.py
   ```

## What This Teaches

- API keys stay outside source code.
- HTTP requests use headers and JSON bodies.
- Chat models receive messages with roles.
- User and assistant messages must both be stored for continuity.
- Persistent storage can be simple JSON before it needs a database.
- Context construction selects useful memory for the current request.
- Context should be limited instead of blindly sending all history.
- Agent goals live in `AgentState`.
- Agent decisions are structured JSON, not arbitrary text parsing.
- Maximum iterations prevent accidental infinite loops.
- Temperature changes response variety.
- Max tokens limits response length.
- Streaming sends the response piece by piece.
- Errors should be handled clearly.
- Logs should explain failures without exposing secrets.

## Local Checks

```powershell
python -m compileall .
python test_memory.py
python test_agent.py
```

## How To Check Stage 3

1. Run the app:

   ```powershell
   streamlit run app.py
   ```

2. Try a simple goal:

   ```text
   Give me three Python interview topics.
   ```

   Expected: the agent may finish in one iteration.

3. Try a multi-step goal:

   ```text
   Create a plan, check whether it covers Python, LLMs, APIs, memory, and agents, then improve it if something is missing.
   ```

   Expected: open `Agent Execution` and see multiple iterations such as `ANALYZE`, `PLAN`, and `FINISH`.

4. Check memory:

   Open `Inspect memory` in the sidebar. You should see both the user goal and the final assistant answer stored.

5. Check persistence:

   Restart Streamlit and verify the previous conversation reloads from local memory.
