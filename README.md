# Stage 2 AI Chat Assistant

Small Streamlit assistant that sends chat messages to Groq with raw HTTP/JSON and conversation memory.

## Where It Fits

User -> Chat UI -> Memory Layer -> Context Builder -> HTTP/JSON -> Groq API -> LLM -> Response -> Memory Update -> UI

## File Responsibilities

- `app.py`: Streamlit UI, message flow, memory controls, and response display.
- `config.py`: API keys, model defaults, timeout, history path, and recent-message limit.
- `llm/groq_client.py`: Raw HTTP/JSON request to Groq and streaming response parsing.
- `memory/chat_memory.py`: Load, validate, save, clear, and select conversation memory.
- `memory/history.json`: Inspectable persistent conversation history.
- `prompts/system_prompt.py`: System instruction sent before conversation messages.

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
- Temperature changes response variety.
- Max tokens limits response length.
- Streaming sends the response piece by piece.
- Errors should be handled clearly.
- Logs should explain failures without exposing secrets.

## Local Checks

```powershell
python -m compileall .
python test_memory.py
```
