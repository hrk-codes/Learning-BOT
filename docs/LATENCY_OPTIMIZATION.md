# Free Latency Optimization

This project keeps Groq as its only provider. The optimization changes the route,
model size, prompt size, and output budget; it does not bypass quotas or add paid
infrastructure.

## Model Routing

| Work | Model | Why |
| --- | --- | --- |
| Simple conversational question | `GROQ_FAST_MODEL` | One streamed request gives the quickest first visible token. |
| Planning, replanning, task execution, and Stage 10 specialists | `GROQ_FAST_MODEL` | These are bounded intermediate steps. |
| Completed Stage 7/9 answer and Stage 10 manager synthesis | `GROQ_FINAL_MODEL` | The larger model is used once, after verified work is available. |

Defaults are `openai/gpt-oss-20b` for fast work and `openai/gpt-oss-120b` for final
synthesis. Set both values to the same model to return to a single-model setup.

## Fast Path

`latency/fast_path.py` sends a simple explanation or normal chat question directly to
one streamed completion. It keeps recent conversation messages, but deliberately
bypasses this path for documents/RAG, current information, tools, memory commands,
side effects, and requests needing a plan. Those requests retain the established
guarded workflows.

## What To Measure

Each completed response includes a **Latency Debug** panel:

- `request_gate_wait_seconds`: waiting behind another local request, retained to protect rate limits.
- `time_to_first_token_seconds`: when the first visible streamed text arrived.
- `total_seconds`: full provider request time.
- `retry_count`: only rate-limit retries.
- `provider_usage`: token counters and cached-token count, when Groq supplies them.

The panel contains no API key, prompt, memory record, document excerpt, or generated
content. A short simple question should normally show one fast-model call. Complex
workflows intentionally show multiple calls because their safety and quality checks are
separate operations.

## Configuration

```dotenv
GROQ_FAST_MODEL=openai/gpt-oss-20b
GROQ_FINAL_MODEL=openai/gpt-oss-120b
GROQ_SIMPLE_MAX_TOKENS=384
GROQ_FAST_MAX_TOKENS=512
PLANNER_MIN_OUTPUT_TOKENS=640
PLANNER_MAX_TASKS=5
```

Keep stable system prompts and schemas first in each message list. Put conversation,
memory, RAG excerpts, and task state afterwards. This preserves eligibility for Groq's
automatic prompt cache when the provider supports it.
