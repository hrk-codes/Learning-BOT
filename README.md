# Stage 5 RAG Agent

A small Streamlit AI agent built from first principles with Groq, conversation memory,
a controlled tool runtime, and a local retrieval-augmented knowledge layer.

Stage 5 does not create a separate PDF chatbot. It adds knowledge retrieval as one
decision available to the existing agent:

```text
User goal
  -> Agent decision
     -> answer directly
     -> RETRIEVE_KNOWLEDGE
     -> TOOL_CALL
     -> continue
     -> finish
```

## Core Distinctions

```text
Memory -> information from the conversation
RAG    -> evidence from indexed reference documents
Tools  -> capabilities that perform actions
LLM    -> language generation and structured decisions
```

RAG does not train or fine-tune the LLM. It retrieves relevant text and deliberately
places that evidence into the model context for one answer.

## Two Pipelines

Ingestion happens when a PDF is added:

```text
PDF -> validate -> save original -> parse pages -> chunk -> metadata
    -> embed once -> store vectors
```

Retrieval happens when the agent requests document knowledge:

```text
Question -> normalize -> embed query -> cosine search -> top-k chunks
         -> grounded observation -> agent -> answer + sources
```

Document embeddings are not recomputed for every question. Query time creates one
query vector and compares it with the stored chunk vectors.

## Project Structure

```text
app.py                         Streamlit chat, knowledge UI, and debug views
agent/                         State, decisions, routing, and bounded agent loop
memory/                        Conversation memory
tools/                         Calculator, weather, search, registry, and permissions
rag/ingestion/                 PDF validation, parsing, cleaning, and chunking
rag/embeddings/embedder.py     Local text-to-vector boundary
rag/storage/vector_store.py    Inspectable JSON vectors and cosine search
rag/retrieval/retriever.py     Query processing and ranked structured results
rag/context/context_builder.py Grounded evidence and source construction
rag/pipeline.py                RAG orchestration and document lifecycle
documents/raw/                 Uploaded source-of-truth PDFs, ignored by Git
documents/sample/              Controlled PDFs for testing
vector_store/                  Generated retrieval index, ignored by Git
```

The implementation uses:

- `pypdf` for text-based PDF extraction. Scanned PDFs need future OCR support.
- `sentence-transformers/all-MiniLM-L6-v2` for local 384-dimensional embeddings.
- A JSON vector store with explicit cosine similarity for a small learning corpus.
- Deterministic 1,200-character chunks with 200-character overlap by default.

The JSON index is intentionally simple and inspectable. FAISS, Chroma, Qdrant,
Weaviate, Pinecone, or pgvector become preferable when corpus size, filtering,
concurrency, durability, or distributed deployment outgrow a local learning system.

## Setup

```powershell
cd "C:\Users\hrkgh\Agent learn\BOT 1"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Put the Groq key in `.env`:

```text
GROQ_API_KEY=your-real-key
```

Start the app:

```powershell
streamlit run app.py
```

The first PDF indexing run downloads the local embedding model. Later runs reuse the
cached model and stored document vectors.

The project disables Streamlit's automatic source watcher because it probes optional
Transformers vision modules and produces misleading `torchvision` errors in a text-only
app. Restart Streamlit manually after changing Python files.

## How To Check Stage 5

1. Open `http://localhost:8501` and confirm the title is `Stage 5 RAG Agent`.

2. In `Knowledge Base`, upload `documents/sample/employee-handbook.pdf` and click
   `Index PDF`. Verify the status passes through upload, parsing, chunking, embedding,
   indexing, and completion. The document should show `indexed`, one page, at least one
   chunk, an embedding count, content hash, version, and page metadata.

3. Ask:

   ```text
   According to the employee handbook, how many unused leave days can I take into next year?
   ```

   Expected: the trace contains `RETRIEVE_KNOWLEDGE`; the answer says `10`; and the
   runtime appends `employee-handbook.pdf - page 1` under `Sources`.

4. Open `RAG Debug`. Verify it shows the retrieval query, chunk ID, source filename,
   page number, similarity score, and retrieval latency.

5. Ask an unsupported document question:

   ```text
   According to the handbook, where can employees park their cars?
   ```

   Expected: no chunk should pass the configured similarity threshold, or the agent
   should state that the available documents do not provide enough evidence.

6. Upload `documents/sample/internal-api-guide.pdf`, then ask:

   ```text
   According to the uploaded API guide, what authentication method is required?
   ```

   Expected: `OAuth 2.0 bearer-token authentication`, cited to the API guide.

7. Upload `documents/sample/product-manual.pdf`, enable `calculator.evaluate`, and ask:

   ```text
   According to the product manual, what is the operating temperature range,
   and convert the maximum Celsius temperature to Fahrenheit?
   ```

   Expected trace:

   ```text
   RETRIEVE_KNOWLEDGE -> TOOL_CALL calculator.evaluate -> FINISH
   ```

   Expected answer: `5 to 40 C`, `104 F`, plus the product manual source.

8. Ask `Explain recursion.` Expected: the agent can answer without retrieval.

9. Test lifecycle controls. Use `Re-index`, inspect the unchanged document metadata,
   then use `Delete`. Verify the document count, chunks, embeddings, and original file
   are removed together.

10. Run all local tests:

    ```powershell
    python test_memory.py
    python test_agent.py
    python test_tools.py
    python test_rag.py
    ```

## Debugging RAG

If an answer is wrong, inspect the layers in order:

```text
PDF parsing -> chunks -> embeddings -> retrieved results -> context -> final answer
```

- Retrieval failure: the correct fact exists in a PDF, but its chunk is absent from
  `RAG Debug`. Investigate parsing, chunk boundaries, query wording, scores, `top_k`,
  and the minimum similarity threshold.
- Generation failure: the correct chunk appears in `RAG Debug`, but the answer is
  wrong. Investigate grounding instructions, conflicting evidence, and model behavior.

Do not change the LLM prompt first when the retriever never found the evidence.

For the deeper architecture, trade-offs, and security model, read
[`docs/STAGE5_RAG.md`](docs/STAGE5_RAG.md).
