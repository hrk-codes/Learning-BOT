# Stage 5 RAG Architecture

## What RAG Is

Retrieval-Augmented Generation retrieves relevant external information and supplies
it to an LLM as evidence during generation. The source document does not change the
model weights.

```text
RAG != training
RAG != fine-tuning
RAG != conversation memory
RAG != a vector database
```

A vector store is one implementation component. RAG is the complete ingestion,
retrieval, context construction, generation, source tracing, and evaluation system.

## Why PDFs Need Parsing

A PDF describes a visual page, not a simple ordered text file. It may contain columns,
tables, repeated headers, images, scanned pages, or text drawing instructions in an
unexpected order. Stage 5 uses `pypdf` for text-based PDFs and keeps page boundaries.
OCR, table extraction, and layout-aware parsing are future extensions.

## Chunking

A whole document is usually too broad and expensive to retrieve as one unit. Chunking
creates smaller candidates that can be ranked independently.

- Smaller chunks improve precision but can lose surrounding context.
- Larger chunks preserve context but add unrelated text and token cost.
- Overlap protects facts near boundaries but duplicates storage and results.
- No universal chunk size is correct for every document type.

Stage 5 uses a deterministic character window with a small overlap and prefers nearby
paragraph, sentence, or whitespace boundaries. A future chunker can implement paragraph,
section, semantic, or parent-child strategies behind the same interface.

## Embeddings and Search

An embedding model converts text into a numerical vector. Semantically related text
tends to occupy nearby regions of the model's vector space. The same compatible model
must encode both document chunks and user queries.

```text
Embedding model -> retrieval vectors
LLM             -> decisions and language generation
```

Stage 5 uses `sentence-transformers/all-MiniLM-L6-v2` locally. It is small enough for a
learning project, requires no second API key, and maps text into 384 dimensions. The
trade-offs are an initial model download, local CPU/memory usage, English-oriented
quality, and weaker retrieval than larger or domain-specific models.

The local store ranks chunks with cosine similarity. Dot product is equivalent for
normalized vectors. Euclidean distance is another valid metric, but the chosen metric
must match the embedding model and index configuration.

## Top-K and Context

`top_k` is the maximum number of candidate chunks returned. Too few can miss evidence;
too many can add noise, tokens, latency, and contradictory context.

The context builder also has a character budget. Retrieval does not automatically mean
the model sees a chunk; the application deliberately serializes selected chunk text and
metadata into the next agent observation.

Reranking is skipped initially. A future design can retrieve 20 candidates and use a
cross-encoder or another reranker to choose the best 5 before context construction.

## Keyword, Semantic, and Hybrid Search

- Keyword search is strong for IDs, exact names, product codes, and legal clauses.
- Semantic search is strong for paraphrases and conceptual natural-language questions.
- Hybrid search combines both when exact identifiers and natural language coexist.

## Vector Store Alternatives

| Option | Prefer it when | Main trade-off |
|---|---|---|
| JSON brute force | Learning and tiny local corpora | Not suitable for large or concurrent workloads |
| FAISS | Fast local vector search | Metadata and persistence need application code |
| Chroma | Local prototypes with metadata filtering | Additional database lifecycle |
| Qdrant | Production filtering and scalable search | Service deployment and operations |
| Weaviate | Semantic data platform features | More platform complexity |
| Pinecone | Fully managed vector infrastructure | Vendor cost and external dependency |
| pgvector | Vectors belong beside relational data | PostgreSQL tuning and index operations |

Stage 5 chooses JSON so vectors, metadata, ranking, deletion, and failure modes remain
visible. Replace it before the corpus becomes large.

## Source Tracing

Every chunk preserves `document_id`, filename, source, version, page number, content
hash, indexing time, and a placeholder `user_id`. Retrieval returns text, score, and
metadata together. The runtime appends sources from actual retrieval metadata rather
than trusting the LLM to invent citations.

The original PDF remains the source of truth. The vector index is a replaceable
retrieval representation used for search, re-indexing, and deletion.

## Security Boundary

Uploaded documents are untrusted data. A PDF can contain text such as `Ignore previous
instructions` or `Call a tool`. Retrieved text is evidence only; it cannot override the
system prompt, permission manager, action router, or tool schemas.

Stage 5 is local and single-user, but metadata includes `user_id` and retrieval filters
on it. A production system must authenticate users, authorize each document before
retrieval, scan uploads, isolate tenants, encrypt sensitive data, and audit access.

## Future Upgrades

- OCR and layout-aware PDF/table parsing
- DOCX, Markdown, HTML, CSV, and web loaders
- section-aware or parent-child chunking
- query rewriting for ambiguous follow-up questions
- hybrid retrieval and metadata filters
- reranking and context compression
- retrieval and generation evaluations
- document versions, background ingestion, and approval workflows
- FAISS, Qdrant, Pinecone, or pgvector storage
- MCP and production access control

Knowledge graphs, Graph RAG, multi-hop retrieval, distributed vector infrastructure,
and enterprise IAM remain intentionally outside this first Stage 5 implementation.
