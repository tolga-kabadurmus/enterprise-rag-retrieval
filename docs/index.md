# Enterprise RAG Query

This project is the query-side service of Enterprise RAG. The current code is
a runnable FastAPI/Granian foundation with validated environment configuration.
Embedding, vector retrieval, reranking, and LLM answer generation are
implemented. The complete query endpoint returns JSON, while the streaming
query endpoint uses Server-Sent Events and emits citations after answer deltas.
Both paths call the configured LLM in streaming mode; the JSON endpoint buffers
the received deltas before responding.
Dependency failures return sanitized, capability-specific non-500 responses.

## Query endpoints

Return one complete answer and response-scoped citations:

```bash
curl -X POST http://localhost:8000/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the company leave policy?"}'
```

Stream `delta`, `sources`, and `done` events:

```bash
curl -N -X POST http://localhost:8000/v1/query/stream \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the company leave policy?"}'
```

## Retrieval endpoint

`POST /v1/retrieval` obtains aligned dense and sparse query vectors, prefetches
both named Qdrant vector indexes, fuses their rankings with RRF, and returns the
candidate chunks. FlashRank then reranks those candidates and the endpoint
returns the final top-k. Its request and response schema did not change.

```bash
curl -X POST http://localhost:8000/v1/retrieval \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the company leave policy?"}'
```

## Temporary query-vectorization endpoint

`POST /v1/embeddings` sends one user question through the existing
query-embedding use case and shared embedding-service network hop. It is a
temporary development endpoint and may be removed after end-to-end verification.

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the company leave policy?"}'
```

## Development

```bash
python3 -m virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-docs.txt
python -m unittest discover -s tests
mkdocs build --strict
```

See [Architecture](architecture.md) for the intended layer boundaries.
See [Configuration](configuration.md) for the validated environment contract.
