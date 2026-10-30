# Enterprise RAG Query

This project is the query-side service of Enterprise RAG. The current code is
a runnable FastAPI/Granian foundation with validated environment configuration.
Embedding, vector retrieval, and LLM repository/use-case slices are implemented.
The retrieval controller and endpoint now cover question embedding through
retrieved context; the final LLM-backed query endpoint is not implemented yet.
Dependency failures return sanitized, capability-specific non-500 responses.

## Retrieval endpoint

`POST /v1/retrieval` embeds the question and retrieves matching Qdrant chunks.

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
