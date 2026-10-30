# Enterprise RAG Query

This project is the query-side service of Enterprise RAG. The current code is
only a runnable FastAPI/Granian foundation; retrieval, embedding, LLM calls,
configuration, and public query endpoints are not implemented yet.

## Development

```bash
python3 -m virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-docs.txt
python -m unittest discover -s tests
mkdocs build --strict
```

See [Architecture](architecture.md) for the intended layer boundaries.
