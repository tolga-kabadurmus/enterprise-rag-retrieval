# Configuration

Configuration is validated with immutable Pydantic settings groups. The
application will fail during composition when a required value is missing or
invalid. Copy `dev-envs/enterprise-rag-query.env.example` in the orchestration
repository to its non-committed `.env` counterpart and replace deployment
placeholders.

| Prefix | Purpose | Required values |
| --- | --- | --- |
| `QUERY_API_` | HTTP runtime limits | Defaults are provided |
| `EMBEDDING_SERVICE_` | Shared embedding API | `BASE_URL` |
| `QDRANT_` | Vector retrieval | `URL`, `COLLECTION_NAME` |
| `RAG_` | Retrieval/context limits | Defaults are provided |
| `LLM_` | Generation provider | `PROVIDER`, `BASE_URL`, `MODEL` |

`QDRANT_API_KEY` and `LLM_API_KEY` use `SecretStr` and must be supplied by the
deployment environment when required. Never commit actual credentials. The
LLM example endpoint and model are placeholders, not an implemented or selected
deployment.

Provider clients, startup wiring, and external connectivity are not part of
this stage and remain unimplemented.
