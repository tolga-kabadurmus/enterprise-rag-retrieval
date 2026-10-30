# Configuration

Configuration is validated with immutable Pydantic settings groups. The
application will fail during composition when a required value is missing or
invalid. Copy `dev-envs/enterprise-rag-query.env.example` in the orchestration
repository to its non-committed `.env` counterpart and replace deployment
placeholders.

| Prefix | Purpose | Required values |
| --- | --- | --- |
| `QUERY_API_` | HTTP runtime limits | Defaults are provided |
| `EMBEDDING_SERVICE_` | Shared embedding API and contract validation | Defaults are provided |
| `QDRANT_` | Vector retrieval | `URL`, `COLLECTION_NAME` |
| `RAG_` | Retrieval/context limits | Defaults are provided |
| `LLM_` | Generation provider | `PROVIDER`, `BASE_URL`, `MODEL` |

`QDRANT_API_KEY` and `LLM_API_KEY` use `SecretStr`. `LLM_PROVIDER` accepts
`openai_compatible`, `anthropic`, or `cohere`; native Anthropic and Cohere
selection requires an API key. Never commit actual credentials. The
LLM example endpoint and model are placeholders, not an implemented or selected
deployment.

The Qdrant group also configures `DENSE_VECTOR_NAME` and `VECTOR_SIZE`. Startup
creates one asynchronous Qdrant client for shared use and closes it during
shutdown.

The embedding group configures `URL`, `API_PATH`, `PROVIDER`, HTTP timeouts and
connection limits, request bounds, and the expected model, revision, vector
dimension, and normalization status.
FastAPI startup uses these values to create one pooled HTTPX client and closes
it at shutdown. Live connectivity validation remains unimplemented.

The LLM group additionally configures TLS verification and HTTP connection-pool
limits. `LLM_SYSTEM_PROMPT` configures the short instruction automatically
prepended by the provider-neutral repository base. `LLM_API_KEY` may remain empty only for an unauthenticated compatible
local endpoint; authenticated deployments must provide it. The model and base
URL remain deployment-specific. Startup creates only the SDK client and pooled
HTTP client selected by `LLM_PROVIDER`.
