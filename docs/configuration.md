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
| `RERANKING_` | Local FlashRank model and inference bounds | Defaults are provided |
| `LLM_` | Generation provider | `PROVIDER`, `BASE_URL`, `MODEL` |

`QDRANT_API_KEY` and `LLM_API_KEY` use `SecretStr`. `LLM_PROVIDER` accepts
`openai_compatible`, `anthropic`, or `cohere`; native Anthropic and Cohere
selection requires an API key. Never commit actual credentials. The
LLM example endpoint and model are placeholders, not an implemented or selected
deployment.

The Qdrant group also configures `DENSE_VECTOR_NAME`, `SPARSE_VECTOR_NAME`, and
`VECTOR_SIZE`. Startup
creates one asynchronous Qdrant client for shared use and closes it during
shutdown.

The embedding group configures `URL`, `API_PATH`, `PROVIDER`, HTTP timeouts and
connection limits, request bounds, and the expected model, revision, vector
dimension, normalization status, and `EXPECTED_SPARSE_MODEL`.
FastAPI startup uses these values to create one pooled HTTPX client and closes
it at shutdown. Live connectivity validation remains unimplemented.

`RAG_HYBRID_PREFETCH_LIMIT` controls how many candidates each dense and sparse
branch supplies to RRF. `RAG_CANDIDATE_K` controls how many fused candidates are
sent to FlashRank, while `RAG_TOP_K` controls how many reranked chunks are
returned. The required ordering is prefetch limit >= candidate-k >= top-k.
`RAG_SCORE_THRESHOLD`, when set, is applied to the fused Qdrant score rather than
to either branch independently.

The reranking group configures `PROVIDER`, `MODEL_NAME`, `CACHE_DIR`,
`MAX_LENGTH`, and `MAX_CONCURRENCY`. The default provider is FlashRank with the
small `ms-marco-TinyBERT-L-2-v2` cross-encoder. Startup downloads the model when
it is absent from the cache, loads it once, and fails rather than accepting
traffic with an unavailable reranker. `/tmp/flashrank` is an ephemeral default;
deployments may point `CACHE_DIR` at a prepared or persistent model location.
The tiny default model is optimized for English and `MAX_LENGTH=128` truncates
long query/chunk pairs. Multilingual deployments can select FlashRank's larger
`ms-marco-MultiBERT-L-12` model after validating its latency and memory cost.

The LLM group additionally configures TLS verification and HTTP connection-pool
limits. `LLM_SYSTEM_PROMPT` configures the short instruction automatically
prepended by the provider-neutral repository base. `LLM_API_KEY` may remain empty only for an unauthenticated compatible
local endpoint; authenticated deployments must provide it. The model and base
URL remain deployment-specific. Startup creates only the SDK client and pooled
HTTP client selected by `LLM_PROVIDER`.
