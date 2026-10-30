# OpenAPI and interactive API documentation

The retrieval service publishes an OpenAPI 3 contract generated directly from
its FastAPI routes and Pydantic models. The contract documents stable operation
IDs, tags, request and response schemas, dependency failures, field
descriptions, and example payloads.

With the local Compose service running on its default host port:

| Resource | URL |
| --- | --- |
| Swagger UI | <http://localhost:8000/docs> |
| ReDoc | <http://localhost:8000/redoc> |
| OpenAPI JSON | <http://localhost:8000/openapi.json> |

The documented operations are:

| Method and path | Purpose |
| --- | --- |
| `POST /v1/query` | Return one complete grounded answer with citations. |
| `POST /v1/query/stream` | Stream answer deltas, then citations and completion. |
| `POST /v1/retrieval` | Return reranked, citation-ready source chunks. |
| `POST /v1/embeddings` | Return the dense vector for one question. |

The query-vectorization route is diagnostic, while the retrieval route returns
context without calling the LLM. The complete query route uses JSON. The stream
route uses Server-Sent Events in `delta`, `sources`, and `done` order.
Authentication and authorization are outside the project scope, so the OpenAPI
contract intentionally defines no security scheme.

## Complete query contract

`POST /v1/query` accepts the documented `QueryRequest` body and returns a
`QueryResponse`. Each `CitationReference` has a response-scoped identifier such
as `S1`, plus document, bucket, object-key, page, and chunk metadata. Swagger UI
includes complete request, answer, citation, validation, rate-limit, provider,
readiness, and timeout examples.

## Streaming query contract

`POST /v1/query/stream` uses the `text/event-stream` media type. Each SSE frame
contains an `event` line and a JSON `data` line. The OpenAPI operation includes
an `x-sse-event-schemas` extension because OpenAPI does not natively model a
sequence of differently shaped SSE events.

| Event | Payload | Ordering |
| --- | --- | --- |
| `delta` | A non-empty answer fragment. | One or more before sources. |
| `sources` | The final response-scoped citations. | Once after all deltas. |
| `error` | A sanitized code and safe detail. | Reserved for the next mid-stream error-handling change; not emitted yet. |
| `done` | A terminal marker. | Once at the end of a successful stream. |

HTTP `422`, `429`, `502`, `503`, and `504` responses apply when the failure is
known before streaming starts. Once response headers have been sent, HTTP status
cannot change; the reserved `error` event will cover that later lifecycle stage.

The checked contract tests inspect `/openapi.json` to prevent accidental loss
of endpoint descriptions, response schemas, error statuses, or stable
operation IDs.
