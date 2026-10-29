# Enterprise RAG — Query

`enterprise-rag-query` is the retrieval and question-answering component of the Enterprise RAG prototype. It exposes a Python **FastAPI** service (served with **Granian**) that embeds a user question, retrieves relevant document chunks from Qdrant, asks an LLM to answer using the retrieved context, and returns the answer **together with source references**.

Its detailed internal design is **to be defined later**, as stated in the project's root [`AGENTS.md`](https://github.com/tolga-kabadurmus/enterprise-rag/blob/main/AGENTS.md). The URL assumes that repository is created under the indicated GitHub account. This README only establishes its minimum functional contract and conceptual boundaries.

## Responsibilities

- Accept a user question through an HTTP API.
- Generate a query embedding compatible with the indexing component's embedding model and vector dimensions.
- Retrieve relevant document chunks and their source metadata from Qdrant.
- Construct an LLM request using the question and retrieved context.
- Return the answer and references to the retrieved sources.
- Handle invalid requests, unavailable providers, empty retrieval results, and operational errors with logging and appropriate API responses.

## Conceptual request flow

```text
      HTTP client
           |
           v
 FastAPI endpoint (Granian)
           |
           v
 Use case / controller
           |
           v
     Query embedding
           |
           v
     Qdrant retrieval
           |
           v
 Retrieved chunks + sources
           |
           v
     LLM generation
           |
           v
   Answer + references
```

## Minimum response contract

The API must include the generated answer and references that identify supporting material (document identifier, source object key, and page/chunk information where available). The exact endpoint route, request/response schema, and HTTP port remain **undecided** and should be specified during implementation; do not treat any example path or port as established.

Answers should be grounded in retrieved context. If no usable source material is retrieved, the service should respond transparently rather than invent document citations. The indexing service remains responsible for producing the metadata needed for citations.

## Architectural conventions

- Implement in Python, using FastAPI for REST endpoints and Granian as the ASGI server.
- Follow Clean Architecture: repositories encapsulate Qdrant, embeddings, and LLM provider access; use cases hold application logic; controllers coordinate multiple use cases where required.
- Instantiate provider-specific adapters through `RepositoryFactory` and inject them into use cases via constructors.
- Each API endpoint should delegate to one use case or one controller, with centralized configuration, validation, logging, and error handling.
- Keep the LLM provider and model configurable. No LLM provider, authentication mechanism, endpoint path, or response-streaming protocol has yet been selected.
- Keep the first working implementation simple before introducing optional features.

## Configuration

Configuration will be provided through environment variables with safe templates in `.env.example` and real secrets in ignored local `.env` files. Expected configuration categories include Qdrant connectivity and collection, embedding provider/model, LLM provider/model/credentials, and basic runtime settings. **Exact variable names remain to be defined.** The embedding configuration must match the indexing pipeline.

## Running with Docker Compose

This repository is a Git submodule of [`enterprise-rag`](https://github.com/tolga-kabadurmus/enterprise-rag), which owns `docker-compose.yaml`. From the parent repository root, the intended startup command is:

```bash
docker compose up -d --build minio qdrant enterprise-rag-query
docker compose logs -f enterprise-rag-query
```

**Planned commands, not yet verified:** These assume the corresponding services are defined in the parent Compose file. The actual API URL, route and exposed port will be documented after the FastAPI application is implemented. The query service depends on content having already been indexed by `enterprise-rag-indexing`.

Stop the stack from the parent repository root:

```bash
docker compose down
```

## Local development

```bash
python3 -m virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This component keeps its own `requirements.txt`, tests, and Dockerfile under `Dockerfiles/`. Use host-side virtual environments for local Python development and Compose for stack integration.

## Testing and documentation

Use unit tests for use cases, controllers, source-reference mapping, and request validation, with fake/mock repositories. Use integration tests for Qdrant and the configured embedding/LLM provider as appropriate, plus an end-to-end check against documents indexed by the indexing component. Do not leak secrets, sensitive document contents, or detailed internal errors in API responses or logs. MkDocs is the selected documentation framework.

## Future enhancements

Potential additions include hybrid search, reranking, response streaming, conversation memory, prompt templates, multi-agent workflows, evaluation pipelines, CI/CD, Kubernetes/OpenShift Helm charts, and an observability dashboard. These are **optional** and should not be represented as already implemented.

This README is **conceptual documentation** for an application that has not yet been fully designed or implemented. Keep it synchronized with the actual routes, schemas, dependencies, and container behavior as development proceeds.
