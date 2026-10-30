# Architecture

The service will follow the project Clean Architecture conventions:

- REST API handlers validate transport input.
- Controllers coordinate multi-step RAG operations.
- Use cases contain application behavior and depend on one primary repository.
- Capability-specific repositories isolate embedding, vector-store, and LLM I/O.
- `app/main.py` is the composition and ASGI entry point.

The package boundaries, application entry point, and validated settings groups
currently exist. No provider adapters, use cases, controllers, RAG endpoint,
or health endpoint has been implemented.
