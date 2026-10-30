"""FastAPI application entry point for the query service scaffold."""

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Create the query service application.

    Returns:
        A FastAPI application ready to be served by an ASGI server.
    """

    return FastAPI(
        title="Enterprise RAG Query",
        version="0.1.0",
        description="Query service scaffold; RAG endpoints are not implemented yet.",
    )


app = create_app()
