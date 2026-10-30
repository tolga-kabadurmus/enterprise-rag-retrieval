"""Tests for the FastAPI application composition."""

import os
import unittest
from unittest.mock import patch

from fastapi import FastAPI

from app.main import app, create_app
from app.controller import AnswerQuestionController, RetrievalController
from app.repository.llm import AnthropicLLMRepository, CohereLLMRepository
from app.usecase import QueryEmbeddingUseCase, RetrieveChunksUseCase
from app.usecase import GenerateAnswerUseCase, RerankChunksUseCase


VALID_ENV = {
    "EMBEDDING_SERVICE_URL": "http://embedding:8081",
    "QDRANT_URL": "http://qdrant:6333",
    "QDRANT_COLLECTION_NAME": "chunks",
    "LLM_PROVIDER": "openai_compatible",
    "LLM_BASE_URL": "https://llm.example.test/v1",
    "LLM_MODEL": "test-model",
}


class FakeQdrantClient:
    """Minimal startup client that avoids external connectivity."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class FakeRanker:
    """Avoid model downloads while testing application composition."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def rerank(self, request: object) -> list[dict[str, object]]:
        return []


class ApplicationScaffoldTests(unittest.TestCase):
    """Verify that the ASGI application can be imported and created."""

    def test_module_exposes_fastapi_application(self) -> None:
        self.assertIsInstance(app, FastAPI)

    def test_application_factory_sets_service_identity(self) -> None:
        application = create_app()
        self.assertEqual(application.title, "Enterprise RAG Retrieval Service")
        self.assertEqual(application.version, "0.1.0")
        self.assertIn(
            "/v1/embeddings",
            {route.path for route in application.routes},
        )
        self.assertIn(
            "/v1/retrieval",
            {route.path for route in application.routes},
        )
        self.assertIn("/v1/query", {route.path for route in application.routes})
        self.assertIn(
            "/v1/query/stream",
            {route.path for route in application.routes},
        )


class ApplicationLifespanTests(unittest.IsolatedAsyncioTestCase):
    """Verify startup injection and asynchronous client cleanup."""

    async def test_wires_query_embedding_usecase_and_closes_client(self) -> None:
        application = create_app()

        with (
            patch.dict(os.environ, VALID_ENV, clear=True),
            patch("app.main.AsyncQdrantClient", FakeQdrantClient),
            patch("app.main.Ranker", FakeRanker),
        ):
            async with application.router.lifespan_context(application):
                usecase = application.state.query_embedding_usecase
                repository = application.state.embedding_repository
                self.assertIsInstance(usecase, QueryEmbeddingUseCase)
                self.assertFalse(repository._httpx_client.is_closed)
                vector_repository = application.state.vector_repository
                self.assertIs(
                    application.state.vector_client,
                    vector_repository._client,
                )
                self.assertIsInstance(
                    application.state.retrieve_chunks_usecase,
                    RetrieveChunksUseCase,
                )
                self.assertIsInstance(
                    application.state.retrieval_controller,
                    RetrievalController,
                )
                self.assertIsInstance(
                    application.state.rerank_chunks_usecase,
                    RerankChunksUseCase,
                )
                llm_repository = application.state.llm_repository
                llm_http_client = application.state.llm_http_client
                self.assertIs(
                    application.state.llm_client,
                    llm_repository._client,
                )
                self.assertIsInstance(
                    application.state.generate_answer_usecase,
                    GenerateAnswerUseCase,
                )
                self.assertIsInstance(
                    application.state.answer_question_controller,
                    AnswerQuestionController,
                )
                self.assertFalse(llm_http_client.is_closed)

        self.assertTrue(repository._httpx_client.is_closed)
        self.assertTrue(vector_repository._closed)
        self.assertTrue(application.state.reranking_repository._closed)
        self.assertTrue(llm_repository._closed)
        self.assertTrue(llm_http_client.is_closed)

    async def test_selects_native_llm_repository_from_environment(self) -> None:
        providers = (
            ("anthropic", AnthropicLLMRepository),
            ("cohere", CohereLLMRepository),
        )
        for provider, expected_type in providers:
            application = create_app()
            environment = VALID_ENV | {
                "LLM_PROVIDER": provider,
                "LLM_BASE_URL": f"https://{provider}.example.test",
                "LLM_API_KEY": "secret",
            }
            with self.subTest(provider=provider):
                with (
                    patch.dict(os.environ, environment, clear=True),
                    patch("app.main.AsyncQdrantClient", FakeQdrantClient),
                    patch("app.main.Ranker", FakeRanker),
                ):
                    async with application.router.lifespan_context(application):
                        repository = application.state.llm_repository
                        http_client = application.state.llm_http_client
                        self.assertIsInstance(repository, expected_type)

                self.assertTrue(repository._closed)
                self.assertTrue(http_client.is_closed)


if __name__ == "__main__":
    unittest.main()
