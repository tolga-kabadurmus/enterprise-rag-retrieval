"""Tests for the temporary query-vectorization endpoint."""

import unittest
from types import SimpleNamespace

import httpx
from fastapi import FastAPI

from app.api.embeddings import router
from app.common.settings.embedding import EmbeddingServiceSettings
from app.repository.embedding_service import EmbeddingVector, SparseEmbedding


class FakeQueryEmbeddingUseCase:
    """Return a deterministic vector and record the submitted question."""

    def __init__(self) -> None:
        self.questions: list[str] = []

    async def execute(self, question: str) -> EmbeddingVector:
        self.questions.append(question)
        return EmbeddingVector(
            dense=(0.6, 0.8),
            sparse=SparseEmbedding(indices=(3,), values=(1.0,)),
        )


class QueryEmbeddingEndpointTests(unittest.IsolatedAsyncioTestCase):
    """Verify endpoint delegation, response shape, and safe failures."""

    @staticmethod
    def create_test_app() -> FastAPI:
        """Create the route without starting infrastructure dependencies."""

        application = FastAPI()
        application.include_router(router)
        return application

    async def test_vectorizes_one_user_question(self) -> None:
        application = self.create_test_app()
        usecase = FakeQueryEmbeddingUseCase()
        application.state.query_embedding_usecase = usecase
        application.state.settings = SimpleNamespace(
            embedding=EmbeddingServiceSettings(expected_dimension=2)
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(
                app=application,
                raise_app_exceptions=False,
            ),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/embeddings",
                json={"question": "What is the leave policy?"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(usecase.questions, ["What is the leave policy?"])
        self.assertEqual(
            response.json(),
            {
                "model": "intfloat/multilingual-e5-small",
                "revision": "main",
                "dimension": 2,
                "normalized": True,
                "embeddings": [[0.6, 0.8]],
            },
        )

    async def test_rejects_blank_question(self) -> None:
        application = self.create_test_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/embeddings",
                json={"question": " "},
            )

        self.assertEqual(response.status_code, 422)

    async def test_reports_missing_startup_dependency(self) -> None:
        application = self.create_test_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/embeddings",
                json={"question": "Question"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "query embedding is not ready"})


if __name__ == "__main__":
    unittest.main()
