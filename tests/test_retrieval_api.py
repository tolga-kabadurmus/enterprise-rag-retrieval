"""Tests for the chunk retrieval endpoint."""

import unittest

import httpx
from fastapi import FastAPI

from app.api.retrieval import router
from app.domain import RetrievedChunk, SourceReference


class FakeRetrievalController:
    """Return a fixed retrieved chunk."""

    async def execute(self, question: object) -> tuple[RetrievedChunk, ...]:
        self.question = question
        return (
            RetrievedChunk(
                text="Leave policy context",
                score=0.9,
                source=SourceReference(
                    document_id="doc-1",
                    source_bucket="documents",
                    object_key="handbook.pdf",
                    chunk_id="chunk-1",
                    page_number=2,
                ),
            ),
        )


class RetrievalEndpointTests(unittest.IsolatedAsyncioTestCase):
    """Verify endpoint delegation and response serialization."""

    @staticmethod
    def create_test_app() -> FastAPI:
        application = FastAPI()
        application.include_router(router)
        return application

    async def test_retrieves_chunks_for_question(self) -> None:
        application = self.create_test_app()
        controller = FakeRetrievalController()
        application.state.retrieval_controller = controller

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/retrieval",
                json={"question": "What is the leave policy?"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(controller.question.question, "What is the leave policy?")
        self.assertEqual(response.json()["chunks"][0]["source"]["page_number"], 2)

    async def test_reports_missing_startup_dependency(self) -> None:
        application = self.create_test_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/retrieval",
                json={"question": "Question"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "retrieval is not ready"})
