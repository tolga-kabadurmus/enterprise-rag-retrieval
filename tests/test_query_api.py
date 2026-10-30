"""Tests for complete and streamed answer endpoints."""

import unittest
from collections.abc import AsyncIterator

import httpx
from fastapi import FastAPI

from app.api.query import router
from app.domain import (
    CitationReference,
    QueryDeltaEvent,
    QueryDoneEvent,
    QueryRequest,
    QueryResponse,
    QuerySourcesEvent,
    QueryStreamEvent,
)


class FakeAnswerQuestionController:
    """Return deterministic complete and streamed answers."""

    def __init__(self) -> None:
        self.source = CitationReference(
            citation_id="S1",
            document_id="employee-handbook",
            source_bucket="documents",
            object_key="hr/employee-handbook.pdf",
            chunk_id="employee-handbook-page-12-chunk-0",
            page_number=12,
            chunk_index=0,
        )

    async def execute(self, request: QueryRequest) -> QueryResponse:
        self.complete_request = request
        return QueryResponse(
            answer="Employees receive 20 days of leave [S1].",
            sources=(self.source,),
        )

    async def stream(
        self,
        request: QueryRequest,
    ) -> AsyncIterator[QueryStreamEvent]:
        self.stream_request = request
        yield QueryDeltaEvent(delta="Employees receive 20 days")
        yield QuerySourcesEvent(sources=(self.source,))
        yield QueryDoneEvent()


class QueryEndpointTests(unittest.IsolatedAsyncioTestCase):
    """Verify JSON and SSE response serialization."""

    @staticmethod
    def create_test_app() -> FastAPI:
        application = FastAPI()
        application.include_router(router)
        return application

    async def test_returns_complete_answer_and_citations(self) -> None:
        application = self.create_test_app()
        controller = FakeAnswerQuestionController()
        application.state.answer_question_controller = controller

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/query",
                json={"question": "How much leave do employees receive?"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sources"][0]["citation_id"], "S1")
        self.assertEqual(
            controller.complete_request.question,
            "How much leave do employees receive?",
        )

    async def test_streams_deltas_then_sources_and_done(self) -> None:
        application = self.create_test_app()
        controller = FakeAnswerQuestionController()
        application.state.answer_question_controller = controller

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/query/stream",
                json={"question": "How much leave do employees receive?"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        self.assertLess(response.text.index("event: delta"), response.text.index("event: sources"))
        self.assertLess(response.text.index("event: sources"), response.text.index("event: done"))
        self.assertIn('"citation_id":"S1"', response.text)

    async def test_reports_missing_startup_dependency(self) -> None:
        application = self.create_test_app()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/query",
                json={"question": "Question"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "query answering is not ready"})


if __name__ == "__main__":
    unittest.main()
