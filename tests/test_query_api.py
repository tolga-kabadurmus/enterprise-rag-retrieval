"""Tests for complete and streamed answer endpoints."""

import asyncio
import unittest
from collections.abc import AsyncIterator

import httpx
from fastapi import FastAPI

from app.api.query import _stream_events, router
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


class FailingStreamController(FakeAnswerQuestionController):
    """Fail after a delta with details that must not reach the client."""

    async def stream(
        self,
        request: QueryRequest,
    ) -> AsyncIterator[QueryStreamEvent]:
        yield QueryDeltaEvent(delta="Partial answer")
        raise RuntimeError("secret provider response and internal path")


class BlockingStreamController(FakeAnswerQuestionController):
    """Expose whether cancellation closes the upstream stream."""

    def __init__(self) -> None:
        super().__init__()
        self.closed = False
        self.release = asyncio.Event()

    async def stream(
        self,
        request: QueryRequest,
    ) -> AsyncIterator[QueryStreamEvent]:
        try:
            yield QueryDeltaEvent(delta="First token")
            await self.release.wait()
            yield QueryDoneEvent()
        finally:
            self.closed = True


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

    async def test_sanitizes_failure_after_streaming_starts(self) -> None:
        application = self.create_test_app()
        application.state.answer_question_controller = FailingStreamController()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/query/stream",
                json={"question": "Question"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: delta", response.text)
        self.assertIn("event: error", response.text)
        self.assertIn('"detail":"service temporarily unavailable"', response.text)
        self.assertNotIn("secret provider response", response.text)
        self.assertLess(
            response.text.index("event: error"),
            response.text.index("event: done"),
        )

    async def test_client_cancellation_closes_upstream_stream(self) -> None:
        controller = BlockingStreamController()
        payload = QueryRequest(question="Question")
        stream = _stream_events(controller, payload)

        first = await anext(stream)
        self.assertIn("event: delta", first)

        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending

        self.assertTrue(controller.closed)


if __name__ == "__main__":
    unittest.main()
