"""Tests for sanitized query-pipeline error responses."""

import unittest

import cohere
import httpx
from fastapi import FastAPI
from qdrant_client.http.exceptions import UnexpectedResponse

from app.middleware import UnhandledExceptionMiddleware
from app.repository.embedding_service.exceptions import EmbeddingServiceRequestError
from app.repository.reranking.exceptions import RerankingError


class ExceptionMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    """Verify capability failures never leak as HTTP 500 responses."""

    async def _request_for(self, error: Exception) -> httpx.Response:
        application = FastAPI()
        application.add_middleware(UnhandledExceptionMiddleware)

        @application.get("/failure")
        async def failure() -> None:
            raise error

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(
                app=application,
                raise_app_exceptions=False,
            ),
            base_url="http://test",
        ) as client:
            return await client.get("/failure")

    async def test_maps_embedding_rejection_to_bad_gateway(self) -> None:
        response = await self._request_for(
            EmbeddingServiceRequestError(422, "private downstream detail")
        )

        self.assertEqual(response.status_code, 502)
        self.assertNotIn("private downstream detail", response.text)

    async def test_maps_embedding_timeout_to_gateway_timeout(self) -> None:
        request = httpx.Request("POST", "http://embedding.test/v1/embeddings")
        response = await self._request_for(httpx.ReadTimeout("timeout", request=request))

        self.assertEqual(response.status_code, 504)

    async def test_maps_retrieval_failure_to_service_unavailable(self) -> None:
        response = await self._request_for(
            UnexpectedResponse(404, "collection missing", b"secret", None)
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "vector retrieval is unavailable"})

    async def test_maps_reranking_failure_to_service_unavailable(self) -> None:
        response = await self._request_for(RerankingError("private model detail"))

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "reranking is unavailable"})

    async def test_maps_llm_rate_limit(self) -> None:
        response = await self._request_for(
            cohere.TooManyRequestsError({"message": "private provider detail"})
        )

        self.assertEqual(response.status_code, 429)
        self.assertEqual(
            response.json(),
            {"detail": "LLM provider rate limit exceeded"},
        )

    async def test_maps_unknown_failure_without_http_500(self) -> None:
        response = await self._request_for(RuntimeError("sensitive detail"))

        self.assertEqual(response.status_code, 503)
        self.assertNotIn("sensitive detail", response.text)
