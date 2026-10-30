"""Unit tests for the shared embedding-service HTTP adapter."""

import json
import unittest

import httpx

from app.common.settings.embedding import EmbeddingProvider, EmbeddingServiceSettings
from app.repository.embedding_service import (
    EmbeddingRepositoryFactory,
    EnterpriseEmbeddingServiceRepository,
)


class EmbeddingServiceRepositoryTests(unittest.IsolatedAsyncioTestCase):
    """Verify factory selection, query payloads, validation, and cleanup."""

    async def test_factory_creates_registered_httpx_repository(self) -> None:
        client = httpx.AsyncClient(
            base_url="http://embedding.test",
            transport=httpx.MockTransport(lambda request: httpx.Response(500)),
        )
        factory = EmbeddingRepositoryFactory()
        factory.register(EmbeddingProvider.HTTPX, EnterpriseEmbeddingServiceRepository)

        repository = factory.create(
            EmbeddingProvider.HTTPX,
            client,
            EmbeddingServiceSettings(),
        )

        self.assertIsInstance(repository, EnterpriseEmbeddingServiceRepository)
        await repository.close()
        self.assertTrue(client.is_closed)

    async def test_sends_query_intent_and_validates_response(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json={
                    "model": "intfloat/multilingual-e5-small",
                    "revision": "main",
                    "dimension": 2,
                    "normalized": True,
                    "embeddings": [[0.6, 0.8]],
                },
            )

        client = httpx.AsyncClient(
            base_url="http://embedding.test",
            transport=httpx.MockTransport(handler),
        )
        repository = EnterpriseEmbeddingServiceRepository(
            client,
            EmbeddingServiceSettings(expected_dimension=2),
        )

        vectors = await repository.embed(("question",), input_type="query")

        self.assertEqual(vectors, ((0.6, 0.8),))
        self.assertEqual(requests[0].url.path, "/v1/embeddings")
        self.assertEqual(json.loads(requests[0].content), {
            "input_type": "query",
            "texts": ["question"],
        })
        await repository.close()

    async def test_rejects_wrong_vector_dimension(self) -> None:
        client = httpx.AsyncClient(
            base_url="http://embedding.test",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={
                        "model": "intfloat/multilingual-e5-small",
                        "revision": "main",
                        "dimension": 3,
                        "normalized": True,
                        "embeddings": [[0.1, 0.2, 0.3]],
                    },
                )
            ),
        )
        repository = EnterpriseEmbeddingServiceRepository(
            client,
            EmbeddingServiceSettings(expected_dimension=2),
        )

        with self.assertRaisesRegex(RuntimeError, "unexpected dimension"):
            await repository.embed(("question",), input_type="query")
        await repository.close()
