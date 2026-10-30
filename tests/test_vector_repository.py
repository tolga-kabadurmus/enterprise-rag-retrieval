"""Unit tests for Qdrant vector retrieval and provider selection."""

import unittest

from qdrant_client import models
from qdrant_client.http.models.models import QueryResponse

from app.common.settings.vector import VectorProvider, VectorSettings
from app.repository.vector import (
    QdrantVectorRepository,
    VectorRepositoryFactory,
)
from app.repository.embedding_service import EmbeddingVector, SparseEmbedding


class FakeQdrantClient:
    """Record Qdrant calls without external connectivity."""

    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}
        self.closed = False

    async def query_points(self, **kwargs: object) -> QueryResponse:
        self.kwargs = kwargs
        return QueryResponse(
            points=[
                models.ScoredPoint(
                    id=7,
                    version=1,
                    score=0.91,
                    payload={"document_id": "doc-1", "chunk_id": "chunk-2"},
                )
            ]
        )

    async def close(self) -> None:
        self.closed = True


class VectorRepositoryTests(unittest.IsolatedAsyncioTestCase):
    """Verify dense search mapping, validation, factory, and cleanup."""

    async def test_searches_named_vector_and_maps_payload(self) -> None:
        client = FakeQdrantClient()
        settings = VectorSettings(
            url="http://qdrant:6333",
            collection_name="chunks",
            dense_vector_name="dense",
            vector_size=2,
        )
        repository = QdrantVectorRepository(client, settings)

        query_vector = EmbeddingVector(
            dense=(0.6, 0.8),
            sparse=SparseEmbedding(indices=(11, 29), values=(1.0, 2.0)),
        )
        results = await repository.search(
            query_vector,
            limit=3,
            prefetch_limit=12,
            score_threshold=0.5,
        )

        self.assertEqual(results[0].point_id, "7")
        self.assertEqual(results[0].payload["document_id"], "doc-1")
        self.assertEqual(client.kwargs["collection_name"], "chunks")
        prefetch = client.kwargs["prefetch"]
        self.assertEqual(prefetch[0].using, "dense")
        self.assertEqual(prefetch[0].query, [0.6, 0.8])
        self.assertEqual(prefetch[0].limit, 12)
        self.assertEqual(prefetch[1].using, "sparse")
        self.assertEqual(prefetch[1].query.indices, [11, 29])
        self.assertEqual(client.kwargs["query"], models.FusionQuery(fusion=models.Fusion.RRF))
        await repository.close()
        self.assertTrue(client.closed)

    async def test_rejects_wrong_query_dimension(self) -> None:
        repository = QdrantVectorRepository(
            FakeQdrantClient(),
            VectorSettings(
                url="http://qdrant:6333",
                collection_name="chunks",
                vector_size=2,
            ),
        )

        with self.assertRaisesRegex(ValueError, "dimension"):
            await repository.search(
                EmbeddingVector(
                    dense=(0.1,),
                    sparse=SparseEmbedding(indices=(1,), values=(1.0,)),
                ),
                limit=1,
                prefetch_limit=4,
            )

    async def test_factory_creates_registered_qdrant_repository(self) -> None:
        factory = VectorRepositoryFactory()
        factory.register(VectorProvider.QDRANT, QdrantVectorRepository)

        repository = factory.create(
            VectorProvider.QDRANT,
            FakeQdrantClient(),
            VectorSettings(url="http://qdrant:6333", collection_name="chunks"),
        )

        self.assertIsInstance(repository, QdrantVectorRepository)
        await repository.close()
