"""Qdrant implementation of provider-neutral vector retrieval."""

from qdrant_client import AsyncQdrantClient, models

from app.common.settings.vector import VectorSettings
from app.repository.embedding_service.base import EmbeddingVector
from app.repository.vector.base import VectorRepositoryBase, VectorSearchResult


class QdrantVectorRepository(VectorRepositoryBase):
    """Run hybrid dense and sparse searches through an injected Qdrant client."""

    def __init__(self, client: AsyncQdrantClient, settings: VectorSettings) -> None:
        self._client = client
        self._settings = settings
        self._closed = False

    async def search(
        self,
        query_vector: EmbeddingVector,
        limit: int,
        prefetch_limit: int,
        score_threshold: float | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        """Validate vectors and fuse dense and sparse candidate ranks with RRF."""

        if self._closed:
            raise RuntimeError("vector repository is closed")
        if len(query_vector.dense) != self._settings.vector_size:
            raise ValueError(
                "query vector has dimension "
                f"{len(query_vector.dense)}; expected {self._settings.vector_size}"
            )
        if limit < 1:
            raise ValueError("search limit must be at least 1")
        if prefetch_limit < limit:
            raise ValueError("prefetch limit must be at least the result limit")

        response = await self._client.query_points(
            collection_name=self._settings.collection_name,
            prefetch=[
                models.Prefetch(
                    query=list(query_vector.dense),
                    using=self._settings.dense_vector_name,
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=list(query_vector.sparse.indices),
                        values=list(query_vector.sparse.values),
                    ),
                    using=self._settings.sparse_vector_name,
                    limit=prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,
        )
        return tuple(
            VectorSearchResult(
                point_id=str(point.id),
                score=point.score,
                payload=point.payload or {},
            )
            for point in response.points
        )

    async def close(self) -> None:
        """Close the injected asynchronous client exactly once."""

        if self._closed:
            return
        self._closed = True
        await self._client.close()
