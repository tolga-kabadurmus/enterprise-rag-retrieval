"""Qdrant implementation of provider-neutral vector retrieval."""

from qdrant_client import AsyncQdrantClient

from app.common.settings.vector import VectorSettings
from app.repository.vector.base import VectorRepositoryBase, VectorSearchResult


class QdrantVectorRepository(VectorRepositoryBase):
    """Run dense similarity searches through an injected Qdrant client."""

    def __init__(self, client: AsyncQdrantClient, settings: VectorSettings) -> None:
        self._client = client
        self._settings = settings
        self._closed = False

    async def search(
        self,
        query_vector: tuple[float, ...],
        limit: int,
        score_threshold: float | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        """Validate and execute a named dense-vector search."""

        if self._closed:
            raise RuntimeError("vector repository is closed")
        if len(query_vector) != self._settings.vector_size:
            raise ValueError(
                "query vector has dimension "
                f"{len(query_vector)}; expected {self._settings.vector_size}"
            )
        if limit < 1:
            raise ValueError("search limit must be at least 1")

        response = await self._client.query_points(
            collection_name=self._settings.collection_name,
            query=list(query_vector),
            using=self._settings.dense_vector_name,
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
