"""Use case for retrieving and mapping relevant vector-store chunks."""

from pydantic import ValidationError

from app.common.settings.retrieval import RetrievalSettings
from app.domain import RetrievedChunk, SourceReference
from app.repository.vector import VectorRepositoryBase, VectorSearchResult
from app.usecase.base import UseCase


class RetrieveChunksUseCase(
    UseCase[tuple[float, ...], tuple[RetrievedChunk, ...]]
):
    """Retrieve relevant vectors and map their payloads into domain chunks."""

    def __init__(
        self,
        repo: VectorRepositoryBase,
        settings: RetrievalSettings,
    ) -> None:
        self._repo = repo
        self._settings = settings

    async def execute(
        self,
        value: tuple[float, ...],
    ) -> tuple[RetrievedChunk, ...]:
        """Search with configured limits and return validated retrieved chunks."""

        if not value:
            raise ValueError("query vector must not be empty")
        results = await self._repo.search(
            query_vector=value,
            limit=self._settings.top_k,
            score_threshold=self._settings.score_threshold,
        )
        return tuple(self._to_chunk(result) for result in results)

    def _to_chunk(self, result: VectorSearchResult) -> RetrievedChunk:
        """Validate one repository payload against the indexed metadata contract."""

        payload = result.payload
        try:
            source = SourceReference(
                document_id=payload["document_id"],
                source_bucket=payload["source_bucket"],
                object_key=payload["object_key"],
                chunk_id=payload["chunk_id"],
                page_number=payload.get("page_number"),
                chunk_index=payload.get("chunk_index"),
            )
            return RetrievedChunk(
                text=payload["text"],
                score=result.score,
                source=source,
            )
        except (KeyError, TypeError, ValidationError) as error:
            raise RuntimeError(
                f"invalid retrieval payload for vector point {result.point_id}"
            ) from error
