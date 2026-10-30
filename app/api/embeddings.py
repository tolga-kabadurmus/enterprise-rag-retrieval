"""Temporary REST endpoint for testing query vectorization."""

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from app.domain.models import QueryRequest
from app.usecase import QueryEmbeddingUseCase


class QueryEmbeddingResponse(BaseModel):
    """Embedding-service-compatible response for one user question."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    dimension: int = Field(ge=1)
    normalized: bool
    embeddings: tuple[tuple[float, ...], ...]


router = APIRouter(prefix="/v1", tags=["temporary"])


@router.post("/embeddings", response_model=QueryEmbeddingResponse)
async def create_query_embedding(
    payload: QueryRequest,
    request: Request,
) -> QueryEmbeddingResponse:
    """Vectorize one question through the shared embedding-service hop.

    Raises:
        HTTPException: If application startup has not provided the use case.
    """

    usecase: QueryEmbeddingUseCase | None = getattr(
        request.app.state,
        "query_embedding_usecase",
        None,
    )
    if usecase is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="query embedding is not ready",
        )

    vector = await usecase.execute(payload.question)
    settings = request.app.state.settings.embedding
    return QueryEmbeddingResponse(
        model=settings.expected_model,
        revision=settings.expected_revision,
        dimension=len(vector.dense),
        normalized=settings.expected_normalized,
        embeddings=(vector.dense,),
    )


__all__ = ["QueryEmbeddingResponse", "router"]
