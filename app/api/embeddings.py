"""Temporary REST endpoint for testing query vectorization."""

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from app.api.models import DEPENDENCY_ERROR_RESPONSES
from app.domain.models import QueryRequest
from app.usecase import QueryEmbeddingUseCase


class QueryEmbeddingResponse(BaseModel):
    """Embedding-service-compatible response for one user question."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "model": "intfloat/multilingual-e5-small",
                    "revision": "main",
                    "dimension": 3,
                    "normalized": True,
                    "embeddings": [[0.12, -0.34, 0.93]],
                }
            ]
        },
    )

    model: str = Field(min_length=1, description="Dense embedding model name.")
    revision: str = Field(
        min_length=1,
        description="Configured model artifact revision.",
    )
    dimension: int = Field(
        ge=1,
        description="Number of values in the returned dense vector.",
    )
    normalized: bool = Field(
        description="Whether the dense vector is L2 normalized."
    )
    embeddings: tuple[tuple[float, ...], ...] = Field(
        min_length=1,
        max_length=1,
        description="Dense query vector returned as a single-item batch."
    )


router = APIRouter(prefix="/v1", tags=["embeddings"])


@router.post(
    "/embeddings",
    response_model=QueryEmbeddingResponse,
    summary="Create a query embedding",
    description=(
        "Vectorizes one user question through the shared embedding service. "
        "This diagnostic endpoint does not retrieve document chunks or send "
        "content to an LLM."
    ),
    operation_id="create_query_embedding_v1",
    responses=DEPENDENCY_ERROR_RESPONSES,
)
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
