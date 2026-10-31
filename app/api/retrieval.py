"""REST endpoint for retrieving chunks relevant to a user question."""

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from app.api.models import DEPENDENCY_ERROR_RESPONSES
from app.controller import RetrievalController
from app.domain import QueryRequest, RetrievedChunk


class RetrievalResponse(BaseModel):
    """Retrieved context returned before answer generation."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "chunks": [
                        {
                            "text": "Employees receive 20 days of annual leave.",
                            "score": 0.91,
                            "source": {
                                "document_id": "employee-handbook",
                                "source_bucket": "documents",
                                "object_key": "hr/employee-handbook.pdf",
                                "chunk_id": "employee-handbook-page-12-chunk-0",
                                "page_number": 12,
                                "chunk_index": 0,
                            },
                        }
                    ]
                }
            ]
        },
    )

    chunks: tuple[RetrievedChunk, ...] = Field(
        description=(
            "Relevant chunks ordered by the retrieval pipeline, including "
            "source metadata suitable for citations."
        )
    )


router = APIRouter(prefix="/v1", tags=["retrieval"])


@router.post(
    "/retrieval",
    response_model=RetrievalResponse,
    summary="Retrieve relevant document chunks",
    description=(
        "Embeds one question, performs dense and sparse Qdrant retrieval, "
        "fuses candidates, reranks them, and returns grounded source chunks. "
        "This endpoint does not generate an LLM answer."
    ),
    operation_id="retrieve_chunks_v1",
    responses=DEPENDENCY_ERROR_RESPONSES,
)
async def retrieve_chunks(
    payload: QueryRequest,
    request: Request,
) -> RetrievalResponse:
    """Vectorize a question and retrieve relevant indexed chunks.

    Raises:
        HTTPException: If startup has not provided the retrieval controller.
    """

    controller: RetrievalController | None = getattr(
        request.app.state,
        "retrieval_controller",
        None,
    )
    if controller is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="retrieval is not ready",
        )

    chunks = await controller.execute(payload)
    return RetrievalResponse(chunks=chunks)


__all__ = ["RetrievalResponse", "router"]
