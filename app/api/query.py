"""REST endpoints for complete and streamed grounded answers."""

from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.api.models import ErrorResponse, ValidationErrorResponse
from app.controller import AnswerQuestionController
from app.domain import (
    QueryDeltaEvent,
    QueryDoneEvent,
    QueryErrorEvent,
    QueryRequest,
    QueryResponse,
    QuerySourcesEvent,
    QueryStreamEvent,
)

router = APIRouter(prefix="/v1", tags=["query"])

QUERY_ERROR_RESPONSES = {
    422: {
        "model": ValidationErrorResponse,
        "description": "The request body failed validation.",
        "content": {
            "application/json": {
                "example": {
                    "detail": [
                        {
                            "type": "string_too_short",
                            "loc": ["body", "question"],
                            "msg": "String should have at least 1 character",
                            "input": "",
                        }
                    ]
                }
            }
        },
    },
    429: {
        "model": ErrorResponse,
        "description": "The configured LLM provider rate limit was exceeded.",
        "content": {
            "application/json": {
                "example": {"detail": "LLM provider rate limit exceeded"}
            }
        },
    },
    502: {
        "model": ErrorResponse,
        "description": "An upstream embedding, vector, reranking, or LLM provider failed.",
        "content": {
            "application/json": {
                "example": {"detail": "LLM provider rejected the request"}
            }
        },
    },
    503: {
        "model": ErrorResponse,
        "description": "Query answering or one of its dependencies is unavailable.",
        "content": {
            "application/json": {
                "example": {"detail": "query answering is not ready"}
            }
        },
    },
    504: {
        "model": ErrorResponse,
        "description": "An upstream embedding, vector, reranking, or LLM call timed out.",
        "content": {
            "application/json": {
                "example": {"detail": "LLM provider timed out"}
            }
        },
    },
}

QUERY_RESPONSE_EXAMPLE = {
    "answer": "Employees receive 20 days of annual leave [S1].",
    "sources": [
        {
            "citation_id": "S1",
            "document_id": "employee-handbook",
            "source_bucket": "documents",
            "object_key": "hr/employee-handbook.pdf",
            "chunk_id": "employee-handbook-page-12-chunk-0",
            "page_number": 12,
            "chunk_index": 0,
        }
    ],
}

SUCCESSFUL_SSE_EXAMPLE = """event: delta
data: {"event":"delta","delta":"Employees receive 20 days of annual leave [S1]."}

event: sources
data: {"event":"sources","sources":[{"document_id":"employee-handbook","source_bucket":"documents","object_key":"hr/employee-handbook.pdf","chunk_id":"employee-handbook-page-12-chunk-0","page_number":12,"chunk_index":0,"citation_id":"S1"}]}

event: done
data: {"event":"done"}

"""

RESERVED_ERROR_SSE_EXAMPLE = """event: delta
data: {"event":"delta","delta":"Employees receive "}

event: error
data: {"event":"error","code":"stream_failed","detail":"answer streaming could not be completed"}

event: done
data: {"event":"done"}

"""

SSE_EVENT_SCHEMAS = {
    "delta": QueryDeltaEvent.model_json_schema(
        ref_template="#/components/schemas/{model}"
    ),
    "sources": QuerySourcesEvent.model_json_schema(
        ref_template="#/components/schemas/{model}"
    ),
    "error": QueryErrorEvent.model_json_schema(
        ref_template="#/components/schemas/{model}"
    ),
    "done": QueryDoneEvent.model_json_schema(
        ref_template="#/components/schemas/{model}"
    ),
}


def _controller(request: Request) -> AnswerQuestionController:
    """Return the composed answer controller or report failed readiness."""

    controller: AnswerQuestionController | None = getattr(
        request.app.state,
        "answer_question_controller",
        None,
    )
    if controller is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="query answering is not ready",
        )
    return controller


def _sse(event: QueryStreamEvent) -> str:
    """Serialize one typed query event using the SSE wire format."""

    return f"event: {event.event}\ndata: {event.model_dump_json()}\n\n"


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Answer a question with source citations",
    description=(
        "Retrieves and reranks relevant chunks, generates one complete "
        "grounded LLM answer by buffering the provider's streamed deltas, "
        "and returns response-scoped source citations. "
        "Citation IDs such as S1 match labels used inside the answer."
    ),
    operation_id="answer_query_v1",
    responses={
        200: {
            "description": "Complete grounded answer with deduplicated citations.",
            "content": {
                "application/json": {"example": QUERY_RESPONSE_EXAMPLE}
            },
        },
        **QUERY_ERROR_RESPONSES,
    },
)
async def answer_query(
    payload: QueryRequest,
    request: Request,
) -> QueryResponse:
    """Return one complete grounded answer and its cited sources."""

    return await _controller(request).execute(payload)


@router.post(
    "/query/stream",
    response_class=StreamingResponse,
    summary="Stream an answer followed by source citations",
    description=(
        "Streams UTF-8 Server-Sent Events. A successful stream contains one "
        "or more delta events, one sources event, and one done event in that "
        "order. The documented error event is reserved for the upcoming "
        "mid-stream sanitization step and is not emitted by the current "
        "implementation. HTTP errors listed below apply before streaming starts."
    ),
    operation_id="stream_query_v1",
    responses={
        200: {
            "description": (
                "SSE stream. Each frame uses an event name and one JSON data line."
            ),
            "content": {
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "description": (
                            "SSE frames separated by a blank line. The data field "
                            "contains JSON matching the event schema documented "
                            "in x-sse-event-schemas."
                        ),
                    },
                    "examples": {
                        "successful_stream": {
                            "summary": "Successful answer and citations",
                            "value": SUCCESSFUL_SSE_EXAMPLE,
                        },
                        "reserved_error_stream": {
                            "summary": "Reserved sanitized mid-stream failure",
                            "description": (
                                "Contract reserved for the next error-handling step; "
                                "the current endpoint does not emit this event."
                            ),
                            "value": RESERVED_ERROR_SSE_EXAMPLE,
                        },
                    },
                }
            },
        },
        **QUERY_ERROR_RESPONSES,
    },
    openapi_extra={"x-sse-event-schemas": SSE_EVENT_SCHEMAS},
)
async def stream_query(
    payload: QueryRequest,
    request: Request,
) -> StreamingResponse:
    """Return an SSE stream for one grounded answer request."""

    controller = _controller(request)

    async def event_stream() -> AsyncIterator[str]:
        async for event in controller.stream(payload):
            yield _sse(event)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


__all__ = ["router"]
