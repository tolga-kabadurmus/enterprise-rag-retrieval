"""Shared Pydantic models exposed by the retrieval REST API."""

from pydantic import BaseModel, ConfigDict, Field


class ErrorResponse(BaseModel):
    """Sanitized error returned when a pipeline dependency fails."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [{"detail": "embedding service is unavailable"}]
        },
    )

    detail: str = Field(
        min_length=1,
        description="Safe client-facing error explanation.",
    )


class ValidationIssue(BaseModel):
    """One FastAPI-compatible request validation issue."""

    model_config = ConfigDict(frozen=True, extra="allow")

    type: str = Field(description="Machine-readable validation error type.")
    loc: tuple[str | int, ...] = Field(
        description="Request location of the invalid value."
    )
    msg: str = Field(description="Human-readable validation error message.")
    input: object | None = Field(
        default=None,
        description="Rejected input when it is safe to serialize.",
    )


class ValidationErrorResponse(BaseModel):
    """Validation response returned for a malformed query request."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "detail": [
                        {
                            "type": "string_too_short",
                            "loc": ["body", "question"],
                            "msg": "String should have at least 1 character",
                            "input": "",
                        }
                    ]
                }
            ]
        },
    )

    detail: tuple[ValidationIssue, ...]


DEPENDENCY_ERROR_RESPONSES = {
    429: {
        "model": ErrorResponse,
        "description": "An upstream provider rate limit was exceeded.",
    },
    502: {
        "model": ErrorResponse,
        "description": "An upstream provider rejected the request.",
    },
    503: {
        "model": ErrorResponse,
        "description": "The endpoint or one of its dependencies is unavailable.",
    },
    504: {
        "model": ErrorResponse,
        "description": "An upstream provider timed out.",
    },
}


__all__ = [
    "DEPENDENCY_ERROR_RESPONSES",
    "ErrorResponse",
    "ValidationErrorResponse",
    "ValidationIssue",
]
