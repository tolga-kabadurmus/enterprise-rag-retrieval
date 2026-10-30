"""Sanitized HTTP responses for query-pipeline failures."""

import logging

import anthropic
import cohere
import httpx
import openai
from fastapi import Request, status
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.repository.embedding_service.exceptions import EmbeddingServiceRequestError
from app.repository.reranking.exceptions import RerankingError

LOGGER = logging.getLogger(__name__)


class UnhandledExceptionMiddleware(BaseHTTPMiddleware):
    """Log pipeline failures and return capability-specific safe responses."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        """Handle a request without exposing provider exception details."""

        try:
            return await call_next(request)
        except Exception as error:
            status_code, detail = self._response_for(error)
            LOGGER.exception(
                "query request failed method=%s path=%s status=%d",
                request.method,
                request.url.path,
                status_code,
            )
            return JSONResponse(status_code=status_code, content={"detail": detail})

    @staticmethod
    def _response_for(error: Exception) -> tuple[int, str]:
        """Map known dependency failures to stable public HTTP semantics."""

        if isinstance(error, EmbeddingServiceRequestError):
            if error.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
                return status.HTTP_429_TOO_MANY_REQUESTS, "embedding service rate limit exceeded"
            if error.status_code in {
                status.HTTP_408_REQUEST_TIMEOUT,
                status.HTTP_504_GATEWAY_TIMEOUT,
            }:
                return status.HTTP_504_GATEWAY_TIMEOUT, "embedding service timed out"
            if error.status_code in {
                status.HTTP_502_BAD_GATEWAY,
                status.HTTP_503_SERVICE_UNAVAILABLE,
            }:
                return status.HTTP_503_SERVICE_UNAVAILABLE, "embedding service is unavailable"
            return status.HTTP_502_BAD_GATEWAY, "embedding service rejected the request"

        if isinstance(error, httpx.TimeoutException):
            return status.HTTP_504_GATEWAY_TIMEOUT, "upstream service timed out"
        if isinstance(error, httpx.RequestError):
            return status.HTTP_503_SERVICE_UNAVAILABLE, "upstream service is unavailable"

        if isinstance(error, (UnexpectedResponse, ResponseHandlingException)):
            return status.HTTP_503_SERVICE_UNAVAILABLE, "vector retrieval is unavailable"
        if isinstance(error, RerankingError):
            return status.HTTP_503_SERVICE_UNAVAILABLE, "reranking is unavailable"

        if isinstance(
            error,
            (
                openai.APITimeoutError,
                anthropic.APITimeoutError,
                cohere.GatewayTimeoutError,
            ),
        ):
            return status.HTTP_504_GATEWAY_TIMEOUT, "LLM provider timed out"
        if isinstance(
            error,
            (
                openai.RateLimitError,
                anthropic.RateLimitError,
                cohere.TooManyRequestsError,
            ),
        ):
            return status.HTTP_429_TOO_MANY_REQUESTS, "LLM provider rate limit exceeded"
        if isinstance(
            error,
            (
                openai.APIConnectionError,
                anthropic.APIConnectionError,
                anthropic.ServiceUnavailableError,
                cohere.ServiceUnavailableError,
            ),
        ):
            return status.HTTP_503_SERVICE_UNAVAILABLE, "LLM provider is unavailable"
        if isinstance(
            error,
            (openai.APIStatusError, anthropic.APIStatusError, cohere.BadRequestError),
        ):
            return status.HTTP_502_BAD_GATEWAY, "LLM provider rejected the request"

        if isinstance(error, ValueError):
            return status.HTTP_422_UNPROCESSABLE_ENTITY, "request could not be processed"
        return status.HTTP_503_SERVICE_UNAVAILABLE, "service temporarily unavailable"
