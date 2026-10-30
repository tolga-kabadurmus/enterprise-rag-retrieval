"""FastAPI application entry point and dependency composition."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import httpx2
from anthropic import AsyncAnthropic
from cohere import AsyncClientV2
from fastapi import FastAPI
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from qdrant_client import AsyncQdrantClient

from app.api import router as api_router
from app.common.settings import LLMProvider, load_settings
from app.controller import RetrievalController
from app.middleware import UnhandledExceptionMiddleware
from app.repository.embedding_service import (
    EmbeddingRepositoryFactory,
    EnterpriseEmbeddingServiceRepository,
)
from app.repository.llm import (
    AnthropicLLMRepository,
    CohereLLMRepository,
    LLMRepositoryFactory,
    OpenAILLMRepository,
)
from app.repository.vector import (
    QdrantVectorRepository,
    VectorRepositoryFactory,
)
from app.usecase import (
    GenerateAnswerUseCase,
    QueryEmbeddingUseCase,
    RetrieveChunksUseCase,
)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Create and release process-wide external-service dependencies."""

    settings = load_settings()
    embedding_settings = settings.embedding
    client = httpx.AsyncClient(
        base_url=embedding_settings.url,
        timeout=embedding_settings.timeout_seconds,
        verify=embedding_settings.verify_tls,
        http2=True,
        limits=httpx.Limits(
            max_connections=embedding_settings.max_connections,
            max_keepalive_connections=(
                embedding_settings.max_keepalive_connections
            ),
        ),
    )
    embedding_repository = None
    vector_client = None
    vector_repository = None
    llm_http_client = None
    llm_client = None
    llm_repository = None
    try:
        embedding_factory = EmbeddingRepositoryFactory()
        embedding_factory.register(
            embedding_settings.provider,
            EnterpriseEmbeddingServiceRepository,
        )
        embedding_repository = embedding_factory.create(
            embedding_settings.provider,
            client,
            embedding_settings,
        )

        vector_settings = settings.vector
        api_key = (
            vector_settings.api_key.get_secret_value()
            if vector_settings.api_key is not None
            else None
        )
        vector_client = AsyncQdrantClient(
            url=str(vector_settings.url),
            api_key=api_key,
            timeout=vector_settings.timeout_seconds,
        )
        vector_factory = VectorRepositoryFactory()
        vector_factory.register(
            vector_settings.provider,
            QdrantVectorRepository,
        )
        vector_repository = vector_factory.create(
            vector_settings.provider,
            vector_client,
            vector_settings,
        )

        llm_settings = settings.llm
        llm_api_key = (
            llm_settings.api_key.get_secret_value()
            if llm_settings.api_key is not None
            else "not-required"
        )
        llm_factory = LLMRepositoryFactory()
        llm_factory.register(
            LLMProvider.OPENAI_COMPATIBLE,
            OpenAILLMRepository,
        )
        llm_factory.register(LLMProvider.ANTHROPIC, AnthropicLLMRepository)
        llm_factory.register(LLMProvider.COHERE, CohereLLMRepository)

        if llm_settings.provider is LLMProvider.COHERE:
            llm_http_client = httpx.AsyncClient(
                http2=True,
                verify=llm_settings.verify_tls,
                timeout=llm_settings.timeout_seconds,
                limits=httpx.Limits(
                    max_connections=llm_settings.max_connections,
                    max_keepalive_connections=(
                        llm_settings.max_keepalive_connections
                    ),
                ),
            )
            llm_client = AsyncClientV2(
                api_key=llm_api_key,
                base_url=str(llm_settings.base_url),
                timeout=llm_settings.timeout_seconds,
                httpx_client=llm_http_client,
            )
            llm_repository = llm_factory.create(
                llm_settings.provider,
                llm_client,
                llm_settings,
                llm_http_client,
            )
        else:
            llm_http_client = DefaultAsyncHttpxClient(
                http2=True,
                verify=llm_settings.verify_tls,
                timeout=llm_settings.timeout_seconds,
                limits=httpx2.Limits(
                    max_connections=llm_settings.max_connections,
                    max_keepalive_connections=(
                        llm_settings.max_keepalive_connections
                    ),
                ),
            )
            if llm_settings.provider is LLMProvider.ANTHROPIC:
                llm_client = AsyncAnthropic(
                    api_key=llm_api_key,
                    base_url=str(llm_settings.base_url),
                    timeout=llm_settings.timeout_seconds,
                    http_client=llm_http_client,
                )
            else:
                llm_client = AsyncOpenAI(
                    api_key=llm_api_key,
                    base_url=str(llm_settings.base_url),
                    timeout=llm_settings.timeout_seconds,
                    http_client=llm_http_client,
                )
            llm_repository = llm_factory.create(
                llm_settings.provider,
                llm_client,
                llm_settings,
            )

        application.state.settings = settings
        application.state.embedding_repository = embedding_repository
        query_embedding_usecase = QueryEmbeddingUseCase(
            repo=embedding_repository
        )
        application.state.query_embedding_usecase = query_embedding_usecase
        application.state.vector_client = vector_client
        application.state.vector_repository = vector_repository
        retrieve_chunks_usecase = RetrieveChunksUseCase(
            repo=vector_repository,
            settings=settings.retrieval,
        )
        application.state.retrieve_chunks_usecase = retrieve_chunks_usecase
        application.state.retrieval_controller = RetrievalController(
            query_embedding_usecase=query_embedding_usecase,
            retrieve_chunks_usecase=retrieve_chunks_usecase,
        )
        application.state.llm_http_client = llm_http_client
        application.state.llm_client = llm_client
        application.state.llm_repository = llm_repository
        application.state.generate_answer_usecase = GenerateAnswerUseCase(
            repo=llm_repository
        )
        yield
    finally:
        if llm_repository is not None:
            await llm_repository.close()
        elif llm_client is not None and hasattr(llm_client, "close"):
            await llm_client.close()
        elif llm_http_client is not None:
            await llm_http_client.aclose()
        if vector_repository is not None:
            await vector_repository.close()
        elif vector_client is not None:
            await vector_client.close()
        if embedding_repository is None:
            await client.aclose()
        else:
            await embedding_repository.close()


def create_app() -> FastAPI:
    """Create the query service application.

    Returns:
        A FastAPI application ready to be served by an ASGI server.
    """

    application = FastAPI(
        title="Enterprise RAG Query",
        version="0.1.0",
        description="Enterprise retrieval and answer-generation service.",
        lifespan=lifespan,
    )
    application.add_middleware(UnhandledExceptionMiddleware)
    application.include_router(api_router)
    return application


app = create_app()
