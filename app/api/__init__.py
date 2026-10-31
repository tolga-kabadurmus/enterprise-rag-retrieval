"""REST API package."""

from fastapi import APIRouter

from app.api.embeddings import QueryEmbeddingResponse
from app.api.embeddings import router as embeddings_router
from app.api.query import router as query_router
from app.api.retrieval import RetrievalResponse
from app.api.retrieval import router as retrieval_router

router = APIRouter()
router.include_router(embeddings_router)
router.include_router(retrieval_router)
router.include_router(query_router)

__all__ = ["QueryEmbeddingResponse", "RetrievalResponse", "router"]
