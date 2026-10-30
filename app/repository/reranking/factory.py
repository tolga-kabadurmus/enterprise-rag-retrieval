"""Registry-based factory for reranking providers."""

from typing import Any, TypeVar

from app.common.settings.reranking import RerankingProvider
from app.repository.reranking.base import RerankingRepositoryBase

RerankingRepositoryType = TypeVar(
    "RerankingRepositoryType",
    bound=RerankingRepositoryBase,
)


class RerankingRepositoryFactory:
    """Register and construct implementations of the reranking capability."""

    def __init__(self) -> None:
        self._providers: dict[RerankingProvider, type[RerankingRepositoryBase]] = {}

    def register(
        self,
        provider: RerankingProvider,
        repository_type: type[RerankingRepositoryType],
    ) -> None:
        """Register one implementation under a stable provider key."""

        if not issubclass(repository_type, RerankingRepositoryBase):
            raise TypeError("reranking repository must inherit RerankingRepositoryBase")
        if provider in self._providers:
            raise ValueError(
                f"reranking provider is already registered: {provider.value}"
            )
        self._providers[provider] = repository_type

    def create(
        self,
        provider: RerankingProvider,
        *args: Any,
        **kwargs: Any,
    ) -> RerankingRepositoryBase:
        """Construct the implementation registered for ``provider``."""

        repository_type = self._providers.get(provider)
        if repository_type is None:
            raise ValueError(f"unsupported reranking provider: {provider.value}")
        return repository_type(*args, **kwargs)
