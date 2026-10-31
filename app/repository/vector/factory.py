"""Registry-based factory for vector repository providers."""

from typing import Any, TypeVar

from app.common.settings.vector import VectorProvider
from app.repository.vector.base import VectorRepositoryBase

VectorRepositoryType = TypeVar(
    "VectorRepositoryType",
    bound=VectorRepositoryBase,
)


class VectorRepositoryFactory:
    """Register and construct implementations of the vector capability."""

    def __init__(self) -> None:
        self._providers: dict[VectorProvider, type[VectorRepositoryBase]] = {}

    def register(
        self,
        provider: VectorProvider,
        repository_type: type[VectorRepositoryType],
    ) -> None:
        """Register one implementation under a stable provider key."""

        if not issubclass(repository_type, VectorRepositoryBase):
            raise TypeError("vector repository must inherit VectorRepositoryBase")
        if provider in self._providers:
            raise ValueError(f"vector provider is already registered: {provider.value}")
        self._providers[provider] = repository_type

    def create(
        self,
        provider: VectorProvider,
        *args: Any,
        **kwargs: Any,
    ) -> VectorRepositoryBase:
        """Construct the implementation registered for ``provider``."""

        repository_type = self._providers.get(provider)
        if repository_type is None:
            raise ValueError(f"unsupported vector provider: {provider.value}")
        return repository_type(*args, **kwargs)
