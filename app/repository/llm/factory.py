"""Registry-based factory for LLM repository providers."""

from typing import Any, TypeVar

from app.common.settings.llm import LLMProvider
from app.repository.llm.base import LLMRepositoryBase

LLMRepositoryType = TypeVar("LLMRepositoryType", bound=LLMRepositoryBase)


class LLMRepositoryFactory:
    """Register and construct implementations of the LLM capability."""

    def __init__(self) -> None:
        self._providers: dict[LLMProvider, type[LLMRepositoryBase]] = {}

    def register(
        self,
        provider: LLMProvider,
        repository_type: type[LLMRepositoryType],
    ) -> None:
        """Register one implementation under a validated provider key."""

        if not issubclass(repository_type, LLMRepositoryBase):
            raise TypeError("LLM repository must inherit LLMRepositoryBase")
        if provider in self._providers:
            raise ValueError(f"LLM provider is already registered: {provider.value}")
        self._providers[provider] = repository_type

    def create(
        self,
        provider: LLMProvider,
        *args: Any,
        **kwargs: Any,
    ) -> LLMRepositoryBase:
        """Construct the implementation registered for ``provider``."""

        repository_type = self._providers.get(provider)
        if repository_type is None:
            raise ValueError(f"unsupported LLM provider: {provider.value}")
        return repository_type(*args, **kwargs)
