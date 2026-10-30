"""Shared application use-case contract."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


class UseCase(ABC, Generic[InputType, OutputType]):
    """Small asynchronous contract implemented by application use cases."""

    @abstractmethod
    async def execute(self, value: InputType) -> OutputType:
        """Execute the application operation for ``value``."""
