"""Provider-neutral LLM generation contract and message model."""

from abc import abstractmethod
from collections.abc import AsyncIterator
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.repository.base import Repository


class LLMMessage(BaseModel):
    """One provider-neutral chat message."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class LLMRepositoryBase(Repository):
    """Generate complete or streamed text through a configured LLM."""

    def __init__(self, system_prompt: str) -> None:
        """Register the system prompt applied to every provider request."""

        prompt = system_prompt.strip()
        if not prompt:
            raise ValueError("system prompt must not be blank")
        self._system_prompt = prompt

    async def generate(self, messages: tuple[LLMMessage, ...]) -> str:
        """Prepend the registered system prompt and generate a complete answer."""

        return await self._generate(self._with_system_prompt(messages))

    async def stream(
        self,
        messages: tuple[LLMMessage, ...],
    ) -> AsyncIterator[str]:
        """Prepend the registered system prompt and yield generated deltas."""

        async for delta in self._stream(self._with_system_prompt(messages)):
            yield delta

    def _with_system_prompt(
        self,
        messages: tuple[LLMMessage, ...],
    ) -> tuple[LLMMessage, ...]:
        """Apply the configured prompt as the single authoritative system message."""

        if not messages:
            raise ValueError("at least one LLM message is required")
        non_system_messages = tuple(
            message for message in messages if message.role != "system"
        )
        if not non_system_messages:
            raise ValueError("at least one user or assistant message is required")
        return (
            LLMMessage(role="system", content=self._system_prompt),
            *non_system_messages,
        )

    @abstractmethod
    async def _generate(self, messages: tuple[LLMMessage, ...]) -> str:
        """Generate complete text from messages containing the system prompt."""

    @abstractmethod
    def _stream(self, messages: tuple[LLMMessage, ...]) -> AsyncIterator[str]:
        """Stream text from messages containing the system prompt."""
