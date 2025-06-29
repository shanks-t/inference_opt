"""Base protocol for inference backends."""

from abc import abstractmethod
from collections.abc import AsyncIterator
from typing import Protocol


class InferenceBackend(Protocol):
    """Protocol for inference backends that can generate text from prompts."""

    @abstractmethod
    async def generate(self, prompts: list[str], max_tokens: int = 512) -> list[str]:
        """Generate text for multiple prompts.

        Args:
            prompts: List of input prompts to generate from
            max_tokens: Maximum tokens to generate per prompt

        Returns:
            List of generated text strings, one per prompt (same order)
        """
        ...

    @abstractmethod
    async def generate_stream(self, prompt: str, max_tokens: int = 512) -> AsyncIterator[str]:
        """Generate text for a single prompt with streaming.

        Args:
            prompt: Input prompt to generate from
            max_tokens: Maximum tokens to generate

        Yields:
            Generated tokens as they are produced
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the backend is healthy and ready to serve requests.

        Returns:
            True if backend is healthy, False otherwise
        """
        ...
