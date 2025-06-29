"""llama.cpp backend implementation."""

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path

from llama_cpp import Llama

logger = logging.getLogger(__name__)


class LlamaCppBackend:
    """llama.cpp backend for text generation."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,  # -1 = use all available GPU layers
        n_threads: int | None = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repeat_penalty: float = 1.1,
    ):
        """Initialize llama.cpp backend.

        Args:
            model_path: Path to GGUF model file
            n_ctx: Context length (max tokens)
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            n_threads: Number of CPU threads (None for auto)
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repeat_penalty: Repetition penalty
        """
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads or os.cpu_count()
        self.temperature = temperature
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty

        self._model: Llama | None = None
        self._model_loaded = False

    async def _load_model(self) -> None:
        """Load the model if not already loaded."""
        if self._model_loaded:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        logger.info(f"Loading model from {self.model_path}")
        logger.info(f"GPU layers: {self.n_gpu_layers}, Context: {self.n_ctx}")

        # Load model in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(
            None,
            lambda: Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.n_threads,
                verbose=False,
            ),
        )

        self._model_loaded = True
        logger.info("Model loaded successfully")

    async def generate(self, prompts: list[str], max_tokens: int = 512) -> list[str]:
        """Generate text for multiple prompts."""
        await self._load_model()

        if not self._model:
            raise RuntimeError("Model failed to load")

        results = []

        # Process prompts sequentially (no batching in baseline)
        for prompt in prompts:
            loop = asyncio.get_event_loop()

            # Run generation in thread pool
            def _generate():
                assert self._model is not None
                return self._model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    repeat_penalty=self.repeat_penalty,
                    stop=["</s>", "\n\n"],  # Common stop sequences
                    echo=False,
                )

            response = await loop.run_in_executor(None, _generate)

            generated_text = response["choices"][0]["text"]  # type: ignore
            results.append(generated_text.strip())

        return results

    async def generate_stream(self, prompt: str, max_tokens: int = 512) -> AsyncIterator[str]:
        """Generate text with streaming."""
        await self._load_model()

        if not self._model:
            raise RuntimeError("Model failed to load")

        loop = asyncio.get_event_loop()

        # Create generator in thread pool
        def _create_generator():
            assert self._model is not None
            return self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                repeat_penalty=self.repeat_penalty,
                stop=["</s>", "\n\n"],
                echo=False,
                stream=True,
            )

        generator = await loop.run_in_executor(None, _create_generator)

        # Stream tokens
        for chunk in generator:
            if chunk["choices"][0]["finish_reason"] is None:
                token = chunk["choices"][0]["text"]
                if token:
                    yield token

                # Yield control to allow other tasks
                await asyncio.sleep(0)

    async def health_check(self) -> bool:
        """Check if backend is healthy."""
        try:
            await self._load_model()
            return self._model is not None
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
