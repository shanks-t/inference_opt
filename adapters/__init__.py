"""Backend factory and configuration."""

import os
from typing import cast

from .base import InferenceBackend
from .llama_cpp import LlamaCppBackend


def create_backend() -> InferenceBackend:
    """Create inference backend based on environment configuration."""
    backend_type = os.getenv("BACKEND", "llama_cpp").lower()

    if backend_type == "llama_cpp":
        return cast(InferenceBackend, _create_llama_cpp_backend())
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def _create_llama_cpp_backend() -> LlamaCppBackend:
    """Create llama.cpp backend with environment configuration."""
    model_path = os.getenv("MODEL_PATH")
    if not model_path:
        raise ValueError("MODEL_PATH environment variable is required for llama_cpp backend")

    # Configuration from environment variables with defaults
    n_ctx = int(os.getenv("N_CTX", "2048"))
    n_gpu_layers = int(os.getenv("N_GPU_LAYERS", "-1"))  # -1 = use all GPU layers
    n_threads = os.getenv("N_THREADS")
    temperature = float(os.getenv("TEMPERATURE", "0.7"))
    top_p = float(os.getenv("TOP_P", "0.95"))
    repeat_penalty = float(os.getenv("REPEAT_PENALTY", "1.1"))

    return LlamaCppBackend(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=int(n_threads) if n_threads else None,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
    )


__all__ = ["InferenceBackend", "create_backend"]
