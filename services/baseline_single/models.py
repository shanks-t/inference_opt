"""Pydantic models for API requests and responses."""

from typing import Any

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request model for text generation."""

    prompt: str = Field(..., description="Input prompt for text generation")
    max_tokens: int = Field(512, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    stream: bool = Field(False, description="Whether to stream the response")


class GenerateResponse(BaseModel):
    """Response model for text generation."""

    id: str = Field(..., description="Unique response ID")
    object: str = Field("text_completion", description="Response object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field("llama-cpp", description="Model used for generation")
    choices: list[dict[str, Any]] = Field(..., description="Generated choices")
    usage: dict[str, int] = Field(..., description="Token usage statistics")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Health status")
    backend: str = Field(..., description="Backend type")
    model_loaded: bool = Field(..., description="Whether model is loaded")
