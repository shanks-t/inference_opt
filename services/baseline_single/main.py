"""Baseline single-request inference service."""

import logging
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from sse_starlette.sse import EventSourceResponse

from adapters import InferenceBackend, create_backend

from .models import GenerateRequest, GenerateResponse, HealthResponse

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global backend instance
backend: InferenceBackend | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global backend

    # Startup
    logger.info("Starting baseline single service")
    try:
        backend = create_backend()
        logger.info("Backend created successfully")
    except Exception as e:
        logger.error(f"Failed to create backend: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down baseline single service")


app = FastAPI(
    title="Inference Optimization - Baseline Single",
    description="Single-request inference service for performance benchmarking",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text from a prompt."""
    if backend is None:
        raise HTTPException(status_code=503, detail="Backend not initialized")


    if request.stream:
        # Return streaming response
        return EventSourceResponse(
            _stream_generate(request.prompt, request.max_tokens), media_type="text/event-stream"
        )
    else:
        # Return complete response
        try:
            results = await backend.generate([request.prompt], request.max_tokens)
            generated_text = results[0]

            # Estimate token counts (rough approximation)
            prompt_tokens = len(request.prompt.split()) * 1.3  # ~1.3 tokens per word
            completion_tokens = len(generated_text.split()) * 1.3
            total_tokens = prompt_tokens + completion_tokens

            return GenerateResponse(
                id=str(uuid.uuid4()),
                object="text_completion",
                created=int(time.time()),
                model="llama-cpp",
                choices=[{"text": generated_text, "index": 0, "finish_reason": "stop"}],
                usage={
                    "prompt_tokens": int(prompt_tokens),
                    "completion_tokens": int(completion_tokens),
                    "total_tokens": int(total_tokens),
                },
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


async def _stream_generate(prompt: str, max_tokens: int) -> AsyncGenerator[str, None]:
    """Stream text generation."""
    if backend is None:
        yield "data: {'error': 'Backend not initialized'}\\n\\n"
        return

    try:
        response_id = str(uuid.uuid4())
        created = int(time.time())

        # Send initial response
        initial_data = {
            "id": response_id,
            "object": "text_completion",
            "created": created,
            "model": "llama-cpp",
            "choices": [{"text": "", "index": 0, "finish_reason": None}]
        }
        yield f"data: {initial_data}\\n\\n"

        # Stream tokens
        stream = await backend.generate_stream(prompt, max_tokens)
        async for token in stream:
            chunk_data = {
                "id": response_id,
                "object": "text_completion",
                "created": created,
                "model": "llama-cpp",
                "choices": [{"text": token, "index": 0, "finish_reason": None}],
            }
            yield f"data: {chunk_data}\\n\\n"

        # Send final chunk
        final_chunk = {
            "id": response_id,
            "object": "text_completion",
            "created": created,
            "model": "llama-cpp",
            "choices": [{"text": "", "index": 0, "finish_reason": "stop"}],
        }
        yield f"data: {final_chunk}\\n\\n"
        yield "data: [DONE]\\n\\n"

    except Exception as e:
        logger.error(f"Streaming failed: {e}")
        yield f'data: {{"error": "{str(e)}"}}\\n\\n'


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health."""
    if backend is None:
        return HealthResponse(status="unhealthy", backend="none", model_loaded=False)

    is_healthy = await backend.health_check()

    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        backend="llama_cpp",
        model_loaded=is_healthy,
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Inference Optimization - Baseline Single Service", "version": "1.0.0"}
