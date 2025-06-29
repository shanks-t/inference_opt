.PHONY: dev run bench dash docker clean help

# Default service if not specified
SERVICE ?= baseline_single

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

dev: ## Set up development environment
	@echo "Setting up development environment..."
	uv sync
	@echo "Development environment ready!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Copy .env.example to .env and configure MODEL_PATH"
	@echo "2. Download a GGUF model (e.g., from HuggingFace)"
	@echo "3. Run: make run SERVICE=baseline_single"

run: ## Start a service locally (usage: make run SERVICE=baseline_single)
	@echo "Starting $(SERVICE) service..."
	@if [ ! -f .env ]; then \
		echo "Error: .env file not found. Copy .env.example to .env and configure it."; \
		exit 1; \
	fi
	PYTHONPATH=. uv run python -m uvicorn services.$(SERVICE).main:app --host 0.0.0.0 --port 8000 --reload

generate-prompts: ## Generate prompt corpus
	@echo "Generating prompt corpus..."
	uv run python tools/make_prompts.py --n 500
	@echo "Prompts generated in data/prompts.jsonl"

test-backend: ## Test backend health
	@echo "Testing backend health..."
	@curl -s http://localhost:8000/health | python -m json.tool || echo "Service not running. Start with: make run"

test-generate: ## Test text generation
	@echo "Testing text generation..."
	@curl -s -X POST http://localhost:8000/generate \
		-H "Content-Type: application/json" \
		-d '{"prompt": "The quick brown fox", "max_tokens": 50}' | \
		python -m json.tool || echo "Service not running. Start with: make run"

bench: ## Run benchmark harness (usage: make bench VARIANT=baseline_single)
	@echo "Running benchmark for $(VARIANT)..."
	@echo "TODO: Implement benchmark harness"

clean: ## Clean up temporary files
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	@echo "Cleanup complete"

lint: ## Run linting
	@echo "Running linters..."
	uv run ruff check .
	uv run mypy .

format: ## Format code
	@echo "Formatting code..."
	uv run ruff format .

typecheck: ## Run type checking
	@echo "Running type checks..."
	uv run mypy .