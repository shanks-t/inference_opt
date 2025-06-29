# Inference‑Optimisation Lab – Repo Blueprint

> **Purpose**
> A modular playground for measuring how batching strategies (none → static → dynamic → *continuous*) affect LLM inference latency and throughput on any hardware.  The initial target runtime is **llama.cpp** running on Apple‑Silicon, but every component is hardware‑agnostic so you can swap back‑ends later (e.g. CUDA, ROCm).

---

## 1. High‑level Goals

| Goal                       | Why it matters                                                                          |
| -------------------------- | --------------------------------------------------------------------------------------- |
| **Deterministic load‑gen** | Replays the *exact* same query mix for every service variant so results are comparable. |
| **Pluggable runtimes**     | Single interface → same tests run on CPU, Metal, CUDA, or CoreML.                       |
| **Single‑source metrics**  | TTFT, tokens/s, p50 & p95 latencies collected identically across variants.              |
| **Lean tooling**           | Use **FastAPI**, **uv** for deps, and minimal Python; avoid heavyweight stacks.         |
| **One‑command benchmarks** | `make bench VARIANT=dynamic` spins up the service, fires the harness, exports results.  |

---

## 2. Repository Layout

```
.
├── data/
│   └── prompts.jsonl         # frozen request corpus (generated once)
├── services/
│   ├── baseline_single/      # no batching
│   ├── static_batch/
│   ├── dynamic_batch/
│   └── (continuous_batch/)   # placeholder for future work
├── harness/
│   ├── client.py             # async load‑gen + metric timers
│   ├── run_bench.py          # CLI wrapper: run -> save parquet
│   └── grafana_dash.json     # importable dashboard definition
├── adapters/                 # runtime adapters (llama_cpp, onnx, …)
│   └── llama_cpp.py
├── docker/
│   └── Dockerfile.arm64
├── requirements.lock         # generated via `uv pip compile`
└── README.md (this doc)
```

---

## 3. Core Modules

### 3.1 Query‑Corpus Generator (`tools/make_prompts.py`)

* Generates `prompts.jsonl` with variable‑length prompts.
* Uses a small HF model on MPS/CPU so generation finishes in <1 min.
* Script is **idempotent**; CI fails if the file drifts.

### 3.2 Runtime Adapter (`adapters/`)

```python
class InferenceBackend(Protocol):
    async def generate(self, prompts: list[str], max_tokens: int) -> list[str]:
        """Return one generated string per prompt (ordered)."""
```

* First implementation: `LlamaCppBackend` (quantised GGUF, optional Metal offload).
* Later: `VllmBackend`, `CoremlBackend`, etc.—each lives behind the same interface.

### 3.3 Service Variants (`services/`)

* **baseline\_single** – one prompt → one `llama.cpp` call, streamed SSE.
* **static\_batch**    – client‑side chunking; concat N prompts, single backend call.
* **dynamic\_batch**   – server‑side `asyncio.Queue` with `batch_wait_ms`.
* **continuous\_batch** – *todo* once Metal‑ready implementation lands.

Every service exposes **`POST /generate`** (OpenAI‑style JSON) so the harness never changes.

### 3.4 Benchmark Harness (`harness/`)

1. Reads `prompts.jsonl`.
2. Launches N concurrent workers (configurable QPS).
3. Captures per‑request:

   * `ttft_ms`
   * `tokens_per_sec`
   * `total_ms`
4. Writes `<variant>/<timestamp>.parquet` with raw rows.
5. Optional: pushes counters to **Prometheus**; Grafana dashboard reads both parquet & Prometheus.

---

## 4. Metrics Definitions

| Metric            | Description                                                 |
| ----------------- | ----------------------------------------------------------- |
| **TTFT**          | Wall‑clock ms from HTTP request until first streamed token. |
| **Tokens/s**      | `(len(gen_tokens) – 1) / (t_end – t_first_tok)`             |
| **p50 / p95 lat** | Median / 95th percentile of `total_ms` across run.          |

All timing captured **client‑side** to include network + ASGI overhead.

---

## 5. Hardware Abstraction

* **env.yml** defines runtime settings: `BACKEND=llama_cpp`, `N_THREADS`, `N_GPU_LAYERS` …
* Docker images are built for **arm64** by default; switch `--platform` to `linux/amd64` for GPU rigs.
* Backend factories read env vars → instantiate the proper adapter.

---

## 6. Dependency & Build Strategy

* **uv** for deterministic, lock‑file‑driven installs (replaces pip, poetry, pyenv, virtualenv).

  ```bash
  # Project initialization
  uv init inference_opt
  uv python pin 3.11
  
  # Dependency management
  uv add torch transformers fastapi uvicorn
  uv add --dev pytest ruff mypy
  uv lock
  uv sync
  
  # Run commands
  uv run python tools/make_prompts.py
  uv run fastapi dev services/baseline_single/main.py
  ```
* Each service folder has its own `pyproject.toml`; top‑level lock captures *all*.
* CI caches `~/.cache/uv` for speedy re‑installs.
* Reference: https://docs.astral.sh/uv/

---

## 7. Makefile Commands

```makefile
make dev           # create venv & pre‑commit hooks
make run SERVICE=baseline_single  # start service locally
make bench VARIANT=static_batch   # run harness & store parquet
make dash          # start Grafana with pre‑wired board
make docker        # build arm64 image with uv deps
```

---

## 8. Continuous Integration (GitHub Actions)

* **lint** – ruff + mypy
* **unit‑test** – pytest (fast mocks of back‑end)
* **bench‑nightly** – 02:00 UTC: build image → run `make bench` for each variant → upload parquet artefacts → trigger Grafana image render posted as PR comment.
* **regen‑prompts** job fails if `data/prompts.jsonl` differs from main.

---

## 9. Extension Hooks

| Idea                    | Drop‑in location                                                                      |
| ----------------------- | ------------------------------------------------------------------------------------- |
| **Continuous batching** | `services/continuous_batch/` once llama.cpp or MLC‑LLM supports Metal tokens ⇆ batch. |
| **CoreML / ONNX**       | `adapters/coreml.py`; mention in `BACKEND` env.                                       |
| **Cost modelling**      | Add `price_per_token_usd` to harness summary, param per hardware type.                |

---

## 10. Simplicity Principles

1. **One interface, many back‑ends** – keeps harness & dashboards constant.
2. **No hidden state** – all variants read config via env vars or CLI flags; nothing hard‑coded.
3. **Lightweight Python** – FastAPI + uvicorn; avoid Ray, Celery, etc., until justified.
4. **Quantise early** – default models are q4\_K\_M GGUF so they fit in <6 GiB.
5. **Fail fast in CI** – schema checks on prompts, lock‑file drift, or missing metrics.

---

## 11. Getting Started (developer)

```bash
# clone & bootstrap
brew install uv
make dev                   # creates .venv & installs hooks
python tools/make_prompts.py --n 500  # only if you need to regen corpus
make run SERVICE=baseline_single     # start FastAPI on :8000
make bench VARIANT=baseline_single   # run quick benchmark
open http://localhost:3000           # Grafana board
```

---

## 12. FAQ

**Q – Can I run this on a Linux + A100 box?**
*A – Yes.* Install CUDA build of llama.cpp or swap `BACKEND=vllm`.  The harness & dashboards stay untouched.

**Q – How do I add a new batching strategy?**
Fork `services/baseline_single`, implement the batching, keep `/generate` contract; add its name to `VARIANTS` list in `run_bench.py`.

---

*Happy hacking – may your p95s be ever low!*
