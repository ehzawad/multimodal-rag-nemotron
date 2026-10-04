# Deployment branch entry point

Use **[deployment/README.md](deployment/README.md)** for the API-first Triton/vLLM microservices, Docker Compose, independent model APIs, durable jobs/events, GPU-machine verification, and paired evaluation commands.

The notebook/standalone entry points in the original README are the research application. Its GPU-memory and timing results are not measurements of this new deployment. The deployment defaults to the final **direct Qwen3-VL gated-hybrid + text-rerank pipeline**; agent planning is optional. Direct had higher top-1 recall in the recorded evaluation; the final agent had slightly higher top-10 recall at higher latency.

Branch: `deployment/triton-vllm-microservices`. Based on `main` commit `bc8697c4cf5b15bc2bbdbcffb9dbc1427a850c71`. No merge into main.

**Validation:** 42 CPU contract tests passed. Docker build, live PostgreSQL/Qdrant, Triton/vLLM model execution, and CUDA capacity validation are delegated to the supplied machine-side verification script, not claimed as completed here.
