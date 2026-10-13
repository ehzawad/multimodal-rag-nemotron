# Multimodal RAG: deployment branch

**Start here: [deployment/README.md](deployment/README.md).** This branch adds independently callable embedding, reranking, generation, semantic-search, RAG, and durable job/event APIs, with Triton Python model backends, vLLM generation, Qdrant, PostgreSQL, and Docker Compose.

The default is the final direct Qwen3-VL gated-hybrid + text-rerank pipeline. Agent planning is optional, not assumed to be more accurate. The historical report has higher top-1 recall for direct mode and slightly higher top-10 recall for the final agent, at higher latency. See the deployment guide for the exact comparison and evaluation scope.

**Validation: 42 CPU contract tests passed. Docker/CUDA/model-server execution has not been performed in the authoring environment.** Run the supplied machine-side verification script before relying on the deployment. The 24 GB shared-GPU configuration is an unbenchmarked starting point, not a memory-fit guarantee.

```bash
cd deployment
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tools.txt
python manage.py init-env
bash verify-machine.sh
python manage.py seed-recipes --index recipes-v1
```

To start only the independently usable reranker, after generating `.env`:

```bash
docker compose up -d --build edge reranking
```

Research code remains under `mmrag/`, `standalone/`, `notebooks/`, and `scripts/`. Recorded experiments remain under [`results/`](results/) and [`reports/report.pdf`](reports/report.pdf). Those results do not measure this new deployment. The original in-process demo can still be run with `pip install -r requirements.txt` followed by `python standalone/app.py`.

Branch: `deployment/triton-vllm-microservices`, based on main commit `bc8697c4cf5b15bc2bbdbcffb9dbc1427a850c71`. This work is not merged into main.
