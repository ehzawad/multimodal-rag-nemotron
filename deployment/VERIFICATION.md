# Verification record

Prepared 2026-10-05. Research source: `bc8697c4cf5b15bc2bbdbcffb9dbc1427a850c71`.

## Executed in the authoring environment

- `PYTHONPATH=. python -m pytest -q tests`: **42 passed**, final pre-publication run 0.63 seconds.
- `python -m compileall -q stack manage.py start-vllm.py triton tests`: passed.
- `bash -n verify-machine.sh`: passed.
- Compose YAML was parsed and topology invariants exercised by CPU tests; this is not a successful `docker compose config` or image build.

Environment: Python 3.13.5 on Linux, CPU only. Test dependencies: FastAPI 0.128.2, Starlette 0.50.0, HTTPX 0.28.1, Pydantic 2.13.4, NumPy 2.3.5, Pillow 12.3.0, SQLAlchemy 2.0.50, pytest 9.0.2, PyYAML 6.0.3. The supplied containers use their own Python 3.12 environment and dependency installs, which were not built here.

Coverage includes input/media bounds, forbidden URL/path inputs, request-body limits and redaction, HTTP authentication and administrative writes, model tensor contracts and batch splitting, shared-host admission/lock behavior, immutable ingestion and publication, embedding-identity mismatches, corpus snapshot checksums, metadata catalog pagination, original-query-preserving agent fusion, rerank permutations and citations, HTTP deadline handling, job/event idempotency, retries, lease expiry/fencing, worker API routing for every supported job kind, SSE replay, and independent-service Docker topology.

Explicit test doubles replace model servers, Qdrant, and BM25. SQLite exercises the job state machine in CPU tests; it does not prove PostgreSQL `SKIP LOCKED` concurrency. There is no mock inference fallback in the deployed server paths.

## Not executed here

Docker image builds/pulls, Docker Compose runtime/config resolution, NVIDIA driver and container CUDA probes, real model downloads/inference, Triton Python-stub compatibility, vLLM memory allocation/tool parsing, live PostgreSQL and Qdrant integration, full-corpus ingestion, evaluation, latency/load measurement, or backup/restore tests. No 24 GB capacity guarantee, serving speedup, production readiness certification, or reproduction of the historical report's numbers is asserted.

## Required machine verification

Run `bash verify-machine.sh` from `deployment/` in a host virtual environment with `requirements-tools.txt` installed. This builds the supplied images, probes the real container CUDA runtimes and vLLM flags, records image identities, starts services, and runs real API/model smoke tests. Then index the full corpus and run `manage.py eval` and `manage.py benchmark` as documented in README.

Inspect and retain `artifacts/doctor.json`, `images.json`, `smoke.json`, the corpus manifest, `evaluation.json`, and `benchmark.json`. They are generated only by actual machine-side execution. Their absence is not a passing validation result. The historical `results/` directory belongs to the research implementation and is not newly generated deployment evidence.
