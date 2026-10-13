"""Model APIs: independently callable adapters; they never import the RAG application."""
from __future__ import annotations

import json
import os
import time

import numpy as np
from fastapi import HTTPException
from .common import Gate, HTTP, RemoteError, gpu_slot
from .contracts import (ChatRequest, EmbedRequest, EMBED_DIM, GEN_MODEL, GEN_REVISION,
                        RerankRequest, RERANK_MODEL, RERANK_REVISION, embedding_identity)


class ModelAPI:
    def __init__(self, role: str, http: HTTP):
        self.role, self.http = role, http
        self.base = os.environ.get("MODEL_URL", {"embeddings": "http://triton-embed:8000",
                                               "reranking": "http://triton-rerank:8000",
                                               "generation": "http://vllm:8000"}[role])
        active = int(os.environ.get("MODEL_API_CONCURRENCY", "1"))
        if not 1 <= active <= 16:
            raise ValueError("MODEL_API_CONCURRENCY must be 1..16")
        self.gate = Gate(active=active, waiting=4, timeout=5)

    def ready(self):
        path = "/health" if self.role == "generation" else f"/v2/models/{'embed' if self.role == 'embeddings' else 'rerank'}/ready"
        # Triton/vLLM health endpoints need not return JSON.
        try:
            response = self.http.client.get(self.base + path, timeout=5,
                                            headers={"Authorization": "Bearer " + os.environ.get("API_KEY", "")})
            return response.status_code == 200
        except Exception:
            return False

    def infer(self, kind, values, dimensions):
        result = []
        # Triton max_batch_size is eight. Larger public batches are split, never sent illegally.
        for start in range(0, len(values), 8):
            rows = values[start:start + 8]
            payload = {"inputs": [{"name": "INPUT", "shape": [len(rows), 1], "datatype": "BYTES",
                                   "data": [json.dumps(v, ensure_ascii=False) for v in rows]}],
                       "outputs": [{"name": "OUTPUT"}]}
            response = self.http.call("POST", self.base + f"/v2/models/{kind}/versions/1/infer", payload)
            try:
                tensor = next(t for t in response["outputs"] if t["name"] == "OUTPUT")
                if tensor["shape"] != [len(rows), dimensions] or tensor["datatype"] != "FP32":
                    raise ValueError("shape or type mismatch")
                array = np.asarray(tensor["data"], dtype=np.float32).reshape(len(rows), dimensions)
                if not np.isfinite(array).all():
                    raise ValueError("non-finite output")
                result.extend(array.tolist())
            except (KeyError, StopIteration, ValueError) as exc:
                raise RemoteError(502, "triton-contract") from exc
        return result

    def embed(self, request: EmbedRequest):
        start = time.monotonic()
        with self.gate.enter():
            rows = [{"input": m.model_dump(exclude_none=True), "input_type": request.input_type,
                     "instruction": request.instruction} for m in request.inputs]
            vectors = self.infer("embed", rows, EMBED_DIM)
        return {"identity": embedding_identity(), "input_type": request.input_type,
                "embeddings": vectors, "elapsed_s": time.monotonic() - start}

    def rerank(self, request: RerankRequest):
        start = time.monotonic()
        with self.gate.enter():
            rows = [{"query": request.query.model_dump(exclude_none=True),
                     "document": d.model_dump(exclude_none=True), "instruction": request.instruction}
                    for d in request.documents]
            scores = [v[0] for v in self.infer("rerank", rows, 1)]
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:request.top_k]
        return {"model": RERANK_MODEL, "revision": RERANK_REVISION, "score_type": "reranker_logit",
                "results": [{"index": i, "score": scores[i]} for i in order],
                "elapsed_s": time.monotonic() - start}

    def chat(self, request: ChatRequest):
        payload = request.model_dump(exclude_none=True)
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            with self.gate.enter(), gpu_slot():
                result = self.http.call("POST", self.base + "/v1/chat/completions", payload,
                                        headers={"Authorization": "Bearer " + os.environ.get("API_KEY", "")})
        except TimeoutError as exc:
            raise HTTPException(429, "shared GPU busy", headers={"Retry-After": "2"}) from exc
        if not result.get("choices"):
            raise RemoteError(502, "vllm-contract")
        result["deployment"] = {"model": GEN_MODEL, "revision": GEN_REVISION, "engine": "vllm"}
        return result
