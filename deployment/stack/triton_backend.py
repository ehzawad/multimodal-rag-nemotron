"""Real Triton Python backend for pinned Qwen3-VL embedding/reranking weights.

The HTTP adapters convert domain JSON to BYTES tensors. We combine compatible
requests inside execute() and let Sentence Transformers bound the GPU microbatch.
No fake inference path exists in this module.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
from .common import gpu_slot
from .contracts import (EMBED_DIM, EMBED_MODEL, EMBED_REVISION, IMAGE_MAX_PIXELS,
                        QUERY_PROMPT, RERANK_MODEL, RERANK_PROMPT, RERANK_REVISION, Media)


class TritonPythonModel:
    def initialize(self, args):
        import torch
        from importlib.metadata import version
        for package, expected in [('sentence-transformers', '6.1.0'), ('transformers', '5.2.0'),
                                  ('torch', '2.11.0'), ('torchvision', '0.26.0')]:
            if version(package).split('+')[0] != expected:
                raise RuntimeError(f'{package} does not match the declared embedding toolchain')
        from sentence_transformers import CrossEncoder, SentenceTransformer
        self.kind = args["model_name"]
        if self.kind not in {"embed", "rerank"}:
            raise ValueError("unknown model role")
        self.microbatch = int(os.environ.get("MODEL_MICROBATCH", "2"))
        if not 1 <= self.microbatch <= 8:
            raise ValueError("MODEL_MICROBATCH must be 1..8")
        device = f"cuda:{int(args['model_instance_device_id'])}"
        torch.cuda.set_device(device)
        kwargs = dict(device=device, model_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"},
                      processor_kwargs={"max_pixels": IMAGE_MAX_PIXELS, "min_pixels": 64 * 28 * 28})
        with gpu_slot(timeout=1800):
            if self.kind == "embed":
                self.model = SentenceTransformer(EMBED_MODEL, revision=EMBED_REVISION, **kwargs)
                if self.model.get_sentence_embedding_dimension() != EMBED_DIM:
                    raise ValueError("embedding dimension mismatch")
            else:
                self.model = CrossEncoder(RERANK_MODEL, revision=RERANK_REVISION, **kwargs)

    def execute(self, requests):
        import torch
        import triton_python_backend_utils as pb
        output = [None] * len(requests)
        arrays, groups = {}, defaultdict(list)
        for i, request in enumerate(requests):
            try:
                if request.is_cancelled():
                    output[i] = pb.InferenceResponse(error=pb.TritonError("cancelled", pb.TritonError.CANCELLED))
                    continue
                values = pb.get_input_tensor_by_name(request, "INPUT").as_numpy().reshape(-1)
                arrays[i] = np.empty((len(values), EMBED_DIM if self.kind == "embed" else 1), dtype=np.float32)
                for j, raw in enumerate(values):
                    data = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
                    if self.kind == "embed":
                        media = Media.model_validate(data["input"]).model_input()
                        role = data.get("input_type", "query")
                        if role not in {"query", "document"}:
                            raise ValueError("invalid input_type")
                        instruction = data.get("instruction")
                        prompt = instruction if instruction is not None else (QUERY_PROMPT if role == "query" else None)
                        groups[prompt].append((i, j, media))
                    else:
                        pair = (Media.model_validate(data["query"]).model_input(),
                                Media.model_validate(data["document"]).model_input())
                        groups[data.get("instruction", RERANK_PROMPT)].append((i, j, pair))
            except (ValueError, KeyError, TypeError):
                output[i] = pb.InferenceResponse(error=pb.TritonError("invalid model input", pb.TritonError.INVALID_ARG))
        try:
            with gpu_slot(), torch.inference_mode():
                for prompt, rows in groups.items():
                    rows = [r for r in rows if output[r[0]] is None]
                    if not rows:
                        continue
                    inputs = [r[2] for r in rows]
                    if self.kind == "embed":
                        encoded = self.model.encode(inputs, batch_size=self.microbatch, convert_to_tensor=True,
                                                    normalize_embeddings=True, prompt=prompt,
                                                    show_progress_bar=False).float()
                        values = torch.nn.functional.normalize(encoded, dim=-1).cpu().numpy()
                    else:
                        values = np.asarray(self.model.predict(inputs, batch_size=self.microbatch,
                                                               prompt=prompt, show_progress_bar=False),
                                            dtype=np.float32).reshape(-1, 1)
                    if not np.isfinite(values).all():
                        raise RuntimeError("non-finite model output")
                    for (i, j, _), value in zip(rows, values, strict=True):
                        arrays[i][j] = value
            for i, array in arrays.items():
                if output[i] is None:
                    output[i] = pb.InferenceResponse(output_tensors=[pb.Tensor("OUTPUT", array)])
        except TimeoutError:
            for i in range(len(output)):
                if output[i] is None:
                    output[i] = pb.InferenceResponse(error=pb.TritonError("GPU_BUSY", pb.TritonError.UNAVAILABLE))
        except Exception as exc:
            # Backend logs contain the traceback; the API deliberately does not expose model internals.
            pb.Logger.log_error(f"{self.kind} inference failed: {type(exc).__name__}: {exc}")
            for i in range(len(output)):
                if output[i] is None:
                    output[i] = pb.InferenceResponse(error=pb.TritonError("model execution failed"))
        return output

    def finalize(self):
        self.model = None
