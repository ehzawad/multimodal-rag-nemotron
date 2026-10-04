"""Experiment: Qwen3-VL-Embedding-2B + Qwen3-VL-Reranker-2B as a drop-in replacement for the Nemotron VL models.

Both are multimodal (text, image, text + image) and multilingual (30+ languages), built on Qwen3-VL-2B
(Jan 2026, https://arxiv.org/abs/2601.04720). The 10k recipe corpus has to be re-embedded once with
`build_document_embeddings` (image + recipe text per document).
"""
from __future__ import annotations

import os
import time

import torch
from PIL import Image

from . import pipeline as P

EMBED_MODEL_PATH = "Qwen/Qwen3-VL-Embedding-2B"
EMBED_REVISION = "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
RERANK_MODEL_PATH = "Qwen/Qwen3-VL-Reranker-2B"
RERANK_REVISION = "4bd860ac4f15ad1897a214615cccc700f8f71818"

QUERY_PROMPT = "Retrieve recipes (recipe card image and text) relevant to the user's search query."
RERANK_PROMPT = "Judge whether the recipe matches the user's recipe search query."
EMBEDDING_FILE = "qwen3_vl_embedding_2b_image_text_10k.safetensors"

# The recipe card images are 1275x1650; cap the image tokens (~400) and drop the user reviews from the text
# (about half of each recipe's tokens) so embedding the corpus takes ~30-40 min on an L4 instead of ~2h
IMAGE_MAX_PIXELS = 400 * 28 * 28


def _processor_kwargs():
    return {"max_pixels": IMAGE_MAX_PIXELS, "min_pixels": 64 * 28 * 28}


def load_embed_model(device="cuda"):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL_PATH, revision=EMBED_REVISION, device=device,
                               model_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"},
                               processor_kwargs=_processor_kwargs())


def load_rerank_model(device="cuda"):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL_PATH, revision=RERANK_REVISION, device=device,
                        model_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"},
                        processor_kwargs=_processor_kwargs())


def strip_reviews(text: str) -> str:
    return text.split("\n## Reviews")[0]


def _document(text: str, image: Image.Image | None) -> dict:
    document = {"text": strip_reviews(text)}
    if image is not None:
        document["image"] = image.convert("RGB")
    return document


def build_document_embeddings(dataset, embed_model, batch_size=8, text_column="recipe_markdown", image_column="image",
                              path=EMBEDDING_FILE):
    """Embed every recipe as image + text (one-off, ~30 min for 10k recipes on an L4) and save to `path`."""
    from safetensors.torch import save_file
    from tqdm.auto import tqdm

    chunks = []
    start = time.time()
    for i in tqdm(range(0, len(dataset), batch_size), desc="Embedding recipes (Qwen3-VL)"):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        documents = [_document(text, image) for text, image in zip(batch[text_column], batch[image_column])]
        with torch.inference_mode():
            chunks.append(embed_model.encode(documents, batch_size=batch_size, convert_to_tensor=True,
                                             normalize_embeddings=True).to(torch.float16).cpu())
    embeddings = torch.cat(chunks)
    save_file({"image_text_embeddings": embeddings.contiguous()}, path)
    print(f"[INFO] Embedded {len(embeddings)} recipes in {time.time() - start:.0f}s -> {path}")
    return embeddings


def load_document_embeddings(path=EMBEDDING_FILE, device="cuda"):
    from safetensors.torch import load_file

    embeddings = load_file(path)["image_text_embeddings"].to(device=device, dtype=torch.float32)
    return torch.nn.functional.normalize(embeddings, dim=-1)


class QwenVLRAG(P.MultimodalRAG):
    """Same pipeline (hybrid retrieval, rewriting, generation) with Qwen3-VL embedding + reranking models."""

    @torch.inference_mode()
    def embed_query(self, query: str | Image.Image) -> torch.Tensor:
        if isinstance(query, Image.Image):
            embedding = self.embed_model.encode([{"image": query.convert("RGB")}], convert_to_tensor=True,
                                                prompt=QUERY_PROMPT)
        else:
            embedding = self.embed_model.encode([query], convert_to_tensor=True, prompt=QUERY_PROMPT)
        return torch.nn.functional.normalize(embedding.float(), dim=-1)

    @torch.inference_mode()
    def rerank(self, query_text: str, result: P.RetrievalResult, top_n: int = 20, batch_size: int = 4,
               use_images: bool = True) -> P.RetrievalResult:
        start = time.time()
        indices = result.indices[:top_n]
        samples = self.dataset.select(indices)
        texts = samples[self.text_column]
        if use_images:
            documents = [_document(text, image) for text, image in zip(texts, samples[self.image_column])]
        else:
            documents = [strip_reviews(text) for text in texts]
        scores = torch.as_tensor(self.rerank_model.predict([(query_text, document) for document in documents],
                                                           batch_size=batch_size, prompt=RERANK_PROMPT,
                                                           show_progress_bar=False), dtype=torch.float32).view(-1)
        order = torch.argsort(scores, descending=True).tolist()
        timings = dict(result.timings, rerank_s=round(time.time() - start, 3))
        return P.RetrievalResult(indices=[indices[i] for i in order] + result.indices[top_n:],
                                 scores=[scores[i].item() for i in order] + result.scores[top_n:],
                                 query_text=result.query_text, timings=timings,
                                 rerank_movement=order + list(range(top_n, len(result.indices))))
