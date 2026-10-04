"""Improved multimodal recipe RAG pipeline.

Changes compared to the baseline notebook:

* Hybrid retrieval: Nemotron VL dense embeddings + BM25 lexical search, fused with Reciprocal Rank Fusion (RRF).
* Text + image queries can be combined (both are embedded and fused) instead of text silently winning.
* Image queries are rewritten into a short text description by the VLM, so they can use BM25 and the reranker too.
* Batched reranking (bounded GPU memory on a 24GB L4) that returns scores, with a fast text-only mode.
* Generation with Qwen3.5-4B (natively multimodal) that answers the user's query grounded in the retrieved recipes,
  with numbered citations, instead of a generic summary.
* Document embeddings are L2-normalised once at load time (the baseline re-normalised 10k embeddings per query).
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

# bm25s uses JAX for top-k selection when it's installed (it is on Colab) and JAX preallocates most of the GPU's
# memory, leaving none for the models. Disable preallocation in case anything imports JAX.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import torch
from PIL import Image

# Same weights as the baseline's pinned commits (identical safetensors hashes), newer remote code that supports
# transformers 5.x, which Qwen3.5 requires.
# Set MMRAG_NEMOTRON_REVISIONS=original to use the baseline's pinned commits (needs transformers 4.57.x).
EMBED_MODEL_PATH = "nvidia/llama-nemotron-embed-vl-1b-v2"
RERANK_MODEL_PATH = "nvidia/llama-nemotron-rerank-vl-1b-v2"
if os.environ.get("MMRAG_NEMOTRON_REVISIONS") == "original":
    EMBED_REVISION = "5b5ca69c35bf6ec1484d2d5ff238626e67a745e2"
    RERANK_REVISION = "47e5a355d1a050c3e5f69d53f14964b1d34bcd9d"
else:
    EMBED_REVISION = "582e3bf72aee355e3c59ed89de53543c5b0657ee"
    RERANK_REVISION = "b8a9987b05b75db5ad949c825cbcbf6eb7c48b5a"
GENERATION_MODEL_ID = "Qwen/Qwen3.5-4B"
GENERATION_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

DATASET_PATH = "mrdbourke/recipe-synthetic-images-10k"
SPACE_REPO = "mrdbourke/multimodal-rag-with-nemotron"
EMBEDDING_FILE = "image_text_embeddings_10k.safetensors"

MODALITY_TO_TOKENS = {"image": 2048, "image_text": 10240, "text": 8192}


# ----------------------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------------------

def load_embed_model(device_map="auto", attn_implementation="sdpa"):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(EMBED_MODEL_PATH,
                                      revision=EMBED_REVISION,
                                      dtype=torch.bfloat16,
                                      trust_remote_code=True,
                                      attn_implementation=attn_implementation,
                                      device_map=device_map).eval()
    model.processor.p_max_length = MODALITY_TO_TOKENS["image_text"]
    model.processor.max_input_tiles = 6
    model.processor.use_thumbnail = True
    return model


def load_rerank_model(device_map="auto", attn_implementation="sdpa", modality="image_text"):
    from transformers import AutoModelForSequenceClassification, AutoProcessor

    model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL_PATH,
                                                               revision=RERANK_REVISION,
                                                               dtype=torch.bfloat16,
                                                               trust_remote_code=True,
                                                               attn_implementation=attn_implementation,
                                                               device_map=device_map).eval()
    processor = AutoProcessor.from_pretrained(RERANK_MODEL_PATH,
                                              revision=RERANK_REVISION,
                                              trust_remote_code=True,
                                              max_input_tiles=6,
                                              use_thumbnail=True,
                                              rerank_max_length=MODALITY_TO_TOKENS[modality])
    return model, processor


def load_generation_model(device_map="auto"):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(GENERATION_MODEL_ID,
                                                        revision=GENERATION_REVISION,
                                                        dtype=torch.bfloat16,
                                                        device_map=device_map).eval()
    processor = AutoProcessor.from_pretrained(GENERATION_MODEL_ID, revision=GENERATION_REVISION)
    return model, processor


# The demo Space's precomputed embeddings only match the tutorial's pinned embed commit (5b5ca69, transformers 4.57):
# every official revision tokenizes/masks differently (cosine ~0.86 to the stored vectors). With the official revision
# we load here, the corpus is re-embedded once with scripts/build_nemotron_embeddings.py and loaded from this file.
REEMBEDDED_FILE = "nemotron_embed_vl_582e3bf_image_text_10k.safetensors"


def build_document_embeddings(dataset, embed_model, batch_size=8, text_column="recipe_markdown", image_column="image",
                              path=REEMBEDDED_FILE):
    """Embed every recipe (image + full markdown text, same as the tutorial) with the loaded embedding model."""
    from safetensors.torch import save_file
    from tqdm.auto import tqdm

    chunks = []
    for i in tqdm(range(0, len(dataset), batch_size), desc="Embedding recipes (Nemotron)"):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        with torch.inference_mode():
            chunks.append(embed_model.encode_documents(images=batch[image_column], texts=batch[text_column])
                          .to(torch.bfloat16).cpu())
    embeddings = torch.cat(chunks)
    save_file({"image_text_embeddings": embeddings.contiguous()}, path)
    return embeddings


def load_document_embeddings(path=None, device="cuda"):
    """Load image + text document embeddings, L2-normalised once so retrieval is a single matmul.

    Default: the re-embedded corpus if present (matches the official revision), else the demo Space's file
    (only matches MMRAG_NEMOTRON_REVISIONS=original)."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    if path is None:
        original = os.environ.get("MMRAG_NEMOTRON_REVISIONS") == "original"
        path = EMBEDDING_FILE if original else os.environ.get("MMRAG_EMBEDDINGS", REEMBEDDED_FILE)
    if not os.path.exists(path) and path != EMBEDDING_FILE:
        import warnings

        warnings.warn(f"{path} not found, falling back to the demo Space's precomputed embeddings. These were made "
                      f"with the tutorial's pinned embed commit and only roughly match the official revision loaded "
                      f"here (cosine ~0.86), which costs retrieval quality. Run scripts/build_nemotron_embeddings.py "
                      f"(~1.5h on an L4) to re-embed the corpus.")
        path = EMBEDDING_FILE
    if not os.path.exists(path):
        path = hf_hub_download(repo_id=SPACE_REPO, repo_type="space", filename=EMBEDDING_FILE)
    embeddings = load_file(path)["image_text_embeddings"].to(device=device, dtype=torch.float32)
    return torch.nn.functional.normalize(embeddings, dim=-1)


# ----------------------------------------------------------------------------------------------------------
# Lexical index
# ----------------------------------------------------------------------------------------------------------

_MARKDOWN_NOISE = re.compile(r"[#*_`>\[\]()|-]+")


def recipe_to_search_text(markdown: str) -> str:
    """Strip markdown syntax so BM25 matches on words rather than formatting tokens."""
    return _MARKDOWN_NOISE.sub(" ", markdown)


class BM25Index:
    def __init__(self, texts: list[str]):
        import bm25s

        self._bm25s = bm25s
        self.retriever = bm25s.BM25(backend="numpy")
        self.retriever.index(bm25s.tokenize([recipe_to_search_text(t) for t in texts], stopwords="en",
                                            show_progress=False), show_progress=False)

    def vocabulary_coverage(self, query: str) -> float:
        """Fraction of the query's (non-stopword) tokens that occur in the corpus vocabulary.

        A cheap, language-agnostic signal: an English query scores ~1.0, a Spanish/Bengali/Chinese query against this
        English corpus scores low, and BM25 on it only adds noise to the fusion."""
        tokens = self._bm25s.tokenize([query], stopwords="en", show_progress=False, return_ids=False)[0]
        if not tokens:
            return 0.0
        vocab = self.retriever.vocab_dict
        return sum(token in vocab for token in tokens) / len(tokens)

    def search(self, query: str, k: int = 100) -> tuple[list[int], list[float]]:
        tokens = self._bm25s.tokenize([query], stopwords="en", show_progress=False)
        if not tokens.vocab or all(len(ids) == 0 for ids in tokens.ids):
            return [], []
        k = min(k, self.retriever.scores["num_docs"])
        # backend_selection="numpy": the "auto" default picks JAX if installed, which grabs GPU memory
        docs, scores = self.retriever.retrieve(tokens, k=k, show_progress=False, backend_selection="numpy")
        keep = scores[0] > 0
        return docs[0][keep].tolist(), scores[0][keep].tolist()


# ----------------------------------------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------------------------------------

def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60, weights: list[float] | None = None) -> list[tuple[int, float]]:
    """Fuse ranked lists of document ids: score(d) = sum_i w_i / (k + rank_i(d))."""
    weights = weights or [1.0] * len(rankings)
    fused: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + rank + 1)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)


@dataclass
class RetrievalResult:
    indices: list[int]
    scores: list[float]
    query_text: str | None = None  # text used for BM25/reranking (the user's text or the image description)
    timings: dict = field(default_factory=dict)
    rerank_movement: list[int] | None = None  # per result: position before reranking


class MultimodalRAG:
    def __init__(self, dataset, embed_model, document_embeddings, rerank_model=None, rerank_processor=None,
                 generation_model=None, generation_processor=None, text_column="recipe_markdown",
                 image_column="image"):
        self.dataset = dataset
        self.embed_model = embed_model
        self.document_embeddings = document_embeddings
        self.rerank_model = rerank_model
        self.rerank_processor = rerank_processor
        self.generation_model = generation_model
        self.generation_processor = generation_processor
        self.text_column = text_column
        self.image_column = image_column
        self.texts = list(dataset[text_column])
        self.bm25 = BM25Index(self.texts)

    # --- dense -------------------------------------------------------------------------------------------
    @torch.inference_mode()
    def embed_query(self, query: str | Image.Image) -> torch.Tensor:
        if isinstance(query, Image.Image):
            embedding = self.embed_model.encode_documents(images=[query.convert("RGB")])
        else:
            embedding = self.embed_model.encode_queries([query])
        return torch.nn.functional.normalize(embedding.float(), dim=-1)

    def dense_search(self, query: str | Image.Image, k: int = 100) -> tuple[list[int], list[float]]:
        similarities = (self.embed_query(query) @ self.document_embeddings.T).flatten()
        scores, indices = torch.topk(similarities, k=min(k, similarities.numel()))
        return indices.tolist(), scores.tolist()

    def bm25_search(self, query: str, k: int = 100) -> tuple[list[int], list[float]]:
        return self.bm25.search(query, k=k)

    # --- query rewriting -----------------------------------------------------------------------------------
    def describe_image(self, image: Image.Image, max_new_tokens: int = 64) -> str:
        """Turn an image query into a short text search query (dish name + visible ingredients)."""
        prompt = ("You are helping search a recipe database. Look at the image and write one short search query "
                  "(under 20 words) naming the dish if it is a prepared dish, or the main visible ingredients "
                  "if it shows raw ingredients. Reply with the query only.")
        return self._generate([{"type": "image", "image": image.convert("RGB")}, {"type": "text", "text": prompt}],
                              max_new_tokens=max_new_tokens, do_sample=False).strip().strip('"')

    # --- hybrid ----------------------------------------------------------------------------------------------
    def search(self, query_text: str | None = None, query_image: Image.Image | None = None, k: int = 20,
               candidates: int = 100, use_bm25: bool = True, rewrite_image_query: bool = True,
               rrf_k: int = 60, bm25_weight: float = 1.0, bm25_min_coverage: float = 0.0) -> RetrievalResult:
        query_text = query_text.strip() if query_text and query_text.strip() else None
        if query_text is None and query_image is None:
            raise ValueError("Please provide a text query, an image query or both.")
        timings = {}

        start = time.time()
        rankings, weights = [], []
        if query_text is not None:
            rankings.append(self.dense_search(query_text, k=candidates)[0])
            weights.append(1.0)
        if query_image is not None:
            rankings.append(self.dense_search(query_image, k=candidates)[0])
            weights.append(1.0)
        timings["dense_s"] = round(time.time() - start, 3)

        lexical_query = query_text
        if lexical_query is None and rewrite_image_query and self.generation_model is not None:
            start = time.time()
            lexical_query = self.describe_image(query_image)
            timings["image_rewrite_s"] = round(time.time() - start, 3)

        if use_bm25 and lexical_query and bm25_min_coverage > 0:
            timings["bm25_coverage"] = round(self.bm25.vocabulary_coverage(lexical_query), 3)
            use_bm25 = timings["bm25_coverage"] >= bm25_min_coverage
        if use_bm25 and lexical_query:
            start = time.time()
            bm25_ranking = self.bm25_search(lexical_query, k=candidates)[0]
            if bm25_ranking:
                rankings.append(bm25_ranking)
                weights.append(bm25_weight)
            timings["bm25_s"] = round(time.time() - start, 3)

        fused = reciprocal_rank_fusion(rankings, k=rrf_k, weights=weights)[:k]
        return RetrievalResult(indices=[doc_id for doc_id, _ in fused], scores=[score for _, score in fused],
                               query_text=lexical_query, timings=timings)

    # --- rerank ----------------------------------------------------------------------------------------------
    @torch.inference_mode()
    def rerank(self, query_text: str, result: RetrievalResult, top_n: int = 20, batch_size: int = 4,
               use_images: bool = True) -> RetrievalResult:
        """Rerank the first `top_n` results with the cross-encoder, in batches to bound GPU memory.

        use_images=False reranks on recipe text only (without reviews), which is several times faster."""
        if self.rerank_model is None:
            raise ValueError("No rerank model loaded.")
        start = time.time()
        indices = result.indices[:top_n]
        samples = self.dataset.select(indices)
        texts = samples[self.text_column]
        if use_images:
            images = samples[self.image_column]
        else:
            # Text-only mode: also drop the user reviews (about half of each recipe's tokens) for speed
            texts = [text.split("\n## Reviews")[0] for text in texts]
            images = [""] * len(indices)

        logits = []
        for i in range(0, len(indices), batch_size):
            examples = [{"question": query_text, "doc_text": text, "doc_image": image}
                        for text, image in zip(texts[i:i + batch_size], images[i:i + batch_size])]
            batch = self.rerank_processor.process_queries_documents_crossencoder(examples)
            batch = {k: v.to(self.rerank_model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits.append(self.rerank_model(**batch, return_dict=True).logits.float().view(-1).cpu())
        logits = torch.cat(logits)
        order = torch.argsort(logits, descending=True).tolist()

        timings = dict(result.timings, rerank_s=round(time.time() - start, 3))
        return RetrievalResult(indices=[indices[i] for i in order] + result.indices[top_n:],
                               scores=[logits[i].item() for i in order] + result.scores[top_n:],
                               query_text=result.query_text, timings=timings,
                               rerank_movement=order + list(range(top_n, len(result.indices))))

    # --- generation ------------------------------------------------------------------------------------------
    @torch.inference_mode()
    def _generate(self, content: list[dict], max_new_tokens: int = 512, do_sample: bool = True) -> str:
        messages = [{"role": "user", "content": content}]
        inputs = self.generation_processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                               return_dict=True, return_tensors="pt",
                                                               enable_thinking=False)
        inputs = inputs.to(self.generation_model.device)
        # Qwen3.5 recommended sampling settings for non-thinking mode
        sampling = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20) if do_sample else dict(do_sample=False)
        output_ids = self.generation_model.generate(**inputs, max_new_tokens=max_new_tokens, **sampling)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.generation_processor.decode(new_tokens, skip_special_tokens=True).strip()

    def answer(self, query_text: str | None, result: RetrievalResult, query_image: Image.Image | None = None,
               num_recipes: int = 3, max_new_tokens: int = 512) -> str:
        """Answer the user's query using only the top retrieved recipes, citing them as [1], [2], ..."""
        recipes = "\n\n".join(f"[{n}] {self.texts[idx][:6000]}"
                              for n, idx in enumerate(result.indices[:num_recipes], 1))
        request = query_text or "Recipes matching the attached image"
        prompt = f"""You are a helpful culinary assistant. A user searched a recipe database with this request:

"{request}"

Here are the top {num_recipes} retrieved recipes:

<recipes>
{recipes}
</recipes>

Using only these recipes, write a concise markdown answer:
- Start with one sentence saying which recipe best fits the request and why.
- Then one short section per recipe: a 1-2 sentence overview, key ingredients, total time and an Easy/Medium/Hard difficulty.
- Cite recipes by their number, e.g. [1].
- If none of the recipes fit the request, say so plainly."""
        content = ([{"type": "image", "image": query_image.convert("RGB")}] if query_image is not None else [])
        content.append({"type": "text", "text": prompt})
        return self._generate(content, max_new_tokens=max_new_tokens)
