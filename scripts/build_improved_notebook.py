"""Build notebooks/multimodal_rag_improved.ipynb (the improved pipeline walkthrough)."""
import json
from pathlib import Path

DST = Path(__file__).resolve().parent.parent / "notebooks" / "multimodal_rag_improved.ipynb"
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


md("""
# Multimodal RAG with Nemotron VL: improved pipeline

This notebook builds on the [baseline tutorial](multimodal_rag_tutorial.ipynb) (recipe search over
[`mrdbourke/recipe-synthetic-images-10k`](https://huggingface.co/datasets/mrdbourke/recipe-synthetic-images-10k)
with text or image queries) and improves the pipeline:

| Stage | Baseline | Improved |
|---|---|---|
| Retrieval | Dense only (Nemotron Embed VL) | **Hybrid**: dense + BM25, fused with Reciprocal Rank Fusion |
| Query types | Text *or* image (text silently wins if both) | Text, image, or **both combined** |
| Image queries | Dense only, can't be reranked | **Rewritten to text** by the VLM so they can use BM25 + reranking |
| Reranking | All 20 image+text pairs in one batch (OOM on a 24GB L4) | **Batched**, returns scores, optional fast text-only mode |
| Generation | Qwen3-VL-2B, generic summary | **Qwen3.5-4B**, answers the query grounded in the results with **citations**, sees the query image |
| Evaluation | Vibes | **Recall@k / MRR** on title and paraphrased queries (`scripts/eval_retrieval.py`) |

The pipeline code lives in [`mmrag/pipeline.py`](../mmrag/pipeline.py). Runs on a single 24GB GPU (Colab L4).
""")

md("## Install dependencies")
code("""
# Qwen3.5 needs transformers 5.x (Colab ships it, but the baseline notebook pins 4.57.3 so upgrade if needed)
!pip install -q -U "transformers>=5.2" "huggingface_hub>=1.0" "accelerate>=1.12" "datasets>=4.4" bm25s "gradio>=5"
""")
code("""
import os
import sys
from pathlib import Path

# Make the `mmrag` package importable (works from the repo's notebooks/ folder or a fresh Colab runtime)
REPO_URL = "https://github.com/ehzawad/multimodal-rag-nemotron"
for candidate in [Path.cwd(), Path.cwd().parent, Path("multimodal-rag-nemotron"), Path("/content/mmrag")]:
    if (candidate / "mmrag" / "pipeline.py").exists():
        REPO_DIR = candidate.resolve()
        break
else:
    !git clone -q {REPO_URL}
    REPO_DIR = Path("multimodal-rag-nemotron").resolve()
sys.path.insert(0, str(REPO_DIR))
print(f"[INFO] Using pipeline code from: {REPO_DIR}")
""")

md("""
## Load dataset, embeddings and models

> **Embedding compatibility:** the demo Space's precomputed document embeddings only match the tutorial's pinned
> Nemotron commit (`5b5ca69`), which needs transformers 4.57. Every official revision (needed for transformers 5 /
> Qwen3.5) tokenizes and masks differently: re-embedding a document gives cosine ≈ 0.86 to its stored vector.
> We therefore re-embed the corpus once with the official revision (`scripts/build_nemotron_embeddings.py`, ~2h on an
> L4) and load that file. Without it, the pipeline falls back to the Space's file with a warning (lower accuracy).
""")
code("""
%%time
import time
import torch
import transformers
from datasets import load_dataset
from PIL import Image

from mmrag import pipeline as P

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE} | transformers: {transformers.__version__}")

dataset = load_dataset(P.DATASET_PATH, split="train")
document_embeddings = P.load_document_embeddings(device=DEVICE)  # downloaded from the demo Space, L2-normalised once
print(f"[INFO] Samples: {len(dataset)} | Document embeddings: {tuple(document_embeddings.shape)}")

embed_model = P.load_embed_model()
rerank_model, rerank_processor = P.load_rerank_model(modality="image_text")
generation_model, generation_processor = P.load_generation_model()
print(f"[INFO] GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
""")
code("""
%%time
# Builds the BM25 index over the recipe texts (a few seconds for 10k recipes)
rag = P.MultimodalRAG(dataset, embed_model, document_embeddings, rerank_model, rerank_processor,
                      generation_model, generation_processor)
""")

md("## Helpers for displaying results")
code("""
import base64
import html
import io

from IPython.display import HTML, display


def _thumbnail_b64(image, width=220):
    image = image.copy()
    image.thumbnail((width, width * 2))
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=80)
    return base64.b64encode(buffer.getvalue()).decode()


def show_results(result, title="", n=5, show_images=True):
    samples = dataset.select(result.indices[:n])
    cards = []
    for rank, (sample, score) in enumerate(zip(samples, result.scores[:n]), 1):
        moved = ""
        if result.rerank_movement is not None:
            before = result.rerank_movement[rank - 1] + 1
            moved = f" (was #{before})" if before != rank else " (unchanged)"
        image_html = f"<img src='data:image/jpeg;base64,{_thumbnail_b64(sample['image'])}' style='width:100%'>" if show_images else ""
        cards.append(f"<div style='flex:1;min-width:0;font-family:system-ui;font-size:12px'>{image_html}"
                     f"<b>#{rank}{moved}</b> {html.escape(sample['name'])}<br><span style='color:#888'>score {score:.4f}</span></div>")
    rewritten = f"<div style='color:#888;font-size:12px'>Text used for BM25/rerank: <i>{html.escape(result.query_text)}</i></div>" if result.query_text else ""
    display(HTML(f"<h4 style='font-family:system-ui;margin:12px 0 4px'>{html.escape(title)}</h4>{rewritten}"
                 f"<div style='display:flex;gap:10px'>{''.join(cards)}</div>"
                 f"<div style='color:#888;font-size:12px'>timings: {result.timings}</div>"))
""")

md("""
## Hybrid retrieval vs. dense only

Dense embeddings capture meaning ("summer lunch ideas"), BM25 captures exact words (an ingredient or dish name).
Reciprocal Rank Fusion combines the two ranked lists without needing to calibrate their scores:
`score(d) = Σ 1 / (60 + rank_i(d))`.
""")
code("""
for query in ["recipes with tomato and parmesan cheese", "egg sandwich", "summer lunch ideas"]:
    dense = P.RetrievalResult(*rag.dense_search(query, k=20), query_text=query)
    show_results(dense, f"Baseline (dense only): {query}", show_images=False)
    show_results(rag.search(query_text=query, k=20), f"Improved (hybrid): {query}", show_images=False)
""")

md("""
## Batched reranking

The baseline reranked 20 image + text pairs in a single forward pass, which runs out of memory on a 24GB L4.
Batching keeps memory bounded. Reranking on recipe *text only* is much faster and often just as good, since the
recipe images in this dataset are rendered versions of the same recipe text.
""")
code("""
query = "Banana Sour Cream Bread"
hybrid_result = rag.search(query_text=query, k=20)
show_results(hybrid_result, f"Hybrid: {query}")
show_results(rag.rerank(query, hybrid_result, top_n=20, use_images=False), f"Hybrid + rerank (text only): {query}")
show_results(rag.rerank(query, hybrid_result, top_n=20, use_images=True), f"Hybrid + rerank (image + text): {query}")
""")

md("""
## Image queries (rewritten to text) and combined text + image queries

For an image query, Qwen3.5-4B writes a short text description of the image (dish name or visible ingredients).
That text then powers BM25 and the reranker, which previously only worked for text queries.
""")
code("""
from huggingface_hub import hf_hub_download

query_image = Image.open(hf_hub_download(P.SPACE_REPO, "kitchen_bench.png", repo_type="space")).convert("RGB")
display(query_image.resize((400, int(400 * query_image.height / query_image.width))))

dense_image = P.RetrievalResult(*rag.dense_search(query_image, k=20))
show_results(dense_image, "Baseline (dense only): image query")
image_result = rag.search(query_image=query_image, k=20)
show_results(image_result, "Improved: image query (dense image + BM25 on the image description)")
show_results(rag.rerank(image_result.query_text, image_result, top_n=20, use_images=False), "Improved + rerank: image query")
""")
code("""
# Combine an image with a text constraint: both are embedded and fused, text also drives BM25
combined = rag.search(query_text="vegetarian soup", query_image=query_image, k=20)
show_results(combined, "Improved: image + 'vegetarian soup'")
""")

md("""
## Grounded answer generation with Qwen3.5-4B

Instead of a generic summary, the model answers the user's request using only the retrieved recipes and cites
them by number. For image queries it also sees the query image. Thinking mode is disabled for low latency.
""")
code("""
%%time
query = "quick vegetarian dinner with spinach"
result = rag.rerank(query, rag.search(query_text=query, k=20), top_n=20, use_images=False)
show_results(result, f"Retrieved for: {query}", n=3)
start = time.time()
answer = rag.answer(query, result)
print(f"[INFO] Generation time: {time.time() - start:.1f}s\\n")
print(answer)
""")
code("""
%%time
answer = rag.answer(None, image_result, query_image=query_image)
print(answer)
""")

md("""
## Evaluation

`scripts/eval_retrieval.py` samples recipes and measures how often each pipeline retrieves the target recipe for
(a) its **title** and (b) a **paraphrased** query written by Qwen3.5-4B without the title words. Run it as a
separate process on a fresh GPU (it loads its own copy of the models):

```bash
python scripts/eval_retrieval.py --num-queries 200 --num-rerank-queries 100
```

Results from our L4 run are in [`results/eval_results.md`](../results/eval_results.md).
""")
code("""
results_file = REPO_DIR / "results" / "eval_results.md"
if results_file.exists():
    from IPython.display import Markdown
    display(Markdown(results_file.read_text()))
else:
    print("[INFO] No results yet, run scripts/eval_retrieval.py")
""")

md("## Gradio demo")
code("""
import gradio as gr


def demo_fn(query_text, query_image, rerank, generate):
    result = rag.search(query_text=query_text, query_image=query_image, k=20)
    if rerank and result.query_text:
        result = rag.rerank(result.query_text, result, top_n=20, use_images=False)
    samples = dataset.select(result.indices[:3])
    gallery = [(s["image"], f"#{i} {s['name']}") for i, s in enumerate(samples, 1)]
    answer = rag.answer(query_text, result, query_image=query_image) if generate else ""
    return gallery, answer, {**result.timings, "query_used_for_bm25_rerank": result.query_text}


with gr.Blocks(title="Multimodal RAG (improved)") as demo:
    gr.Markdown("# Multimodal recipe RAG: hybrid retrieval, Nemotron VL rerank, Qwen3.5-4B answers")
    with gr.Row():
        with gr.Column(scale=1):
            text_in = gr.Textbox(label="Text query", placeholder="e.g. quick vegetarian dinner with spinach")
            image_in = gr.Image(label="Image query (optional, can be combined with text)", type="pil", height=200)
            rerank_in = gr.Checkbox(label="Rerank (text-only cross-encoder)", value=True)
            generate_in = gr.Checkbox(label="Generate grounded answer", value=False)
            button = gr.Button("Search", variant="primary")
        with gr.Column(scale=2):
            gallery_out = gr.Gallery(label="Top recipes", columns=3, height="auto")
            answer_out = gr.Markdown()
            timings_out = gr.JSON(label="Timings")
    button.click(demo_fn, [text_in, image_in, rerank_in, generate_in], [gallery_out, answer_out, timings_out])

# Note: set HEADLESS=1 to build the demo without launching it (e.g. when running the notebook from a script)
if not os.environ.get("HEADLESS"):
    demo.launch(share=False, inline=True, height=900)
else:
    print("[INFO] HEADLESS is set, skipping demo.launch()")
""")

for n, cell in enumerate(cells):
    cell["id"] = f"cell-{n:02d}"
    cell["source"] = cell["source"].splitlines(keepends=True)

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3", "language": "python"},
                   "language_info": {"name": "python"}, "accelerator": "GPU",
                   "colab": {"gpuType": "L4", "provenance": []}}}
DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"Wrote {DST} ({len(cells)} cells)")
