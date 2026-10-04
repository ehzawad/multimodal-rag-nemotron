"""Build notebooks/multimodal_rag.ipynb: walkthrough of the final (best) multimodal RAG pipeline."""
import json
from pathlib import Path

DST = Path(__file__).resolve().parent.parent / "notebooks" / "multimodal_rag.ipynb"
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


md("""
# Multilingual multimodal recipe RAG

Search 10,096 recipes ([`mrdbourke/recipe-synthetic-images-10k`](https://huggingface.co/datasets/mrdbourke/recipe-synthetic-images-10k),
each a recipe-card image + markdown text) with a text query in any of 30+ languages, an image, or both, and get a grounded
answer. This is the best pipeline from our evaluation ([`reports/report.pdf`](../reports/report.pdf)):

1. **Dense retrieval** with [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) over image + text
   document embeddings (multimodal, multilingual).
2. **BM25** fused with Reciprocal Rank Fusion, but only when the query's words occur in the (English) corpus vocabulary:
   lexical search helps English queries and only adds noise to others.
3. **Rerank** the top 20 with [Qwen3-VL-Reranker-2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) (recipe text by
   default, image + text for maximum accuracy).
4. **Answer** with [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), grounded in the top recipes with citations.
   Image queries are first described in text by Qwen3.5-4B so they can use BM25 and the reranker.

Runs on one 24GB GPU (Colab L4 / RTX A5000). The original tutorial pipeline (fixed) is kept in
[`multimodal_rag_tutorial_baseline.ipynb`](multimodal_rag_tutorial_baseline.ipynb) for reference.
""")
md("## Install dependencies")
code("""
!pip install -q -U "transformers>=5.2" "huggingface_hub>=1.0" "sentence-transformers>=5.4" "accelerate>=1.12" "datasets>=4.4" bm25s flash-linear-attention "gradio>=5"
""")
code("""
import os
import sys
from pathlib import Path

# Make the `mmrag` package importable (from the repo's notebooks/ folder or a fresh Colab runtime)
REPO_URL = "https://github.com/ehzawad/multimodal-rag-nemotron"
for candidate in [Path.cwd(), Path.cwd().parent, Path("multimodal-rag-nemotron"), Path("/content/mmrag")]:
    if (candidate / "mmrag" / "qwen_vl_retrieval.py").exists():
        REPO_DIR = candidate.resolve()
        break
else:
    !git clone -q {REPO_URL}
    REPO_DIR = Path("multimodal-rag-nemotron").resolve()
sys.path.insert(0, str(REPO_DIR))
print(f"[INFO] Using pipeline code from: {REPO_DIR}")
""")
md("""
## Load dataset, models and corpus embeddings

The corpus is embedded once (image + recipe text per document, ~50 min on an L4) and cached to a safetensors file.
Set `QWEN_EMBEDDINGS` to reuse an existing file.
""")
code("""
%%time
import time
import torch
from datasets import load_dataset
from PIL import Image

from mmrag import agent as A, pipeline as P, qwen_vl_retrieval as Q

dataset = load_dataset(P.DATASET_PATH, split="train")
embed_model = Q.load_embed_model()
EMBEDDINGS_FILE = os.environ.get("QWEN_EMBEDDINGS", Q.EMBEDDING_FILE)
if not Path(EMBEDDINGS_FILE).exists():
    print("[INFO] Embedding the corpus with Qwen3-VL-Embedding-2B (one-off, ~50 min on an L4)...")
    Q.build_document_embeddings(dataset, embed_model, path=EMBEDDINGS_FILE)

rag = Q.QwenVLRAG(dataset, embed_model, Q.load_document_embeddings(EMBEDDINGS_FILE), Q.load_rerank_model(), None,
                  *P.load_generation_model())
SEARCH = {"k": 20, "bm25_min_coverage": 0.8}  # BM25 only for queries whose words are in the corpus vocabulary
print(f"[INFO] {len(dataset)} recipes | GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
""")
md("## Helper to display results")
code("""
import base64
import html
import io

from IPython.display import HTML, display


def _thumbnail_b64(image, width=200):
    image = image.copy()
    image.thumbnail((width, width * 2))
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=80)
    return base64.b64encode(buffer.getvalue()).decode()


def show(result, title, n=3):
    samples = dataset.select(result.indices[:n])
    cards = "".join(
        f"<div style='flex:1;min-width:0;font-family:system-ui;font-size:12px'>"
        f"<img src='data:image/jpeg;base64,{_thumbnail_b64(s['image'])}' style='width:100%'>"
        f"<b>#{rank}</b> {html.escape(s['name'])}</div>"
        for rank, s in enumerate(samples, 1))
    display(HTML(f"<h4 style='font-family:system-ui;margin:12px 0 4px'>{html.escape(title)}</h4>"
                 f"<div style='display:flex;gap:10px;max-width:700px'>{cards}</div>"
                 f"<div style='color:#888;font-size:12px'>{html.escape(str(result.timings))}</div>"))


def search(query_text=None, query_image=None, rerank_images=False):
    result = rag.search(query_text=query_text, query_image=query_image, **SEARCH)
    return rag.rerank(query_text or result.query_text, result, top_n=20, use_images=rerank_images)
""")
md("## Text queries in any language")
code("""
for query in ["banana bread with sour cream", "pan de plátano con crema agria", "টক দই দিয়ে কলার রুটি", "酸奶油香蕉面包",
              "quick vegetarian dinner with spinach", "egg sandwich"]:
    show(search(query), query)
""")
md("""
## Image queries and combined image + text queries

Qwen3.5-4B describes the image ("tomato pasta with fresh basil…"); that text drives BM25 and the reranker while the
image itself is embedded for dense retrieval. Text and image can be combined to constrain the search.
""")
code("""
from huggingface_hub import hf_hub_download

query_image = Image.open(hf_hub_download(P.SPACE_REPO, "kitchen_bench.png", repo_type="space")).convert("RGB")
display(query_image.resize((360, int(360 * query_image.height / query_image.width))))
image_result = search(query_image=query_image)
print("[INFO] Image described as:", image_result.query_text)
show(image_result, "Image query")
show(search("vegetarian soup", query_image), "Image + 'vegetarian soup'")
""")
md("## Grounded answers")
code("""
%%time
query = "quick vegetarian dinner with spinach"
result = search(query)
show(result, query)
print(rag.answer(query, result))
""")
code("""
%%time
print(rag.answer(None, image_result, query_image=query_image))
""")
md("""
## Optional: let Qwen3.5-4B plan the searches (tool calling)

The agent rewrites/translates the request into 1-3 English searches. In our evaluation it did **not** beat the
multilingual embedding pipeline and adds ~5 s per query, so it is off by default; it can still help for vague,
multi-part requests.
""")
code("""
agent = A.AgenticRAG(rag, search_kwargs={"bm25_min_coverage": 0.8})
result = agent.search(query_text="something warm and filling for a rainy evening, no meat", k=20, rerank_top_n=20)
print("[INFO] Agent searches:", result.timings["searches"])
show(result, "Agent: something warm and filling for a rainy evening, no meat")
""")
md("## Evaluation summary")
code("""
from IPython.display import Markdown

summary = REPO_DIR / "results" / "summary.md"
display(Markdown(summary.read_text()) if summary.exists() else Markdown("Run the scripts in `scripts/eval_*.py`."))
""")
md("## Gradio demo")
code("""
import gradio as gr


def demo_fn(query_text, query_image, rerank_images, answer):
    if not (query_text and query_text.strip()) and query_image is None:
        raise gr.Error("Please provide a text query, an image query or both.")
    result = search(query_text, query_image, rerank_images=rerank_images)
    samples = dataset.select(result.indices[:3])
    gallery = [(s["image"], f"#{i} {s['name']}") for i, s in enumerate(samples, 1)]
    return gallery, rag.answer(query_text, result, query_image=query_image) if answer else "", result.timings


with gr.Blocks(title="Multimodal recipe RAG") as demo:
    gr.Markdown("# Multilingual multimodal recipe RAG")
    with gr.Row():
        with gr.Column(scale=1):
            text_in = gr.Textbox(label="Text query (any language)", lines=2)
            image_in = gr.Image(label="Image query (optional)", type="pil", height=200)
            rerank_images_in = gr.Checkbox(label="Rerank with recipe images too (slower, most accurate)", value=False)
            answer_in = gr.Checkbox(label="Generate grounded answer", value=True)
            button = gr.Button("Search", variant="primary")
        with gr.Column(scale=2):
            gallery_out = gr.Gallery(label="Top recipes", columns=3, height="auto", object_fit="contain")
            answer_out = gr.Markdown()
            timings_out = gr.JSON(label="Timings")
    button.click(demo_fn, [text_in, image_in, rerank_images_in, answer_in], [gallery_out, answer_out, timings_out])

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
                   "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"gpuType": "L4"}}}
DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"Wrote {DST} ({len(cells)} cells)")
