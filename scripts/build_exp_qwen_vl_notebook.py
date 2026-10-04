"""Build notebooks/exp_qwen_vl_embedding.ipynb (Qwen3-VL-Embedding/Reranker experiment walkthrough)."""
import json
from pathlib import Path

DST = Path(__file__).resolve().parent.parent / "notebooks" / "exp_qwen_vl_embedding.ipynb"
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


md("""
# Experiment: multilingual multimodal retrieval with Qwen3-VL-Embedding-2B + Qwen3-VL-Reranker-2B

The baseline's Nemotron Embed VL retrieves well for English queries but collapses for other languages
(Recall@1 0.25 for Spanish vs 0.65 for the same queries in English, see [REPORT.md](../REPORT.md)).
[Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) and
[Qwen3-VL-Reranker-2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) (Jan 2026) are multimodal (text, images,
text + image) and support 30+ languages. This notebook swaps them in for the Nemotron models, keeping the rest of the
improved pipeline (hybrid BM25 + dense retrieval, image-query rewriting, Qwen3.5-4B grounded answers).

The recipe corpus (image + text per recipe) has to be embedded once with Qwen3-VL-Embedding
(`scripts/build_qwen_vl_embeddings.py`, ~50 min on an L4).
""")
md("## Install dependencies")
code("""
!pip install -q -U "transformers>=5.2" "huggingface_hub>=1.0" "sentence-transformers>=5.4" "accelerate>=1.12" "datasets>=4.4" bm25s flash-linear-attention
""")
code("""
import os
import sys
from pathlib import Path

REPO_URL = "https://github.com/ehzawad/multimodal-rag-nemotron"
for candidate in [Path.cwd(), Path.cwd().parent, Path("multimodal-rag-nemotron"), Path("/content/mmrag")]:
    if (candidate / "mmrag" / "qwen_vl_retrieval.py").exists():
        REPO_DIR = candidate.resolve()
        break
else:
    !git clone -q -b exp-multilingual-embed {REPO_URL}
    REPO_DIR = Path("multimodal-rag-nemotron").resolve()
sys.path.insert(0, str(REPO_DIR))
print(f"[INFO] Using pipeline code from: {REPO_DIR}")
""")
md("## Load models and the Qwen3-VL corpus embeddings")
code("""
%%time
import torch
from datasets import load_dataset
from PIL import Image

from mmrag import pipeline as P, qwen_vl_retrieval as Q

dataset = load_dataset(P.DATASET_PATH, split="train")
embed_model = Q.load_embed_model()

EMBEDDINGS_FILE = os.environ.get("QWEN_EMBEDDINGS", Q.EMBEDDING_FILE)
if not Path(EMBEDDINGS_FILE).exists():
    print("[INFO] Embedding the corpus with Qwen3-VL-Embedding-2B (one-off, ~50 min on an L4)...")
    Q.build_document_embeddings(dataset, embed_model, path=EMBEDDINGS_FILE)
document_embeddings = Q.load_document_embeddings(EMBEDDINGS_FILE)

rerank_model = Q.load_rerank_model()
generation_model, generation_processor = P.load_generation_model()
rag = Q.QwenVLRAG(dataset, embed_model, document_embeddings, rerank_model, None, generation_model, generation_processor)
print(f"[INFO] GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
""")
md("## The same request in four languages")
code("""
def top(result, n=3):
    return [dataset[i]["name"] for i in result.indices[:n]]


requests = {
    "en": "banana bread with sour cream",
    "es": "pan de plátano con crema agria",
    "bn": "টক দই দিয়ে কলার রুটি",
    "zh": "酸奶油香蕉面包",
}
for language, query in requests.items():
    dense = P.RetrievalResult(*rag.dense_search(query, k=20))
    reranked = rag.rerank(query, rag.search(query_text=query, k=20), top_n=20, use_images=False)
    print(f"[{language}] {query}\\n    dense:           {top(dense)}\\n    hybrid + rerank: {top(reranked)}  {reranked.timings}")
""")
md("## Image query and grounded answer")
code("""
from huggingface_hub import hf_hub_download

query_image = Image.open(hf_hub_download(P.SPACE_REPO, "kitchen_bench.png", repo_type="space")).convert("RGB")
image_result = rag.search(query_image=query_image, k=20)
print("[INFO] Image rewritten to:", image_result.query_text)
image_result = rag.rerank(image_result.query_text, image_result, top_n=20, use_images=False)
print("[INFO] Top recipes:", top(image_result))
print(rag.answer(None, image_result, query_image=query_image))
""")
md("## Evaluation")
code("""
results_file = REPO_DIR / "results" / "eval_results_qwen_vl.md"
if results_file.exists():
    from IPython.display import Markdown, display
    display(Markdown(results_file.read_text()))
else:
    print("[INFO] Run: python scripts/eval_qwen_vl.py")
""")

for n, cell in enumerate(cells):
    cell["id"] = f"cell-{n:02d}"
    cell["source"] = cell["source"].splitlines(keepends=True)
nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3", "language": "python"},
                   "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"gpuType": "L4"}}}
DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"Wrote {DST} ({len(cells)} cells)")
