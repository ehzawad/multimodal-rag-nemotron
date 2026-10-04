"""Build notebooks/exp_agentic.ipynb (agentic tool-calling retrieval experiment walkthrough)."""
import json
from pathlib import Path

DST = Path(__file__).resolve().parent.parent / "notebooks" / "exp_agentic.ipynb"
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


md("""
# Experiment: agentic retrieval with Qwen3.5-4B tool calling

Instead of searching with the user's raw request, [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) gets a
`search_recipes` tool and plans the searches: it translates non-English requests, extracts the dish or key
ingredients from vague ones and adds alternative phrasings. Each tool call runs the improved hybrid search; the
rankings are fused (RRF) with the original query's ranking, reranked with the Nemotron reranker, and the agent answers
in the user's language from the tool results. `think=True` enables Qwen3.5's reasoning before the tool calls.

Code: [`mmrag/agent.py`](../mmrag/agent.py). Results: [REPORT.md](../REPORT.md).
""")
md("## Install dependencies")
code("""
!pip install -q -U "transformers>=5.2" "huggingface_hub>=1.0" "accelerate>=1.12" "datasets>=4.4" bm25s flash-linear-attention
""")
code("""
import os
import sys
from pathlib import Path

REPO_URL = "https://github.com/ehzawad/multimodal-rag-nemotron"
for candidate in [Path.cwd(), Path.cwd().parent, Path("multimodal-rag-nemotron"), Path("/content/mmrag")]:
    if (candidate / "mmrag" / "agent.py").exists():
        REPO_DIR = candidate.resolve()
        break
else:
    !git clone -q -b exp-agentic {REPO_URL}
    REPO_DIR = Path("multimodal-rag-nemotron").resolve()
sys.path.insert(0, str(REPO_DIR))
print(f"[INFO] Using pipeline code from: {REPO_DIR}")
""")
md("## Load models")
code("""
%%time
import torch
from datasets import load_dataset

from mmrag import agent as A, pipeline as P

dataset = load_dataset(P.DATASET_PATH, split="train")
rag = P.MultimodalRAG(dataset, P.load_embed_model(), P.load_document_embeddings(), *P.load_rerank_model(),
                      *P.load_generation_model())
agent = A.AgenticRAG(rag)
thinking_agent = A.AgenticRAG(rag, think=True)
print(f"[INFO] GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
""")
md("## What the agent searches for")
code("""
requests = [
    "banana bread with sour cream",
    "pan de plátano con crema agria",
    "টক দই দিয়ে কলার রুটি",
    "酸奶油香蕉面包",
    "something warm and filling for a rainy evening, no meat",
]
for request in requests:
    result = agent.search(query_text=request, k=20, rerank_top_n=20)
    print(f"{request}\\n    searches: {result.timings['searches']}\\n    top 3:    {[dataset[i]['name'] for i in result.indices[:3]]}"
          f"\\n    timings:  plan {result.timings['plan_s']}s, rerank {result.timings.get('rerank_s')}s")
""")
md("## Answer in the user's language")
code("""
%%time
request = "টক দই দিয়ে কলার রুটি"
result = agent.search(query_text=request, k=20, rerank_top_n=20)
print(agent.answer(request, result))
""")
md("## With reasoning (thinking mode)")
code("""
%%time
request = "something warm and filling for a rainy evening, no meat"
searches, raw = thinking_agent.plan_searches(request)
print(raw[:1500])
print("searches:", searches)
""")
md("## Evaluation")
code("""
results_file = REPO_DIR / "results" / "eval_results_agentic.md"
if results_file.exists():
    from IPython.display import Markdown, display
    display(Markdown(results_file.read_text()))
else:
    print("[INFO] Run: python scripts/eval_agentic.py")
""")

for n, cell in enumerate(cells):
    cell["id"] = f"cell-{n:02d}"
    cell["source"] = cell["source"].splitlines(keepends=True)
nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3", "language": "python"},
                   "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"gpuType": "L4"}}}
DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"Wrote {DST} ({len(cells)} cells)")
