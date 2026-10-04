# Multilingual multimodal recipe RAG

Search 10,096 recipes ([`mrdbourke/recipe-synthetic-images-10k`](https://huggingface.co/datasets/mrdbourke/recipe-synthetic-images-10k):
recipe-card image + markdown text each) with a **text query in 30+ languages, an image, or both**, and get an answer grounded in the
retrieved recipes. Runs on one 24GB GPU (Colab L4, RTX A5000). Full evaluation: **[`reports/report.pdf`](reports/report.pdf)**.

```
query (text / image) ─► Qwen3-VL-Embedding-2B dense search ─┐
                    └─► BM25 (only if query words are in the corpus vocabulary) ─┴─► RRF ─► Qwen3-VL-Reranker-2B ─► Qwen3.5-4B answer
```

| Pipeline | Avg Recall@1 (title, paraphrase, es, bn, zh) | Non-English Recall@1 | Latency / query |
|---|---|---|---|
| Original tutorial (Nemotron VL, image + text rerank) | 0.400 | 0.13 | ~15 s |
| **This repo (default)** | **0.748** | **0.63** | **~1 s** |

Started from the [Learn Hugging Face multimodal RAG tutorial](https://www.learnhuggingface.com/notebooks/hugging_face_multimodal_rag_tutorial)
(Nemotron Embed/Rerank VL + Qwen3-VL-2B); the fixed original is kept as a reference baseline.

## Quick start

```bash
pip install -r requirements.txt
python standalone/app.py                                             # Gradio demo on http://0.0.0.0:7860
python standalone/app.py --query "酸奶油香蕉面包" --answer              # one query, any language
python standalone/app.py --image photo.jpg --query "vegetarian"      # image + text
```

The corpus is embedded once on first run (~50 min on an L4, 41MB file). Options: `--rerank-images` (most accurate for English,
~3× slower), `--agent` (Qwen3.5-4B plans the searches with tool calls), `--backend nemotron`.
Notebook: [`notebooks/multimodal_rag.ipynb`](notebooks/multimodal_rag.ipynb) (Colab, L4).

## Repository

| Path | What |
|---|---|
| `mmrag/` | pipeline (`pipeline.py` hybrid retrieval, rerank, generation; `qwen_vl_retrieval.py`; `agent.py`; `evaluation.py`) |
| `standalone/app.py`, `notebooks/multimodal_rag.ipynb` | the final pipeline as CLI/Gradio app and notebook |
| `notebooks/multimodal_rag_tutorial_baseline.ipynb`, `standalone/baseline_app.py` | original tutorial, fixed (`requirements-baseline.txt`) |
| `scripts/` | corpus embedding, evaluation (`eval_*.py`, `make_eval_queries.py`), report, headless notebook runner |
| `results/` | evaluation queries and results (JSON/markdown), executed notebooks |
| `reports/` | `report.pdf` / `report.html` and figures |

## Key findings

* The original notebook crashed (`ValueError: Sample larger than population`, local image folder) and its image + text rerank
  ran out of memory on 24GB; both fixed, plus three latent bugs.
* The tutorial's precomputed embeddings only match its pinned Nemotron commit; official revisions need a re-embedded corpus
  (which also improved dense retrieval).
* Multilingual embedding (Qwen3-VL) is the biggest lever; BM25 helps English but hurts other languages unless gated;
  agentic query planning helps weak retrievers but costs ~5 s and doesn't beat the default.
