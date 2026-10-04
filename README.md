# Multimodal RAG with NVIDIA Nemotron VL

Recipe search over [`mrdbourke/recipe-synthetic-images-10k`](https://huggingface.co/datasets/mrdbourke/recipe-synthetic-images-10k)
(10,096 recipe images + markdown texts) with text or image queries, based on the
[Learn Hugging Face multimodal RAG tutorial](https://www.learnhuggingface.com/notebooks/hugging_face_multimodal_rag_tutorial).

* **Embedding:** [`nvidia/llama-nemotron-embed-vl-1b-v2`](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2) (image + text document embeddings)
* **Rerank:** [`nvidia/llama-nemotron-rerank-vl-1b-v2`](https://huggingface.co/nvidia/llama-nemotron-rerank-vl-1b-v2)
* **Generation:** [`Qwen/Qwen3-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)

## Branches

| Branch | Contents |
|---|---|
| `main` | The original tutorial pipeline, fixed so it runs top to bottom on a Google Colab L4 (24GB). |
| `improved` | Hybrid retrieval, image-query rewriting, batched reranking, Qwen3.5-4B grounded answers and a retrieval evaluation. |
| `exp-multilingual-embed` | Experiment: Qwen3-VL-Embedding-2B + Qwen3-VL-Reranker-2B (multimodal, multilingual) instead of Nemotron VL. |
| `exp-agentic` | Experiment: Qwen3.5-4B tool-calling agent that plans/translates searches (optional thinking). |

See [`REPORT.md`](REPORT.md) for the head-to-head evaluation of all pipelines.

## Improved pipeline (`improved`)

| Stage | Baseline (`main`) | Improved |
|---|---|---|
| Retrieval | Dense only (Nemotron Embed VL) | **Hybrid**: dense + BM25, fused with Reciprocal Rank Fusion |
| Query types | Text *or* image (text silently wins if both) | Text, image, or **both combined** |
| Image queries | Dense only, can't be reranked | **Rewritten to text** by the VLM so they can use BM25 + reranking |
| Reranking | Image + text pairs, batched | Batched, returns scores, fast **text-only** mode (no reviews) |
| Generation | Qwen3-VL-2B, generic summary | **Qwen3.5-4B**, answers the query grounded in the results with citations |
| Evaluation | – | Recall@k / MRR on 200 recipes × {title, paraphrase, Spanish, Bengali, Chinese} queries |

Code: [`mmrag/pipeline.py`](mmrag/pipeline.py) · notebook: [`notebooks/multimodal_rag_improved.ipynb`](notebooks/multimodal_rag_improved.ipynb)
(executed L4 run in [`results/executed_improved.ipynb`](results/executed_improved.ipynb)) · standalone:
[`standalone/app.py`](standalone/app.py).

```bash
pip install -r requirements-improved.txt
python standalone/app.py                                                   # Gradio demo on http://0.0.0.0:7860
python standalone/app.py --query "quick vegetarian dinner with spinach" --answer
python standalone/app.py --image photo.jpg --query "vegetarian"            # combined image + text query
python scripts/eval_retrieval.py                                           # retrieval evaluation
```

All models together use ~16GB of GPU memory (fits a 24GB L4 / RTX A5000).

## Fixes to the original notebook (`main`)

| Problem | Fix |
|---|---|
| `ValueError: Sample larger than population` – image paths were read from a local folder that only existed on the author's machine (0 images found). | Sample from the dataset's `image` column. |
| Rerank of 20 image + text pairs in one forward pass → CUDA OOM on a 24GB L4. | Batched reranking (`RERANK_BATCH_SIZE = 4`) in the notebook, `rerank_samples()` and `demo/app.py`. |
| Rerank processor kwargs (`rerank_max_length`, tiles) were built under a misspelled name and never passed. | Pass them to `AutoProcessor.from_pretrained`. |
| Rerank with an image query passed a PIL image as the reranker's text `question`. | Skip reranking for image queries (as the demo UI says). |
| `get_top_n_scores_and_samples` ignored `top_n` (`top_n == len(scores)` no-op, `[:3]` hardcoded). | Use `top_n`. |
| Embeddings had to be created from scratch (slow) unless the file was present locally; batch indices hardcoded to `i+8`. | Download the precomputed embeddings from the demo Space; respect `IMAGE_TEXT_BATCH_SIZE`. |
| Test image `moussaka.jpeg` not distributed. | Fall back to the demo's `kitchen_bench.png`. |
| "Run all" blocks on `demo.launch(debug=True)` and would try to publish a Space. | `HEADLESS=1` skips the launch; upload guarded by `UPLOAD_TO_HF = False`. |
| No dependency install cell. | Pinned install cell (`transformers==4.57.3`, which the pinned Nemotron commits were built against). |

The executed notebook from a Colab L4 run is in [`results/executed_baseline.ipynb`](results/executed_baseline.ipynb) (138 s end to end, no errors).

## Run

Open [`notebooks/multimodal_rag_tutorial.ipynb`](notebooks/multimodal_rag_tutorial.ipynb) in Colab with an L4 GPU, or run it headless on a
Colab GPU from your terminal with [colab-persist](https://github.com/ehzawad/colab-persist):

```bash
colab-persist run scripts/run_notebook.py --source . --project mmrag -- notebooks/multimodal_rag_tutorial.ipynb
```

### Standalone (your own GPU server, e.g. RTX A5000 24GB)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r standalone/requirements.txt
python standalone/baseline_app.py                                     # Gradio demo on http://0.0.0.0:7860
python standalone/baseline_app.py --query "eggplant dip" --rerank      # single query in the terminal
python standalone/baseline_app.py --query "best omelette recipes" --summary
```

The embeddings (41MB) and example image are downloaded from the original demo Space on first run
(set `MMRAG_DATA_DIR` to choose where). Peak GPU memory is ~16GB with reranking.

`scripts/build_baseline_notebook.py` regenerates the fixed notebook from the original tutorial notebook.
