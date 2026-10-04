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

`scripts/build_baseline_notebook.py` regenerates the fixed notebook from the original tutorial notebook.
