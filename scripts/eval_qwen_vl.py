"""Retrieval evaluation of the Qwen3-VL-Embedding-2B / Qwen3-VL-Reranker-2B experiment on the shared query set.

Needs the re-embedded corpus (scripts/build_qwen_vl_embeddings.py). Baseline/improved numbers come from
scripts/eval_retrieval.py on the same queries.
Usage (GPU): python scripts/eval_qwen_vl.py --embeddings qwen3_vl_embedding_2b_image_text_10k.safetensors
"""
import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mmrag import evaluation as E, pipeline as P, qwen_vl_retrieval as Q  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--embeddings", default=Q.EMBEDDING_FILE)
parser.add_argument("--query-sets", default=",".join(E.QUERY_SETS))
parser.add_argument("--num-queries", type=int, default=200)
parser.add_argument("--num-rerank-queries", type=int, default=50)
parser.add_argument("--num-slowest-queries", type=int, default=25)
parser.add_argument("--output-name", default="eval_results_qwen_vl")
parser.add_argument("--output-dir", default=os.environ.get("COLAB_OUTPUT_DIR", str(E.REPO_DIR / "results")))
args = parser.parse_args()

from datasets import load_dataset  # noqa: E402

queries = E.load_queries(limit=args.num_queries)
dataset = load_dataset(P.DATASET_PATH, split="train")
rag = Q.QwenVLRAG(dataset, Q.load_embed_model(), Q.load_document_embeddings(args.embeddings), Q.load_rerank_model())
print(f"[INFO] GPU memory after loading: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

FAST, SLOW, SLOWEST = args.num_queries, args.num_rerank_queries, args.num_slowest_queries
METHODS = {
    "qwen3-vl: dense": (lambda q: P.RetrievalResult(*rag.dense_search(q, k=100)), FAST),
    "qwen3-vl: hybrid (dense + bm25, RRF)": (lambda q: rag.search(query_text=q, k=100), FAST),
    "qwen3-vl: hybrid + qwen3-vl rerank (text)":
        (lambda q: rag.rerank(q, rag.search(query_text=q, k=100), top_n=20, use_images=False), SLOW),
    "qwen3-vl: hybrid + qwen3-vl rerank (image+text)":
        (lambda q: rag.rerank(q, rag.search(query_text=q, k=100), top_n=20, use_images=True), SLOWEST),
}
rag.search(query_text="warm up query")
E.run(METHODS, queries, args.query_sets.split(","), Path(args.output_dir), args.output_name)
print((Path(args.output_dir) / f"{args.output_name}.md").read_text())
