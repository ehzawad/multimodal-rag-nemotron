"""Evaluate the final (release) pipeline configurations on the shared query set.

Qwen3-VL-Embedding-2B + language-gated BM25 (RRF) + Qwen3-VL-Reranker-2B, optionally with the Qwen3.5-4B search agent.
Usage (GPU): python scripts/eval_final.py --embeddings qwen3_vl_embedding_2b_image_text_10k.safetensors
"""
import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mmrag import agent as A, evaluation as E, pipeline as P, qwen_vl_retrieval as Q  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--embeddings", default=Q.EMBEDDING_FILE)
parser.add_argument("--query-sets", default=",".join(E.QUERY_SETS))
parser.add_argument("--num-queries", type=int, default=200)
parser.add_argument("--num-rerank-queries", type=int, default=50)
parser.add_argument("--num-slowest-queries", type=int, default=25)
parser.add_argument("--gate", type=float, default=0.8, help="min. query vocabulary coverage to use BM25")
parser.add_argument("--output-name", default="eval_results_final")
parser.add_argument("--output-dir", default=os.environ.get("COLAB_OUTPUT_DIR", str(E.REPO_DIR / "results")))
args = parser.parse_args()

from datasets import load_dataset  # noqa: E402

queries = E.load_queries(limit=args.num_queries)
dataset = load_dataset(P.DATASET_PATH, split="train")
rag = Q.QwenVLRAG(dataset, Q.load_embed_model(), Q.load_document_embeddings(args.embeddings), Q.load_rerank_model(),
                  None, *P.load_generation_model())
agent = A.AgenticRAG(rag, search_kwargs={"bm25_min_coverage": args.gate})
print(f"[INFO] GPU memory after loading: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

G = args.gate
FAST, SLOW, SLOWEST = args.num_queries, args.num_rerank_queries, args.num_slowest_queries
METHODS = {
    "final: qwen3-vl gated hybrid": (lambda q: rag.search(query_text=q, k=100, bm25_min_coverage=G), FAST),
    "final: qwen3-vl gated hybrid + qwen3-vl rerank (text)":
        (lambda q: rag.rerank(q, rag.search(query_text=q, k=100, bm25_min_coverage=G), top_n=20, use_images=False), SLOW),
    "final: qwen3-vl gated hybrid + qwen3-vl rerank (image+text)":
        (lambda q: rag.rerank(q, rag.search(query_text=q, k=100, bm25_min_coverage=G), top_n=20, use_images=True), SLOWEST),
    "final: agent + qwen3-vl gated hybrid + qwen3-vl rerank (text)":
        (lambda q: agent.search(query_text=q, k=100, rerank_top_n=20), SLOW),
}
rag.search(query_text="warm up query")
E.run(METHODS, queries, args.query_sets.split(","), Path(args.output_dir), args.output_name)
print((Path(args.output_dir) / f"{args.output_name}.md").read_text())
