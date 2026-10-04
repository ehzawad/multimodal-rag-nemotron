"""Retrieval evaluation on the shared query set (results/eval_queries.json, see scripts/make_eval_queries.py).

Known-item evaluation: each query should retrieve one target recipe. Reports Recall@1/5/10, MRR@10 and mean
latency per method and query set (title, paraphrase, es, bn, zh).

Usage (GPU):
    python scripts/eval_retrieval.py                          # all methods, all query sets
    python scripts/eval_retrieval.py --num-rerank-queries 50  # rerank methods are slow, evaluate them on a prefix
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
from mmrag import evaluation as E, pipeline as P  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--queries", default=str(E.QUERIES_FILE))
parser.add_argument("--query-sets", default="title,paraphrase,es,bn,zh")
parser.add_argument("--methods", default=None, help="comma separated subset of method names (default: all)")
parser.add_argument("--num-queries", type=int, default=200)
parser.add_argument("--num-rerank-queries", type=int, default=50)
parser.add_argument("--num-slowest-queries", type=int, default=25, help="for image+text reranking (~17s per query)")
parser.add_argument("--rerank-top-n", type=int, default=20)
parser.add_argument("--embeddings", default=None, help="document embeddings file (default: see load_document_embeddings)")
parser.add_argument("--output-name", default="eval_results")
parser.add_argument("--output-dir", default=os.environ.get("COLAB_OUTPUT_DIR", str(REPO_DIR / "results")))
args = parser.parse_args()

queries = E.load_queries(args.queries, args.num_queries)
query_sets = args.query_sets.split(",")
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

from datasets import load_dataset  # noqa: E402

print("[INFO] Loading dataset, embeddings and models...")
dataset = load_dataset(P.DATASET_PATH, split="train")
document_embeddings = P.load_document_embeddings(args.embeddings, device="cuda")
embed_model = P.load_embed_model()
rerank_model, rerank_processor = P.load_rerank_model(modality="image_text")
rag = P.MultimodalRAG(dataset, embed_model, document_embeddings, rerank_model, rerank_processor)
print(f"[INFO] GPU memory after loading: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

# Sanity check: re-embed a few documents with the loaded model and compare with the stored document embeddings
# (should be ~1.0; the demo Space's file only matches MMRAG_NEMOTRON_REVISIONS=original).
with torch.inference_mode():
    check_ids = [0, 42, 4242]
    check = dataset.select(check_ids)
    fresh = embed_model.encode_documents(images=check["image"], texts=check["recipe_markdown"])
cosine = (torch.nn.functional.normalize(fresh.float(), dim=-1) * document_embeddings[check_ids]).sum(-1)
print(f"[INFO] Cosine(fresh document embeddings, precomputed): {[round(c, 4) for c in cosine.tolist()]}")

# --- Methods: name -> (function(query) -> RetrievalResult, number of queries to evaluate) -------------------
TOP_N = args.rerank_top_n
FAST, SLOW, SLOWEST = args.num_queries, args.num_rerank_queries, args.num_slowest_queries
METHODS = {
    "baseline: dense": (lambda q: P.RetrievalResult(*rag.dense_search(q, k=100)), FAST),
    "baseline: dense + rerank (image+text)":
        (lambda q: rag.rerank(q, P.RetrievalResult(*rag.dense_search(q, k=100)), top_n=TOP_N, use_images=True), SLOWEST),
    "bm25 only": (lambda q: P.RetrievalResult(*rag.bm25_search(q, k=100)), FAST),
    "improved: hybrid (dense + bm25, RRF)": (lambda q: rag.search(query_text=q, k=100), FAST),
    "improved: hybrid, bm25 weight 0.5": (lambda q: rag.search(query_text=q, k=100, bm25_weight=0.5), FAST),
    "improved: hybrid + rerank (text)":
        (lambda q: rag.rerank(q, rag.search(query_text=q, k=100), top_n=TOP_N, use_images=False), SLOW),
}
if args.methods:
    METHODS = {name: METHODS[name] for name in args.methods.split(",")}


rag.search(query_text="warm up query")  # first CUDA calls are slow
E.run(METHODS, queries, query_sets, output_dir, args.output_name)
print((output_dir / f"{args.output_name}.md").read_text())
