"""Retrieval evaluation on the shared query set (results/eval_queries.json, see scripts/make_eval_queries.py).

Known-item evaluation: each query should retrieve one target recipe. Reports Recall@1/5/10, MRR@10 and mean
latency per method and query set (title, paraphrase, es, bn, zh).

Usage (GPU):
    python scripts/eval_retrieval.py                          # all methods, all query sets
    python scripts/eval_retrieval.py --num-rerank-queries 50  # rerank methods are slow, evaluate them on a prefix
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
from mmrag import pipeline as P  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--queries", default=str(REPO_DIR / "results" / "eval_queries.json"))
parser.add_argument("--query-sets", default="title,paraphrase,es,bn,zh")
parser.add_argument("--methods", default=None, help="comma separated subset of method names (default: all)")
parser.add_argument("--num-queries", type=int, default=200)
parser.add_argument("--num-rerank-queries", type=int, default=50)
parser.add_argument("--num-slowest-queries", type=int, default=25, help="for image+text reranking (~17s per query)")
parser.add_argument("--rerank-top-n", type=int, default=20)
parser.add_argument("--output-name", default="eval_results")
parser.add_argument("--output-dir", default=os.environ.get("COLAB_OUTPUT_DIR", str(REPO_DIR / "results")))
args = parser.parse_args()

queries = json.loads(Path(args.queries).read_text())[:args.num_queries]
query_sets = args.query_sets.split(",")
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

from datasets import load_dataset  # noqa: E402

print("[INFO] Loading dataset, embeddings and models...")
dataset = load_dataset(P.DATASET_PATH, split="train")
document_embeddings = P.load_document_embeddings(device="cuda")
embed_model = P.load_embed_model()
rerank_model, rerank_processor = P.load_rerank_model(modality="image_text")
rag = P.MultimodalRAG(dataset, embed_model, document_embeddings, rerank_model, rerank_processor)
print(f"[INFO] GPU memory after loading: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

# Sanity check: the precomputed document embeddings were made with an older revision of the embedding model's remote
# code (same weights). Re-embed a few documents with the revision we load and compare.
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
    "improved: hybrid + rerank (text)":
        (lambda q: rag.rerank(q, rag.search(query_text=q, k=100), top_n=TOP_N, use_images=False), SLOW),
}
if args.methods:
    METHODS = {name: METHODS[name] for name in args.methods.split(",")}


def evaluate(method, query_key, limit):
    ranks, latencies = [], []
    for q in queries[:limit]:
        torch.cuda.synchronize()
        start = time.time()
        result = method(q[query_key])
        torch.cuda.synchronize()
        latencies.append(time.time() - start)
        ranks.append(result.indices.index(q["doc_id"]) + 1 if q["doc_id"] in result.indices else None)
    n = len(ranks)
    return {"n": n,
            "recall@1": sum(r is not None and r <= 1 for r in ranks) / n,
            "recall@5": sum(r is not None and r <= 5 for r in ranks) / n,
            "recall@10": sum(r is not None and r <= 10 for r in ranks) / n,
            "mrr@10": sum(1 / r for r in ranks if r is not None and r <= 10) / n,
            "mean_latency_s": sum(latencies) / n}


rag.search(query_text="warm up query")  # first CUDA calls are slow

results = {}
for query_key in query_sets:
    for name, (method, limit) in METHODS.items():
        metrics = evaluate(method, query_key, limit)
        results[f"{query_key} | {name}"] = metrics
        print(f"[{query_key:10s}] {name:42s} n={metrics['n']:4d} R@1={metrics['recall@1']:.3f} "
              f"R@5={metrics['recall@5']:.3f} R@10={metrics['recall@10']:.3f} MRR@10={metrics['mrr@10']:.3f} "
              f"latency={metrics['mean_latency_s'] * 1000:.0f}ms", flush=True)

(output_dir / f"{args.output_name}.json").write_text(json.dumps(results, indent=1))
lines = ["| Query set | Method | n | R@1 | R@5 | R@10 | MRR@10 | Latency |", "|---|---|---|---|---|---|---|---|"]
for key, m in results.items():
    query_key, name = key.split(" | ")
    lines.append(f"| {query_key} | {name} | {m['n']} | {m['recall@1']:.3f} | {m['recall@5']:.3f} | {m['recall@10']:.3f} "
                 f"| {m['mrr@10']:.3f} | {m['mean_latency_s'] * 1000:.0f} ms |")
(output_dir / f"{args.output_name}.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
