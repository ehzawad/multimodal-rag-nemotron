"""Retrieval evaluation of the agentic (tool-calling Qwen3.5-4B) experiment on the shared query set.
Usage (GPU): python scripts/eval_agentic.py
"""
import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mmrag import agent as A, evaluation as E, pipeline as P  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--query-sets", default=",".join(E.QUERY_SETS))
parser.add_argument("--num-queries", type=int, default=200)
parser.add_argument("--num-agent-queries", type=int, default=50)
parser.add_argument("--num-thinking-queries", type=int, default=25)
parser.add_argument("--output-name", default="eval_results_agentic")
parser.add_argument("--output-dir", default=os.environ.get("COLAB_OUTPUT_DIR", str(E.REPO_DIR / "results")))
args = parser.parse_args()

from datasets import load_dataset  # noqa: E402

queries = E.load_queries(limit=args.num_queries)
dataset = load_dataset(P.DATASET_PATH, split="train")
rag = P.MultimodalRAG(dataset, P.load_embed_model(), P.load_document_embeddings(), *P.load_rerank_model(),
                      *P.load_generation_model())
print(f"[INFO] GPU memory after loading: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
agent = A.AgenticRAG(rag, think=False)
thinking_agent = A.AgenticRAG(rag, think=True)

for q in queries[:3]:  # show what the agent does
    for key in ["paraphrase", "bn"]:
        searches, raw = agent.plan_searches(q[key])
        print(f"[PLAN] {q[key]!r} -> {searches}")

N, NT = args.num_agent_queries, args.num_thinking_queries
METHODS = {
    "agentic: tool-call rewrite + hybrid": (lambda q: agent.search(query_text=q, k=100), N),
    "agentic: tool-call rewrite + hybrid + rerank (text)":
        (lambda q: agent.search(query_text=q, k=100, rerank_top_n=20), N),
    "agentic (thinking): rewrite + hybrid + rerank (text)":
        (lambda q: thinking_agent.search(query_text=q, k=100, rerank_top_n=20), NT),
}
E.run(METHODS, queries, args.query_sets.split(","), Path(args.output_dir), args.output_name)
print((Path(args.output_dir) / f"{args.output_name}.md").read_text())
