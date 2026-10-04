"""Shared known-item retrieval evaluation (used by scripts/eval_*.py on every branch)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import torch

REPO_DIR = Path(__file__).resolve().parent.parent
QUERIES_FILE = REPO_DIR / "results" / "eval_queries.json"
QUERY_SETS = ["title", "paraphrase", "es", "bn", "zh"]


def load_queries(path=QUERIES_FILE, limit=None) -> list[dict]:
    return json.loads(Path(path).read_text())[:limit]


def evaluate(method: Callable, queries: list[dict], query_key: str, limit: int) -> dict:
    """method(query_text) -> object with `.indices`. Target rank is looked up in the returned indices."""
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


def run(methods: dict[str, tuple[Callable, int]], queries: list[dict], query_sets: list[str], output_dir: Path,
        output_name: str) -> dict:
    """methods: name -> (method, number of queries to evaluate). Writes <output_name>.json and .md."""
    results = {}
    for query_key in query_sets:
        for name, (method, limit) in methods.items():
            metrics = evaluate(method, queries, query_key, limit)
            results[f"{query_key} | {name}"] = metrics
            print(f"[{query_key:10s}] {name:48s} n={metrics['n']:4d} R@1={metrics['recall@1']:.3f} "
                  f"R@5={metrics['recall@5']:.3f} R@10={metrics['recall@10']:.3f} MRR@10={metrics['mrr@10']:.3f} "
                  f"latency={metrics['mean_latency_s'] * 1000:.0f}ms", flush=True)
            # Save as we go so a crash or timeout keeps the finished rows
            write(results, output_dir, output_name)
    return results


def write(results: dict, output_dir: Path, output_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{output_name}.json").write_text(json.dumps(results, indent=1))
    lines = ["| Query set | Method | n | R@1 | R@5 | R@10 | MRR@10 | Latency |", "|---|---|---|---|---|---|---|---|"]
    for key, m in results.items():
        query_key, name = key.split(" | ")
        lines.append(f"| {query_key} | {name} | {m['n']} | {m['recall@1']:.3f} | {m['recall@5']:.3f} "
                     f"| {m['recall@10']:.3f} | {m['mrr@10']:.3f} | {m['mean_latency_s'] * 1000:.0f} ms |")
    (output_dir / f"{output_name}.md").write_text("\n".join(lines) + "\n")
