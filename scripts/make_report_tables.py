"""Merge evaluation result files into comparison tables (markdown) for REPORT.md.

Usage: python scripts/make_report_tables.py results/eval_results_original_stack.json results/eval_results_improved_reembedded.json ...
Each file is a {"<query set> | <method>": metrics} mapping written by mmrag.evaluation.
"""
import json
import sys
from pathlib import Path

QUERY_SETS = ["title", "paraphrase", "es", "bn", "zh"]
LABELS = {"title": "Title", "paraphrase": "Paraphrase", "es": "Spanish", "bn": "Bengali", "zh": "Chinese"}

rows = {}  # (source, method) -> {query_set: metrics}
for path in sys.argv[1:]:
    source = Path(path).stem.replace("eval_results_", "").replace("eval_results", "default")
    for key, metrics in json.loads(Path(path).read_text()).items():
        query_set, method = key.split(" | ")
        rows.setdefault((source, method), {})[query_set] = metrics


def table(metric, fmt="{:.3f}", scale=1.0):
    header = "| Pipeline (eval run) | Method | " + " | ".join(LABELS[q] for q in QUERY_SETS) + " | Avg |"
    lines = [header, "|---|---|" + "---|" * (len(QUERY_SETS) + 1)]
    for (source, method), by_set in rows.items():
        values = [by_set[q][metric] * scale if q in by_set else None for q in QUERY_SETS]
        present = [v for v in values if v is not None]
        cells = [fmt.format(v) if v is not None else "–" for v in values]
        average = fmt.format(sum(present) / len(present)) if present else "–"
        lines.append(f"| {source} | {method} | " + " | ".join(cells) + f" | {average} |")
    return "\n".join(lines)


print("### Recall@1\n")
print(table("recall@1"))
print("\n### MRR@10\n")
print(table("mrr@10"))
print("\n### Recall@10\n")
print(table("recall@10"))
print("\n### Mean latency per query (ms)\n")
print(table("mean_latency_s", fmt="{:.0f}", scale=1000))
print("\n### Queries evaluated (n)\n")
print(table("n", fmt="{:.0f}"))
