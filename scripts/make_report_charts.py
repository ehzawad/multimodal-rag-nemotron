"""Render the report figures (SVG) from the evaluation result files in results/.

Usage: python scripts/make_report_charts.py  ->  reports/figures/*.svg
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT = REPO / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

QUERY_SETS = ["title", "paraphrase", "es", "bn", "zh"]
SET_LABELS = ["Title", "Paraphrase", "Spanish", "Bengali", "Chinese"]

# Validated categorical palette (light mode), fixed order; text never wears series colors
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# One representative (full-pipeline) configuration per branch: (label, results file, method)
PIPELINES = [
    ("Baseline (main)", "eval_results_original_stack.json", "baseline: dense + rerank (image+text)"),
    ("Improved", "eval_results_improved_reembedded.json", "improved: gated hybrid + rerank (text)"),
    ("Exp: Qwen3-VL", "eval_results_qwen_vl.json", "qwen3-vl: hybrid + qwen3-vl rerank (text)"),
    ("Exp: Agentic", "eval_results_agentic.json", "agentic: tool-call rewrite + hybrid + rerank (text)"),
    ("Final (release)", "eval_results_final.json", "final: qwen3-vl gated hybrid + qwen3-vl rerank (text)"),
]
# Fast (no rerank) retrieval configurations
FAST = [
    ("Nemotron dense (baseline)", "eval_results_original_stack.json", "baseline: dense"),
    ("Nemotron hybrid", "eval_results_improved_reembedded.json", "improved: hybrid (dense + bm25, RRF)"),
    ("Nemotron gated hybrid", "eval_results_improved_reembedded.json", "improved: gated hybrid (bm25 if coverage >= 0.8)"),
    ("Qwen3-VL dense", "eval_results_qwen_vl.json", "qwen3-vl: dense"),
    ("Qwen3-VL gated hybrid", "eval_results_final.json", "final: qwen3-vl gated hybrid"),
]


def load(file):
    path = RESULTS / file
    return json.loads(path.read_text()) if path.exists() else {}


def metric(results, query_set, method, name):
    row = results.get(f"{query_set} | {method}")
    return row[name] if row else None


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def grouped_bars(configs, name, filename, title, ylabel):
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=SURFACE)
    style(ax)
    n = len(configs)
    width = 0.8 / n
    for i, (label, file, method) in enumerate(configs):
        results = load(file)
        values = [metric(results, q, method, name) for q in QUERY_SETS]
        xs = [g + (i - (n - 1) / 2) * width for g in range(len(QUERY_SETS))]
        heights = [v if v is not None else 0 for v in values]
        # 2px surface gap between adjacent bars via edgecolor
        ax.bar(xs, heights, width, color=SERIES[i], edgecolor=SURFACE, linewidth=2, label=label)
        for x, v in zip(xs, values):
            ax.text(x, (v or 0) + 0.015, "–" if v is None else f"{v:.2f}", ha="center", va="bottom", fontsize=6.5,
                    color=TEXT_2)
    ax.set_xticks(range(len(QUERY_SETS)), SET_LABELS, color=TEXT)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(ylabel, color=TEXT_2)
    ax.set_title(title, loc="left", color=TEXT, fontsize=11, fontweight="bold")
    ax.legend(frameon=False, ncol=n, loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=8, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(OUT / filename, format="svg")
    plt.close(fig)


def accuracy_vs_latency(filename):
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=SURFACE)
    style(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    points = []
    for label, file, method in FAST + PIPELINES + [
        ("Final + image rerank", "eval_results_final.json", "final: qwen3-vl gated hybrid + qwen3-vl rerank (image+text)"),
        ("Baseline dense + text rerank*", "eval_results_original_stack.json", "improved: hybrid + rerank (text)"),
    ]:
        results = load(file)
        r1 = [metric(results, q, method, "recall@1") for q in QUERY_SETS]
        lat = [metric(results, q, method, "mean_latency_s") for q in QUERY_SETS]
        if None in r1 or None in lat:
            continue
        points.append((label, sum(lat) / len(lat) * 1000, sum(r1) / len(r1)))
    # Hand-placed offsets for points that sit close together
    offsets = {"Qwen3-VL dense": (6, 6), "Qwen3-VL gated hybrid": (6, -11), "Nemotron hybrid": (-8, 6),
               "Nemotron dense (baseline)": (6, -11), "Nemotron gated hybrid": (6, 4)}
    for label, x, y in points:
        ax.scatter([x], [y], s=64, color=SERIES[0], edgecolor=SURFACE, linewidth=2, zorder=3)
        dx, dy = offsets.get(label, (6, 4))
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7.5, color=TEXT,
                    ha="right" if dx < 0 else "left")
    ax.set_xscale("log")
    ax.set_xlim(left=6)  # room for the left-placed labels
    ax.set_xlabel("Mean latency per query (ms, log scale)", color=TEXT_2)
    ax.set_ylabel("Mean Recall@1 over 5 query sets", color=TEXT_2)
    ax.set_title("Accuracy vs. latency (higher and further left is better)", loc="left", color=TEXT, fontsize=11,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / filename, format="svg")
    plt.close(fig)


grouped_bars(PIPELINES, "recall@1", "pipelines_recall1.svg", "Recall@1 by query set: full pipelines", "Recall@1")
grouped_bars(PIPELINES, "mrr@10", "pipelines_mrr10.svg", "MRR@10 by query set: full pipelines", "MRR@10")
grouped_bars(FAST, "recall@1", "fast_recall1.svg", "Recall@1 by query set: retrieval only (no rerank)", "Recall@1")
accuracy_vs_latency("accuracy_vs_latency.svg")
print("Wrote", sorted(p.name for p in OUT.iterdir()))
