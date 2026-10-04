"""Build reports/report.html and reports/report.pdf from results/*.json and reports/figures/*.svg.

Usage: python scripts/make_report_charts.py && python scripts/make_report.py   (PDF needs google-chrome/chromium)
"""
import html
import json
import shutil
import subprocess
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS, REPORTS = REPO / "results", REPO / "reports"
FIGURES = REPORTS / "figures"
QUERY_SETS = ["title", "paraphrase", "es", "bn", "zh"]
SET_LABELS = ["Title", "Paraphrase", "Spanish", "Bengali", "Chinese"]

RUNS = [  # (results file, pipeline label shown in tables)
    ("eval_results_original_stack.json", "Baseline stack (pinned commits, transformers 4.57)"),
    ("eval_results_improved_reembedded.json", "Improved stack (official Nemotron revisions, re-embedded)"),
    ("eval_results_qwen_vl.json", "Experiment: Qwen3-VL embedding + reranker"),
    ("eval_results_agentic.json", "Experiment: agentic (Qwen3.5-4B tool calls) on Nemotron"),
    ("eval_results_final.json", "Final release: Qwen3-VL + gated hybrid"),
]


def load(file):
    path = RESULTS / file
    return json.loads(path.read_text()) if path.exists() else {}


def avg(results, method, metric):
    values = [results.get(f"{q} | {method}", {}).get(metric) for q in QUERY_SETS]
    return sum(values) / len(values) if None not in values else None


def results_table(metric, fmt="{:.3f}", scale=1.0, highlight=True):
    rows_html = []
    for file, label in RUNS:
        results = load(file)
        methods = list(dict.fromkeys(key.split(" | ")[1] for key in results))
        if not methods:
            continue
        rows_html.append(f"<tr class='group'><td colspan='{len(QUERY_SETS) + 2}'>{html.escape(label)}</td></tr>")
        for method in methods:
            cells = []
            for q in QUERY_SETS:
                row = results.get(f"{q} | {method}")
                cells.append(f"<td>{fmt.format(row[metric] * scale)}</td>" if row else "<td>–</td>")
            mean = avg(results, method, metric)
            cells.append(f"<td><b>{fmt.format(mean * scale)}</b></td>" if mean is not None else "<td>–</td>")
            rows_html.append(f"<tr><td class='method'>{html.escape(method)}</td>{''.join(cells)}</tr>")
    header = "".join(f"<th>{s}</th>" for s in SET_LABELS)
    return (f"<table><thead><tr><th>Method</th>{header}<th>Avg</th></tr></thead>"
            f"<tbody>{''.join(rows_html)}</tbody></table>")


def figure(name, caption):
    svg = (FIGURES / name).read_text()
    svg = svg[svg.index("<svg"):]
    return f"<figure>{svg}<figcaption>{caption}</figcaption></figure>"


def value(file, query_set, method, metric="recall@1"):
    row = load(file).get(f"{query_set} | {method}")
    return f"{row[metric]:.2f}" if row else "–"


B, I, Qv, A, F = (f for f, _ in RUNS)
BASE_FULL = "baseline: dense + rerank (image+text)"
IMPR_FULL = "improved: gated hybrid + rerank (text)"
QWEN_FULL = "qwen3-vl: hybrid + qwen3-vl rerank (text)"
AGENT_FULL = "agentic: tool-call rewrite + hybrid + rerank (text)"
FINAL_TEXT = "final: qwen3-vl gated hybrid + qwen3-vl rerank (text)"
FINAL_IMG = "final: qwen3-vl gated hybrid + qwen3-vl rerank (image+text)"
FINAL_AGENT = "final: agent + qwen3-vl gated hybrid + qwen3-vl rerank (text)"


def fmt_avg(file, method, metric="recall@1"):
    mean = avg(load(file), method, metric)
    return f"{mean:.3f}" if mean is not None else "–"


def fmt_latency(file, method):
    mean = avg(load(file), method, "mean_latency_s")
    return "–" if mean is None else (f"{mean:.1f} s" if mean >= 1 else f"{mean * 1000:.0f} ms")


summary_rows = "".join(
    f"<tr><td>{html.escape(name)}</td><td>{html.escape(stack)}</td>"
    + "".join(f"<td>{value(file, q, method)}</td>" for q in QUERY_SETS)
    + f"<td><b>{fmt_avg(file, method)}</b></td><td>{fmt_latency(file, method)}</td></tr>"
    for name, stack, file, method in [
        ("Baseline (original tutorial)", "Nemotron dense → Nemotron rerank (image+text)", B, BASE_FULL),
        ("Improved", "Nemotron dense + gated BM25 → Nemotron rerank (text)", I, IMPR_FULL),
        ("Exp: Qwen3-VL", "Qwen3-VL dense + BM25 → Qwen3-VL rerank (text)", Qv, QWEN_FULL),
        ("Exp: Agentic", "Qwen3.5-4B tool calls → Nemotron hybrid → rerank", A, AGENT_FULL),
        ("Final (default)", "Qwen3-VL dense + gated BM25 → Qwen3-VL rerank (text)", F, FINAL_TEXT),
        ("Final + image rerank", "… → Qwen3-VL rerank (image+text)", F, FINAL_IMG),
        ("Final + agent", "Qwen3.5-4B tool calls → final pipeline", F, FINAL_AGENT),
    ])

CSS = """
@page { size: A4; margin: 16mm 14mm; }
:root { --text: #0b0b0b; --muted: #52514e; --rule: #e4e3df; --accent: #2a78d6; --surface: #fcfcfb; }
body { font-family: "Inter", "Helvetica Neue", Arial, sans-serif; color: var(--text); background: #fff; font-size: 10pt;
       line-height: 1.45; max-width: 190mm; margin: 0 auto; }
h1 { font-size: 20pt; margin: 0 0 4px; } h2 { font-size: 13.5pt; margin: 22px 0 6px; border-bottom: 1px solid var(--rule);
padding-bottom: 3px; break-after: avoid; } h3 { font-size: 11pt; margin: 14px 0 4px; break-after: avoid; }
.subtitle { color: var(--muted); margin-bottom: 14px; }
table { border-collapse: collapse; width: 100%; font-size: 8.3pt; margin: 6px 0 10px; break-inside: auto; }
th, td { border-bottom: 1px solid var(--rule); padding: 3px 5px; text-align: right; vertical-align: top; }
th:first-child, td:first-child, td.method { text-align: left; } th { color: var(--muted); font-weight: 600; }
tr.group td { text-align: left; font-weight: 700; background: #f3f2ee; }
tr { break-inside: avoid; }
.kpis { display: flex; gap: 10px; margin: 10px 0 6px; } .kpi { flex: 1; border: 1px solid var(--rule); border-radius: 6px;
padding: 8px 10px; } .kpi .v { font-size: 17pt; font-weight: 700; } .kpi .l { color: var(--muted); font-size: 8.5pt; }
figure { margin: 8px 0 12px; break-inside: avoid; } figure svg { width: 100%; height: auto; }
figcaption { color: var(--muted); font-size: 8.5pt; } code { font-size: 8.8pt; background: #f3f2ee; padding: 0 3px;
border-radius: 3px; } ul { margin: 4px 0 8px 18px; padding: 0; } li { margin: 2px 0; }
.note { color: var(--muted); font-size: 8.8pt; }
.summary td:nth-child(2), .summary th:nth-child(2) { text-align: left; }
"""

base_avg, final_avg = fmt_avg(B, BASE_FULL), fmt_avg(F, FINAL_TEXT)
base_lat, final_lat = fmt_latency(B, BASE_FULL), fmt_latency(F, FINAL_TEXT)
multi = lambda file, method: f"{sum(float(value(file, q, method)) for q in ['es', 'bn', 'zh']) / 3:.2f}"

report = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Multimodal RAG evaluation report</title>
<style>{CSS}</style></head><body>
<h1>Multilingual multimodal recipe RAG: evaluation report</h1>
<div class="subtitle">Baseline (original Nemotron VL tutorial) vs. improved pipeline vs. two experiments · Google Colab NVIDIA L4 (24GB) ·
{date.today():%d %B %Y} · repository <code>github.com/ehzawad/multimodal-rag-nemotron</code></div>

<h2>1. Summary</h2>
<div class="kpis">
 <div class="kpi"><div class="v">{base_avg} → {final_avg}</div><div class="l">Mean Recall@1 over 5 query sets, baseline full pipeline → final default</div></div>
 <div class="kpi"><div class="v">{multi(B, BASE_FULL)} → {multi(F, FINAL_TEXT)}</div><div class="l">Mean Recall@1 on Spanish, Bengali, Chinese queries</div></div>
 <div class="kpi"><div class="v">{base_lat} → {final_lat}</div><div class="l">Mean latency per query (retrieval + rerank)</div></div>
</div>
<p>We fixed the original tutorial notebook so it runs on a 24GB GPU, then built and evaluated three alternatives on a shared
1,000-query known-item benchmark (200 recipes × title, paraphrase, Spanish, Bengali and Chinese queries). The best pipeline,
now the repository's <code>main</code>, replaces the Nemotron VL models with <b>Qwen3-VL-Embedding-2B</b> and
<b>Qwen3-VL-Reranker-2B</b>, adds <b>BM25 fused with Reciprocal Rank Fusion only for queries whose words occur in the corpus
vocabulary</b>, reranks on recipe text, and answers with <b>Qwen3.5-4B</b>. It is more accurate on every query set and an
order of magnitude faster than the baseline's image + text reranking.</p>
<table class="summary"><thead><tr><th>Pipeline</th><th>Stack</th>{''.join(f'<th>{s}</th>' for s in SET_LABELS)}<th>Avg R@1</th><th>Latency</th></tr></thead>
<tbody>{summary_rows}</tbody></table>
<p class="note">Recall@1 of the target recipe. Rerank and agent rows use the first 50 queries per set (image + text rerank: 25), fast
rows all 200; see §4 for n and confidence.</p>

<h2>2. Task, data and hardware</h2>
<ul>
<li><b>Dataset:</b> <code>mrdbourke/recipe-synthetic-images-10k</code>, 10,096 Food.com recipes, each a rendered 1275×1650 recipe-card
image plus markdown (title, description, ingredients, steps, tags, five reviews).</li>
<li><b>Task:</b> retrieve the right recipe for a text query (any language), an image, or both; optionally generate an answer grounded in
the top three recipes.</li>
<li><b>Hardware:</b> one Google Colab NVIDIA L4 (24GB, 72W power cap), driven from the terminal with colab-persist over SSH. All final
models together use 17.7GB of GPU memory, so the pipeline also fits an RTX A5000 (24GB).</li>
<li><b>One-off corpus embedding cost on the L4:</b> Qwen3-VL-Embedding-2B 48.5 min (image capped at ~400 tokens + recipe text without
reviews); Nemotron Embed VL 2 h 19 min (tutorial format: 6 tiles + thumbnail and full text, ~4k tokens per recipe).</li>
</ul>

<h2>3. Pipelines compared</h2>
<table><thead><tr><th>Branch</th><th>Retrieval</th><th>Rerank</th><th>Generation</th></tr></thead><tbody>
<tr><td>Baseline</td><td>Nemotron Embed VL 1B (pinned commit 5b5ca69, transformers 4.57), dense only</td><td>Nemotron Rerank VL 1B, image + text, batched</td><td>Qwen3-VL-2B summary</td></tr>
<tr><td>Improved</td><td>Nemotron Embed VL (official revision 582e3bf, corpus re-embedded) + BM25, RRF; image queries described in text by the VLM</td><td>Nemotron Rerank VL (revision b8a9987), recipe text only</td><td>Qwen3.5-4B, grounded, cited</td></tr>
<tr><td>Exp: Qwen3-VL</td><td>Qwen3-VL-Embedding-2B (image + text documents) + BM25, RRF</td><td>Qwen3-VL-Reranker-2B</td><td>Qwen3.5-4B</td></tr>
<tr><td>Exp: Agentic</td><td>Qwen3.5-4B calls a <code>search_recipes</code> tool 1–3 times (translating / rewriting); each call runs the improved hybrid search; rankings fused with the original query's</td><td>Nemotron Rerank VL, text</td><td>Qwen3.5-4B, answers in the user's language</td></tr>
<tr><td><b>Final (main)</b></td><td>Qwen3-VL-Embedding-2B + BM25 <b>gated on query vocabulary coverage ≥ 0.8</b>, RRF</td><td>Qwen3-VL-Reranker-2B, text (image + text optional)</td><td>Qwen3.5-4B; agent optional</td></tr>
</tbody></table>

<h2>4. Evaluation method</h2>
<p>Known-item retrieval on 200 recipes sampled with seed 42 (<code>results/eval_queries.json</code>). For each target recipe there are
five queries: its <b>title</b>; a 5–12 word <b>paraphrase</b> written by Qwen3.5-4B from the recipe without the title's words; and the
paraphrase machine-translated by Qwen3.5-4B into <b>Spanish</b>, <b>Bengali</b> and <b>Chinese</b>. We report Recall@1/@10 and MRR@10
of the target recipe and mean latency (retrieval, rerank and any LLM planning; answer generation excluded, ~20 s with Qwen3.5-4B).</p>
<p class="note">Caveats: queries are synthetic; the Bengali translations are rough (many English food words transliterated, a few
mistranslations); near-duplicate recipes exist (e.g. two "banana sour cream bread" recipes) so Recall@1 under-counts; slow methods
were evaluated on 50 (rerank, agent) or 25 (image + text rerank, thinking agent) queries per set, giving roughly ±7–10 points of
sampling error, so differences smaller than that between slow methods are not significant.</p>

<h2>5. Results</h2>
{figure("pipelines_recall1.svg", "Figure 1. Recall@1 per query set for the representative full pipeline of each branch.")}
{figure("fast_recall1.svg", "Figure 2. Retrieval only (no rerank): the embedding model is the main lever for non-English queries; ungated BM25 hurts Spanish.")}
{figure("accuracy_vs_latency.svg", "Figure 3. Mean Recall@1 against mean latency per query. *Baseline stack with text-only rerank exposes the old reranker's text-only failure.")}
<h3>Recall@1 (all runs and methods)</h3>{results_table("recall@1")}
<h3>MRR@10</h3>{results_table("mrr@10")}
<h3>Recall@10</h3>{results_table("recall@10")}
<h3>Mean latency per query (ms)</h3>{results_table("mean_latency_s", fmt="{:.0f}", scale=1000)}
<h3>Queries evaluated (n)</h3>{results_table("n", fmt="{:.0f}")}

<h2>6. Findings</h2>
<h3>6.1 The original notebook did not run, and had latent bugs</h3>
<ul>
<li><code>ValueError: Sample larger than population</code>: image paths were globbed from a local folder that only existed on the author's
machine (0 images). Fixed by sampling the dataset's <code>image</code> column.</li>
<li>Reranking 20 image + text pairs in one forward pass runs out of memory on a 24GB GPU (CUDA OOM at 20.4GB) → batched (4 pairs).</li>
<li>The reranker's processor kwargs were built under a misspelled name and never passed; image queries were passed to the reranker
as its text <code>question</code>; <code>get_top_n_scores_and_samples</code> ignored <code>top_n</code>; "Run all" blocked on
<code>demo.launch(debug=True)</code> and would have tried to publish a Space. All fixed (<code>notebooks/multimodal_rag_tutorial_baseline.ipynb</code>,
end to end on the L4 in 138 s).</li>
</ul>
<h3>6.2 Hidden embedding-compatibility trap</h3>
<p>The demo Space's precomputed document embeddings only reproduce with the tutorial's pinned embed commit (cosine 0.9999). Every official
revision of <code>llama-nemotron-embed-vl-1b-v2</code> tokenizes and masks differently (cosine ≈ 0.86, under both transformers 4.57 and 5.x),
and the pinned commit cannot run on transformers 5, which Qwen3.5 requires. Re-embedding the corpus with the official revision
fixed the mismatch <i>and</i> improved plain dense retrieval: title Recall@1 {value(B, 'title', 'baseline: dense')} → {value(I, 'title', 'baseline: dense')},
paraphrase {value(B, 'paraphrase', 'baseline: dense')} → {value(I, 'paraphrase', 'baseline: dense')}.</p>
<h3>6.3 The reranker revision matters more than image inputs</h3>
<p>With the tutorial's pinned reranker commit, image + text reranking is <i>worse</i> than no reranking on most sets (mean Recall@1
{fmt_avg(B, BASE_FULL)} vs {fmt_avg(B, 'baseline: dense')}) at ~15 s per query, and text-only reranking is broken (title Recall@1
{value(B, 'title', 'improved: hybrid + rerank (text)')}). The current official revision reranks recipe text well in ~0.6 s
(title {value(I, 'title', IMPR_FULL)}, paraphrase {value(I, 'paraphrase', IMPR_FULL)}). Dropping the reviews from the reranked text halves its cost.</p>
<h3>6.4 Hybrid search helps English and hurts other languages: gate it</h3>
<p>BM25 over the English corpus improves English paraphrase retrieval but injects noise for other languages; with Qwen3-VL, ungated
hybrid drops Spanish Recall@1 from {value(Qv, 'es', 'qwen3-vl: dense')} to {value(Qv, 'es', 'qwen3-vl: hybrid (dense + bm25, RRF)')}.
A free, language-agnostic gate fixes this: use BM25 only if ≥ 80% of the query's non-stopword tokens occur in the corpus vocabulary.
On our queries this passes 100% of English, 9% of Spanish and ~0% of Bengali/Chinese queries, keeping the English gains and restoring
dense-only behaviour elsewhere (Nemotron Spanish: {value(I, 'es', 'improved: hybrid (dense + bm25, RRF)')} ungated vs
{value(I, 'es', 'improved: gated hybrid (bm25 if coverage >= 0.8)')} gated).</p>
<h3>6.5 A multilingual multimodal embedding model is the biggest single lever</h3>
<p>Qwen3-VL-Embedding-2B improves dense Recall@1 on non-English queries 1.6–3.6× versus the re-embedded Nemotron (Spanish
{value(I, 'es', 'baseline: dense')} → {value(Qv, 'es', 'qwen3-vl: dense')}, Bengali {value(I, 'bn', 'baseline: dense')} → {value(Qv, 'bn', 'qwen3-vl: dense')},
Chinese {value(I, 'zh', 'baseline: dense')} → {value(Qv, 'zh', 'qwen3-vl: dense')}) while matching it on English, and its reranker (with the BM25 gate) adds
another 16–30 points on every non-English set.</p>
<h3>6.6 Agentic search planning helps weak retrievers but does not win</h3>
<p>Letting Qwen3.5-4B translate/rewrite the request via tool calls lifts the Nemotron pipeline on Bengali (dense {value(I, 'bn', 'baseline: dense')}
→ agent {value(A, 'bn', 'agentic: tool-call rewrite + hybrid')}) but hurts exact-title queries (rewriting loses the exact name), costs
~5 s per query with the reference kernels, and thinking mode adds ~10 s more with no accuracy gain. On top of the Qwen3-VL pipeline it
does not improve the average ({fmt_avg(F, FINAL_AGENT)} with agent vs {fmt_avg(F, FINAL_TEXT)} without), so it ships as an option
(<code>--agent</code>) for vague or multi-part requests rather than the default.</p>
<h3>6.7 Image + text reranking: best for English only</h3>
<p>Reranking with the recipe-card images reaches the best English numbers (title {value(F, 'title', FINAL_IMG)}, paraphrase
{value(F, 'paraphrase', FINAL_IMG)}) but is worse than text-only reranking for non-English queries and ~3× slower; text reranking is the
default and image + text an option (<code>--rerank-images</code>).</p>
<h3>6.8 Generation</h3>
<p>Qwen3.5-4B (natively multimodal, Feb 2026) produces grounded, cited markdown answers and describes image queries for lexical search
(e.g. the kitchen-bench photo → "tomato pasta with fresh basil and parmesan cheese"). Thinking is disabled for latency; an answer takes
~20 s on the L4 because <code>causal-conv1d</code> is not installed (flash-linear-attention is).</p>

<h2>7. Recommendation (what is on main)</h2>
<ul>
<li>Default: Qwen3-VL-Embedding-2B dense + vocabulary-gated BM25 (RRF) → Qwen3-VL-Reranker-2B on recipe text → Qwen3.5-4B grounded answer.</li>
<li>Options: <code>--rerank-images</code> for maximum English accuracy, <code>--agent</code> for vague requests, <code>--backend nemotron</code> for the
improved Nemotron stack. The fixed original tutorial is kept as <code>notebooks/multimodal_rag_tutorial_baseline.ipynb</code> /
<code>standalone/baseline_app.py</code>.</li>
<li>Future work: human-labelled multi-answer relevance judgements; native-speaker queries (especially Bengali); install causal-conv1d or
serve Qwen3.5 with vLLM for faster answers; Qwen3-VL-Embedding-8B on a 48GB GPU; host the 41MB corpus embeddings on the Hub.</li>
</ul>

<h2>8. Reproduce</h2>
<p><code>pip install -r requirements.txt</code> · <code>python scripts/build_qwen_vl_embeddings.py</code> (corpus, ~50 min) ·
<code>python scripts/eval_final.py</code> · baselines: <code>MMRAG_NEMOTRON_REVISIONS=original</code> with transformers 4.57.3
(<code>requirements-baseline.txt</code>) and <code>python scripts/eval_retrieval.py</code> · <code>python scripts/eval_qwen_vl.py</code> ·
<code>python scripts/eval_agentic.py</code> · <code>python scripts/make_report_charts.py &amp;&amp; python scripts/make_report.py</code>.</p>
</body></html>"""

REPORTS.mkdir(exist_ok=True)
(REPORTS / "report.html").write_text(report)
chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
if chrome:
    subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={REPORTS / 'report.pdf'}", (REPORTS / "report.html").as_uri()],
                   check=True, capture_output=True)
    print(f"Wrote {REPORTS / 'report.html'} and {REPORTS / 'report.pdf'}")
else:
    print(f"Wrote {REPORTS / 'report.html'} (no Chrome found for the PDF)")
