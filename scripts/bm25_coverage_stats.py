"""Distribution of BM25 vocabulary coverage per query set (to calibrate the language gate threshold)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets import load_dataset
from mmrag import evaluation as E, pipeline as P

texts = load_dataset(P.DATASET_PATH, split="train", columns=["recipe_markdown"])["recipe_markdown"]
index = P.BM25Index(texts)
queries = E.load_queries()
for key in E.QUERY_SETS:
    cov = sorted(index.vocabulary_coverage(q[key]) for q in queries)
    pct = lambda p: cov[int(p * (len(cov) - 1))]
    passed = {t: sum(c >= t for c in cov) / len(cov) for t in (0.5, 0.6, 0.7, 0.8, 0.9)}
    print(f"{key:10s} p10={pct(.1):.2f} p50={pct(.5):.2f} p90={pct(.9):.2f} pass-rate {passed}")
