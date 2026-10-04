"""Create the shared retrieval evaluation query set (results/eval_queries.json).

For a fixed random sample of recipes (seed 42) we store, per target recipe:
* title       - the recipe name
* paraphrase  - a search query written by Qwen3.5-4B from the recipe without using the title's words
* es, bn, zh  - the paraphrase translated to Spanish, Bengali and Chinese (multilingual retrieval)

The file is committed so every branch/pipeline is evaluated on exactly the same queries.
Usage (GPU): python scripts/make_eval_queries.py --num-queries 200
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mmrag import pipeline as P  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--num-queries", type=int, default=200)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", default=str(Path(__file__).resolve().parent.parent / "results" / "eval_queries.json"))
args = parser.parse_args()

from datasets import load_dataset  # noqa: E402

dataset = load_dataset(P.DATASET_PATH, split="train")
generation_model, generation_processor = P.load_generation_model()


class _Generator(P.MultimodalRAG):
    def __init__(self):  # only the generation part is needed
        self.generation_model, self.generation_processor = generation_model, generation_processor


generator = _Generator()


def generate(prompt, max_new_tokens=48):
    return generator._generate([{"type": "text", "text": prompt}], max_new_tokens=max_new_tokens,
                               do_sample=False).strip().strip('"').strip()


LANGUAGES = {"es": "Spanish", "bn": "Bengali", "zh": "Simplified Chinese"}

random.seed(args.seed)
target_ids = random.sample(range(len(dataset)), k=args.num_queries)
samples = dataset.select(target_ids)
queries = []
start = time.time()
for doc_id, name, recipe in zip(target_ids, samples["name"], samples["recipe_markdown"]):
    title = re.sub(r"\s+", " ", name).strip()
    paraphrase = generate(f"Here is a recipe:\n\n{recipe.split(chr(10) + '## Reviews')[0][:3000]}\n\n"
                          f"Write one short search query (5 to 12 words) that a home cook might type into a recipe "
                          f"search engine when looking for a dish like this. Describe it by its main ingredients, "
                          f"cooking method, cuisine or occasion. Do NOT use any of these title words: \"{title}\". "
                          f"Reply with the query only.", max_new_tokens=32)
    query = {"doc_id": doc_id, "title": title, "paraphrase": paraphrase}
    for code, language in LANGUAGES.items():
        query[code] = generate(f"Translate this recipe search query into natural {language}, as a native speaker "
                               f"would type it. Reply with the translation only.\n\n{paraphrase}")
    queries.append(query)
    if len(queries) % 25 == 0:
        print(f"[INFO] {len(queries)}/{args.num_queries} queries ({time.time() - start:.0f}s)", flush=True)

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(json.dumps(queries, indent=1, ensure_ascii=False) + "\n")
print(f"[INFO] Wrote {len(queries)} queries to {args.output} in {time.time() - start:.0f}s")
for q in queries[:3]:
    print(json.dumps(q, ensure_ascii=False))
