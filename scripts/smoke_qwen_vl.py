"""Quick check that the Qwen3-VL embedding/reranker load and measure document embedding throughput."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from datasets import load_dataset
from mmrag import qwen_vl_retrieval as Q

dataset = load_dataset("mrdbourke/recipe-synthetic-images-10k", split="train")
model = Q.load_embed_model()
print("[INFO] embed loaded", torch.cuda.memory_allocated() / 1e9)
sub = dataset.select(range(32))
docs = [Q._document(t, i) for t, i in zip(sub["recipe_markdown"], sub["image"])]
for bs in [4, 8]:
    torch.cuda.synchronize(); t = time.time()
    emb = model.encode(docs, batch_size=bs, convert_to_tensor=True, normalize_embeddings=True)
    torch.cuda.synchronize(); print(f"[INFO] bs={bs}: {len(docs) / (time.time() - t):.2f} docs/s, shape {tuple(emb.shape)}, peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
q = model.encode(["banana bread with sour cream", "pan de plátano con crema agria", "টক দই দিয়ে কলার রুটি"], convert_to_tensor=True, prompt=Q.QUERY_PROMPT, normalize_embeddings=True)
print("[INFO] query sims to doc 0..3:", (q @ emb[:4].T).tolist())
reranker = Q.load_rerank_model()
pairs = [("artichoke bacon appetizer", d) for d in docs[:4]]
print("[INFO] rerank scores:", reranker.predict(pairs, batch_size=4, prompt=Q.RERANK_PROMPT))
print("[INFO] names:", sub["name"][:4])
