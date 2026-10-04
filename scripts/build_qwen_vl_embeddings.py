"""Embed the 10k recipe corpus (image + text) with Qwen3-VL-Embedding-2B. One-off, ~30-40 min on an L4.

Usage (GPU): python scripts/build_qwen_vl_embeddings.py --output qwen3_vl_embedding_2b_image_text_10k.safetensors
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets import load_dataset  # noqa: E402

from mmrag import pipeline as P, qwen_vl_retrieval as Q  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--output", default=Q.EMBEDDING_FILE)
parser.add_argument("--batch-size", type=int, default=8)
args = parser.parse_args()

dataset = load_dataset(P.DATASET_PATH, split="train")
Q.build_document_embeddings(dataset, Q.load_embed_model(), batch_size=args.batch_size, path=args.output)
