"""Re-embed the 10k recipe corpus (image + text) with the official Nemotron Embed VL revision used by mmrag.

Needed because the demo Space's precomputed embeddings only match the tutorial's pinned commit. ~45 min on an L4.
Usage (GPU): python scripts/build_nemotron_embeddings.py --output nemotron_embed_vl_582e3bf_image_text_10k.safetensors
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets import load_dataset  # noqa: E402

from mmrag import pipeline as P  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--output", default=P.REEMBEDDED_FILE)
parser.add_argument("--batch-size", type=int, default=8)
args = parser.parse_args()

start = time.time()
dataset = load_dataset(P.DATASET_PATH, split="train")
embeddings = P.build_document_embeddings(dataset, P.load_embed_model(), batch_size=args.batch_size, path=args.output)
print(f"[INFO] Embedded {len(embeddings)} recipes with revision {P.EMBED_REVISION[:7]} in {time.time() - start:.0f}s "
      f"-> {args.output}")
