"""Improved multimodal recipe RAG: standalone CLI + Gradio demo for your own GPU server (e.g. a 24GB A5000/L4).

    python standalone/app.py                                            # Gradio demo on 0.0.0.0:7860
    python standalone/app.py --query "quick vegetarian dinner with spinach" --answer
    python standalone/app.py --image photo.jpg --query "vegetarian"     # combined image + text query
    python standalone/app.py --query "egg sandwich" --no-rerank --top-k 5

Pipeline: Nemotron Embed VL dense search + BM25 (RRF fusion) -> Nemotron Rerank VL -> Qwen3.5-4B grounded answer.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--query", help="text query (runs once in the terminal instead of launching the demo)")
parser.add_argument("--image", help="image query path (can be combined with --query)")
parser.add_argument("--top-k", type=int, default=3, help="number of recipes to show")
parser.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder rerank step")
parser.add_argument("--rerank-images", action="store_true",
                    help="rerank on recipe image + text (slower) instead of recipe text only")
parser.add_argument("--answer", action="store_true", help="generate a grounded answer with Qwen3.5-4B")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
args = parser.parse_args()

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from PIL import Image  # noqa: E402

from mmrag import pipeline as P  # noqa: E402

print("[INFO] Loading dataset, embeddings and models...")
start = time.time()
dataset = load_dataset(P.DATASET_PATH, split="train")
rag = P.MultimodalRAG(dataset,
                      P.load_embed_model(),
                      P.load_document_embeddings(device="cuda" if torch.cuda.is_available() else "cpu"),
                      *P.load_rerank_model(modality="image_text"),
                      *P.load_generation_model())
print(f"[INFO] Ready in {time.time() - start:.0f}s | GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")


def run(query_text, query_image, rerank=True, rerank_images=False, answer=False, top_k=3):
    result = rag.search(query_text=query_text, query_image=query_image, k=20)
    if rerank and result.query_text:
        result = rag.rerank(result.query_text, result, top_n=20, use_images=rerank_images)
    generated = None
    if answer:
        start_time = time.time()
        generated = rag.answer(query_text, result, query_image=query_image)
        result.timings["answer_s"] = round(time.time() - start_time, 3)
    return result, generated, dataset.select(result.indices[:top_k])


if args.query or args.image:
    image = Image.open(args.image).convert("RGB") if args.image else None
    result, generated, samples = run(args.query, image, rerank=not args.no_rerank, rerank_images=args.rerank_images,
                                     answer=args.answer, top_k=args.top_k)
    if result.query_text and result.query_text != args.query:
        print(f"[INFO] Image query rewritten to: {result.query_text!r}")
    for rank, (sample, score) in enumerate(zip(samples, result.scores), 1):
        print(f"[RESULT] #{rank} {sample['name']} (id {sample['id']}, {sample['minutes']} min) | score {score:.4f}")
    if generated:
        print("\n" + generated + "\n")
    print(f"[INFO] Timings: {result.timings}")
    sys.exit(0)

import gradio as gr  # noqa: E402


def demo_fn(query_text, query_image, rerank, rerank_images, answer):
    if not (query_text and query_text.strip()) and query_image is None:
        raise gr.Error("Please provide a text query, an image query or both.")
    result, generated, samples = run(query_text, query_image, rerank=rerank, rerank_images=rerank_images,
                                     answer=answer, top_k=3)
    gallery = [(s["image"], f"#{i} {s['name']}") for i, s in enumerate(samples, 1)]
    return gallery, generated or "", {**result.timings, "query_used_for_bm25_and_rerank": result.query_text}


with gr.Blocks(title="Multimodal recipe RAG") as demo:
    gr.Markdown("# Multimodal recipe RAG\nHybrid retrieval (Nemotron Embed VL + BM25) · Nemotron Rerank VL · "
                "Qwen3.5-4B grounded answers · 10k recipes from "
                "[recipe-synthetic-images-10k](https://huggingface.co/datasets/mrdbourke/recipe-synthetic-images-10k)")
    with gr.Row():
        with gr.Column(scale=1):
            text_in = gr.Textbox(label="Text query", placeholder="e.g. quick vegetarian dinner with spinach", lines=2)
            image_in = gr.Image(label="Image query (optional, can be combined with text)", type="pil", height=220)
            rerank_in = gr.Checkbox(label="Rerank top 20", value=True)
            rerank_images_in = gr.Checkbox(label="Rerank with recipe images too (slower)", value=False)
            answer_in = gr.Checkbox(label="Generate grounded answer", value=True)
            button = gr.Button("Search", variant="primary")
        with gr.Column(scale=2):
            gallery_out = gr.Gallery(label="Top recipes", columns=3, height="auto", object_fit="contain")
            answer_out = gr.Markdown()
            timings_out = gr.JSON(label="Timings")
    gr.Examples([["quick vegetarian dinner with spinach", None], ["egg sandwich", None],
                 ["summer lunch ideas", None], ["Banana Sour Cream Bread", None]], inputs=[text_in, image_in])
    inputs = [text_in, image_in, rerank_in, rerank_images_in, answer_in]
    button.click(demo_fn, inputs, [gallery_out, answer_out, timings_out])
    text_in.submit(demo_fn, inputs, [gallery_out, answer_out, timings_out])

demo.launch(server_name=args.host, server_port=args.port, share=args.share)
