"""Multimodal recipe RAG: standalone CLI + Gradio demo for your own GPU server (e.g. a 24GB RTX A5000 / L4).

    python standalone/app.py                                              # Gradio demo on 0.0.0.0:7860
    python standalone/app.py --query "quick vegetarian dinner with spinach" --answer
    python standalone/app.py --query "টক দই দিয়ে কলার রুটি" --answer        # any of 30+ languages
    python standalone/app.py --image photo.jpg --query "vegetarian"       # combined image + text query
    python standalone/app.py --query "egg sandwich" --rerank-images       # most accurate (image + text rerank)

Default pipeline (best in REPORT.md): Qwen3-VL-Embedding-2B dense search + BM25 (only when the query's words are in the
corpus vocabulary, i.e. English) fused with RRF -> Qwen3-VL-Reranker-2B -> Qwen3.5-4B grounded answer.
Alternatives: --backend nemotron (Nemotron Embed/Rerank VL), --agent (Qwen3.5-4B plans the searches with tool calls).
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
                    help="rerank on recipe image + text (most accurate, ~3x slower) instead of recipe text only")
parser.add_argument("--answer", action="store_true", help="generate a grounded answer with Qwen3.5-4B")
parser.add_argument("--backend", choices=["qwen-vl", "nemotron"], default="qwen-vl")
parser.add_argument("--embeddings", default=None, help="corpus embeddings file for the chosen backend")
parser.add_argument("--agent", action="store_true", help="let Qwen3.5-4B plan the searches with tool calls")
parser.add_argument("--think", action="store_true", help="with --agent: enable Qwen3.5 thinking (slower)")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
args = parser.parse_args()

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from PIL import Image  # noqa: E402

from mmrag import agent as A, pipeline as P, qwen_vl_retrieval as Q  # noqa: E402

SEARCH_KWARGS = {"k": 20, "bm25_min_coverage": 0.8}

print("[INFO] Loading dataset, embeddings and models...")
start = time.time()
dataset = load_dataset(P.DATASET_PATH, split="train")
if args.backend == "qwen-vl":
    embed_model = Q.load_embed_model()
    embeddings_file = args.embeddings or Q.EMBEDDING_FILE
    if not Path(embeddings_file).exists():
        print(f"[INFO] {embeddings_file} not found, embedding the corpus once (~50 min on an L4)...")
        Q.build_document_embeddings(dataset, embed_model, path=embeddings_file)
    rag = Q.QwenVLRAG(dataset, embed_model, Q.load_document_embeddings(embeddings_file), Q.load_rerank_model(), None,
                      *P.load_generation_model())
else:
    rag = P.MultimodalRAG(dataset, P.load_embed_model(), P.load_document_embeddings(args.embeddings),
                          *P.load_rerank_model(modality="image_text"), *P.load_generation_model())
agent = A.AgenticRAG(rag, think=args.think, search_kwargs={"bm25_min_coverage": 0.8}) if args.agent else None
print(f"[INFO] Ready in {time.time() - start:.0f}s | GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")


def run(query_text, query_image, rerank=True, rerank_images=False, answer=False, top_k=3):
    if agent is not None:
        result = agent.search(query_text=query_text, query_image=query_image, k=20,
                              rerank_top_n=20 if rerank else 0, rerank_images=rerank_images)
    else:
        result = rag.search(query_text=query_text, query_image=query_image, **SEARCH_KWARGS)
        if rerank and (query_text or result.query_text):
            # The multilingual reranker scores the user's own words; image-only queries use the VLM's description
            result = rag.rerank(query_text or result.query_text, result, top_n=20, use_images=rerank_images)
    generated = None
    if answer:
        start_time = time.time()
        generated = (agent.answer(query_text, result, query_image=query_image) if agent is not None
                     else rag.answer(query_text, result, query_image=query_image))
        result.timings["answer_s"] = round(time.time() - start_time, 3)
    return result, generated, dataset.select(result.indices[:top_k])


if args.query or args.image:
    image = Image.open(args.image).convert("RGB") if args.image else None
    result, generated, samples = run(args.query, image, rerank=not args.no_rerank, rerank_images=args.rerank_images,
                                     answer=args.answer, top_k=args.top_k)
    if result.query_text and result.query_text != args.query:
        print(f"[INFO] Search text used: {result.query_text!r}")
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
    return gallery, generated or "", {**result.timings, "search_text": result.query_text}


with gr.Blocks(title="Multimodal recipe RAG") as demo:
    gr.Markdown(f"# Multimodal recipe RAG\nBackend: **{args.backend}**{' + agent' if args.agent else ''} · hybrid "
                "dense + BM25 retrieval · cross-encoder rerank · Qwen3.5-4B grounded answers · 10k recipes from "
                "[recipe-synthetic-images-10k](https://huggingface.co/datasets/mrdbourke/recipe-synthetic-images-10k)")
    with gr.Row():
        with gr.Column(scale=1):
            text_in = gr.Textbox(label="Text query (any language)", placeholder="e.g. quick vegetarian dinner with spinach",
                                 lines=2)
            image_in = gr.Image(label="Image query (optional, can be combined with text)", type="pil", height=220)
            rerank_in = gr.Checkbox(label="Rerank top 20", value=True)
            rerank_images_in = gr.Checkbox(label="Rerank with recipe images too (most accurate, slower)", value=False)
            answer_in = gr.Checkbox(label="Generate grounded answer", value=True)
            button = gr.Button("Search", variant="primary")
        with gr.Column(scale=2):
            gallery_out = gr.Gallery(label="Top recipes", columns=3, height="auto", object_fit="contain")
            answer_out = gr.Markdown()
            timings_out = gr.JSON(label="Timings")
    gr.Examples([["quick vegetarian dinner with spinach", None], ["egg sandwich", None],
                 ["pan de plátano con crema agria", None], ["টক দই দিয়ে কলার রুটি", None], ["酸奶油香蕉面包", None]],
                inputs=[text_in, image_in])
    inputs = [text_in, image_in, rerank_in, rerank_images_in, answer_in]
    button.click(demo_fn, inputs, [gallery_out, answer_out, timings_out])
    text_in.submit(demo_fn, inputs, [gallery_out, answer_out, timings_out])

demo.launch(server_name=args.host, server_port=args.port, share=args.share)
