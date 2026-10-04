"""
Multimodal RAG Demo with Nemotron Embed VL and Rerank VL

A Gradio demo for multimodal retrieval augmented generation using:
- Dataset: mrdbourke/recipe-synthetic-images-10k
- Embedding model: nvidia/llama-nemotron-embed-vl-1b-v2
- Rerank model: nvidia/llama-nemotron-rerank-vl-1b-v2
- Generation model: Qwen/Qwen3-VL-2B-Instruct

Standalone version of the baseline notebook's demo for running on your own GPU server (e.g. a 24GB A5000/L4):

    python standalone/baseline_app.py                                  # launch the Gradio demo on 0.0.0.0:7860
    python standalone/baseline_app.py --query "eggplant dip" --rerank  # one query from the terminal
    python standalone/baseline_app.py --image kitchen_bench.png --summary
"""

import argparse
import os
import time
from pathlib import Path

import torch
import gradio as gr
from PIL import Image
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)

# ============================================================================
# Configuration
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model paths and commit hashes (required for sdpa attention on HF Spaces)
EMBED_MODEL_PATH = "nvidia/llama-nemotron-embed-vl-1b-v2"
EMBED_COMMIT_HASH = "5b5ca69c35bf6ec1484d2d5ff238626e67a745e2"

RERANK_MODEL_PATH = "nvidia/llama-nemotron-rerank-vl-1b-v2"
RERANK_COMMIT_HASH = "47e5a355d1a050c3e5f69d53f14964b1d34bcd9d"

GENERATION_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

# Precomputed image + text embeddings and the example image are hosted with the original demo Space
SPACE_REPO = "mrdbourke/multimodal-rag-with-nemotron"
DATA_DIR = Path(os.environ.get("MMRAG_DATA_DIR", Path(__file__).resolve().parent))


def _space_file(filename: str) -> str:
    """Return a local path to `filename`, downloading it from the demo Space if needed."""
    local_path = DATA_DIR / filename
    if local_path.exists():
        return str(local_path)
    return hf_hub_download(repo_id=SPACE_REPO, repo_type="space", filename=filename, local_dir=DATA_DIR)

# ============================================================================
# Load Dataset and Embeddings
# ============================================================================

print("[INFO] Loading dataset...")
dataset = load_dataset(path="mrdbourke/recipe-synthetic-images-10k")
print(f"[INFO] Dataset loaded with {len(dataset['train'])} samples")

print("[INFO] Loading embeddings...")
image_text_embeddings = load_file(_space_file("image_text_embeddings_10k.safetensors"))
image_text_embeddings = image_text_embeddings["image_text_embeddings"].to(DEVICE)
print(f"[INFO] Embeddings loaded: {image_text_embeddings.shape}")

# ============================================================================
# Load Models
# ============================================================================
modality_to_tokens = {
    "image": 2048,
    "image_text": 10240,
    "text": 8192
}

print(f"[INFO] Loading embedding model from: {EMBED_MODEL_PATH} with commit: {EMBED_COMMIT_HASH}")
embed_model = AutoModel.from_pretrained(
    EMBED_MODEL_PATH,
    revision=EMBED_COMMIT_HASH,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="sdpa",
    device_map="auto",
).eval()

# Set embed processor kwargs
# Note: These are the suggest settings from the embed model card
embed_modality = "image_text"
embed_processor_kwargs = {
    "max_input_tiles": 6,
    "use_thumbnail": True,
    "p_max_length": modality_to_tokens[embed_modality]
}

embed_processor = AutoProcessor.from_pretrained(
    EMBED_MODEL_PATH,
    revision=EMBED_COMMIT_HASH,
    trust_remote_code=True,
    **embed_processor_kwargs
)
print(f"[INFO] Embedding model loaded!")
print(f"[INFO] Embed processor using p_max_length: {embed_processor.p_max_length}")

print(f"[INFO] Loading rerank model from: {RERANK_MODEL_PATH} with commit: {RERANK_COMMIT_HASH}")
rerank_model = AutoModelForSequenceClassification.from_pretrained(
    RERANK_MODEL_PATH,
    revision=RERANK_COMMIT_HASH,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="sdpa",
    device_map="auto",
).eval()

# Set rerank processor kwargs
# Note: These are the suggest settings from the rerank model card
rerank_modality = "image_text"
rerank_processor_kwargs = {
    "max_input_tiles": 6,
    "use_thumbnail": True,
    "rerank_max_length": modality_to_tokens[rerank_modality]
}

rerank_processor = AutoProcessor.from_pretrained(
    RERANK_MODEL_PATH,
    revision=RERANK_COMMIT_HASH,
    trust_remote_code=True,
    **rereank_processor_kwargs
)
print(f"[INFO] Rerank processor using rerank_max_length: {rerank_processor.rerank_max_length}")

print(f"[INFO] Rerank model loaded!")

print("[INFO] Loading generation model...")
qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
    GENERATION_MODEL_ID,
    dtype="auto",
    device_map="auto"
)
qwen_processor = AutoProcessor.from_pretrained(GENERATION_MODEL_ID)
print(f"[INFO] Generation model loaded")

# ============================================================================
# Helper Functions
# ============================================================================

def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2 normalize a tensor along the last dimension."""
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)


def match_query_to_embeddings(
    query: str | Image.Image,
    target_embeddings_to_match: torch.Tensor,
    top_k: int = 100
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Match a query (text or image) to target embeddings.

    Returns:
        Tuple of (sorted_scores, sorted_indices)
    """
    with torch.inference_mode():
        if isinstance(query, Image.Image):
            query_embeddings = embed_model.encode_documents(images=[query])
        else:
            query_embeddings = embed_model.encode_queries([query])

    cos_sim = _l2_normalize(query_embeddings) @ _l2_normalize(target_embeddings_to_match).T
    cos_sim_flat = cos_sim.flatten()
    sorted_indices = torch.argsort(cos_sim_flat, descending=True)[:top_k]
    sorted_scores = cos_sim_flat[sorted_indices][:top_k]

    return sorted_scores, sorted_indices


def rerank_samples(
    dataset,
    query_text: str,
    sorted_indices: list | torch.Tensor,
    num_samples_to_rerank: int,
    rerank_model,
    rerank_processor,
    device: str = DEVICE,
    text_column: str = "recipe_markdown",
    image_column: str = "image",
    dataset_split: str = "train",
    batch_size: int = 4,
) -> tuple:
    """
    Rerank top samples using the vision-language reranker model.

    Returns:
        Tuple of (dataset_samples_to_rerank, rerank_sorted_indices)
    """
    top_indices = sorted_indices[:num_samples_to_rerank]
    dataset_samples_to_rerank = dataset[dataset_split].select(top_indices)

    texts_to_rerank = dataset_samples_to_rerank[text_column]
    images_to_rerank = dataset_samples_to_rerank[image_column]

    samples_to_rerank = [
        {"question": query_text, "doc_text": text, "doc_image": image}
        for text, image in zip(texts_to_rerank, images_to_rerank)
    ]

    # Rerank in batches to keep GPU memory bounded (20 image + text pairs at once runs out of memory on a 24GB GPU)
    rerank_logits_chunks = []
    for i in range(0, len(samples_to_rerank), batch_size):
        batch_dict_rerank = rerank_processor.process_queries_documents_crossencoder(samples_to_rerank[i:i + batch_size])
        batch_dict_rerank = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch_dict_rerank.items()
        }

        with torch.inference_mode():
            rerank_logits_chunks.append(rerank_model(**batch_dict_rerank, return_dict=True).logits)

    rerank_logits = torch.cat(rerank_logits_chunks, dim=0).squeeze(-1)
    rerank_sorted_indices = torch.argsort(rerank_logits, descending=True)

    return dataset_samples_to_rerank, rerank_sorted_indices


def generate_recipe_summary(
    recipe_texts: list[str],
    model = None,
    processor = None,
    max_new_tokens: int = 512
) -> str:
    """Generate a markdown summary of multiple recipes."""
    if model is None:
        model = qwen_model
    if processor is None:
        processor = qwen_processor

    recipes_combined = ""
    for i, recipe in enumerate(recipe_texts[:3], 1):
        recipes_combined += f"\n\n--- RECIPE {i} ---\n{recipe}"

    prompt = f"""You are a helpful culinary assistant. Below are {len(recipe_texts[:3])} recipes.
Please provide a brief markdown summary with:
- A short 1-2 sentence overview of each recipe
- Key ingredients highlighted
- Estimated difficulty (Easy/Medium/Hard)
- Which recipe might be best for a quick weeknight dinner

For example use the following format:

```markdown
# Recipe summary

## <recipe_name>

[details]

## <recipe_name>

[details]

## <recipe_name>

[details]
```

Keep the summary concise and well-formatted in markdown. Return in ```markdown``` tags so it can be easily parsed.

<recipes>
{recipes_combined}
</recipes>

## Summary:"""

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output_text.strip()


def _markdown_to_simple_html(markdown_text: str, max_reviews: int = 1) -> str:
    """Convert recipe markdown to a simple HTML card."""
    lines = markdown_text.strip().split('\n')

    title = ""
    description = ""
    recipe_id = ""
    cook_time = ""
    num_ratings = ""
    ingredients = []
    steps = []
    tags = []
    reviews = []

    current_section = None
    in_ingredients = False
    in_steps = False
    in_reviews = False
    in_tags = False
    review_count = 0

    for line in lines:
        line = line.strip()

        if line.startswith('# ') and not title:
            title = line[2:].strip()
            continue

        if line.startswith('**ID:**'):
            recipe_id = line.replace('**ID:**', '').strip()
            continue
        if line.startswith('**Time:**'):
            cook_time = line.replace('**Time:**', '').strip()
            continue
        if line.startswith('**Number of Ratings:**'):
            num_ratings = line.replace('**Number of Ratings:**', '').strip()
            continue

        if line.startswith('## '):
            section_name = line[3:].strip().lower()
            in_ingredients = section_name == 'ingredients'
            in_steps = section_name.startswith('steps')
            in_reviews = section_name == 'reviews'
            in_tags = section_name == 'tags'
            current_section = section_name
            continue

        if current_section == 'description' and line and not line.startswith('#'):
            description = line
            continue

        if in_ingredients and line.startswith('- '):
            ingredients.append(line[2:].strip())
            continue

        if in_steps and line and line[0].isdigit():
            step_text = line.split('. ', 1)[-1] if '. ' in line else line
            steps.append(step_text.strip())
            continue

        if in_tags and line.startswith('`'):
            tag_list = [t.strip().strip('`') for t in line.split(',')]
            tags.extend(tag_list)
            continue

        if in_reviews and line.startswith('> ') and review_count < max_reviews:
            reviews.append(line[2:].strip())
            review_count += 1
            continue

    html = f'''
    <div style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 4px; background: #fff; font-family: system-ui, -apple-system, sans-serif; font-size: 12px; height: 400px; overflow-y: auto; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <div style="font-weight: bold; font-size: 14px; color: #333; margin-bottom: 8px;">{title}</div>

        <div style="display: flex; gap: 12px; font-size: 11px; color: #666; margin-bottom: 10px; flex-wrap: wrap;">
            {f'<span>⏱️ {cook_time}</span>' if cook_time else ''}
            {f'<span>⭐ {num_ratings} ratings</span>' if num_ratings else ''}
            {f'<span style="color: #999;">ID: {recipe_id}</span>' if recipe_id else ''}
        </div>

        <div style="color: #555; margin-bottom: 12px; font-style: italic; line-height: 1.4;">{description[:150]}{"..." if len(description) > 150 else ""}</div>

        <div style="margin-bottom: 12px;">
            <div style="font-weight: bold; font-size: 11px; color: #333; margin-bottom: 4px;">📝 Ingredients</div>
            <div style="color: #444; line-height: 1.5;">{", ".join(ingredients[:8])}{"..." if len(ingredients) > 8 else ""}</div>
        </div>

        <div style="margin-bottom: 12px;">
            <div style="font-weight: bold; font-size: 11px; color: #333; margin-bottom: 4px;">👨‍🍳 Steps ({len(steps)} total)</div>
            <ol style="margin: 0; padding-left: 20px; color: #444; line-height: 1.5;">
                {"".join(f'<li style="margin-bottom: 4px;">{step[:80]}{"..." if len(step) > 80 else ""}</li>' for step in steps[:4])}
                {f'<li style="color: #999;">...and {len(steps) - 4} more steps</li>' if len(steps) > 4 else ''}
            </ol>
        </div>
    '''

    if tags:
        display_tags = tags[:5]
        html += f'''
        <div style="margin-bottom: 12px;">
            <div style="font-weight: bold; font-size: 11px; color: #333; margin-bottom: 4px;">🏷️ Tags</div>
            <div style="display: flex; flex-wrap: wrap; gap: 4px;">
                {"".join(f'<span style="background: #f0f0f0; padding: 2px 6px; border-radius: 4px; font-size: 10px;">{tag}</span>' for tag in display_tags)}
                {f'<span style="color: #999; font-size: 10px;">+{len(tags) - 5} more</span>' if len(tags) > 5 else ''}
            </div>
        </div>
        '''

    if reviews:
        html += f'''
        <div style="border-top: 1px solid #eee; padding-top: 10px; margin-top: 10px;">
            <div style="font-weight: bold; font-size: 11px; color: #333; margin-bottom: 4px;">💬 Review</div>
            <div style="color: #555; font-size: 11px; line-height: 1.4; background: #f9f9f9; padding: 8px; border-radius: 4px; font-style: italic;">"{reviews[0][:200]}{"..." if len(reviews[0]) > 200 else ""}"</div>
        </div>
        '''

    html += '</div>'
    return html


def create_recipe_cards_html(
    scores_and_samples: list[dict],
    num_results: int = 3,
    text_key: str = "text",
    max_reviews: int = 1
) -> str:
    """Generate combined HTML cards from scored recipe samples."""
    recipe_cards_html = []

    for item in scores_and_samples[:num_results]:
        sample = item["sample"]
        markdown_text = sample.get(text_key, "") or sample.get("markdown", "")
        card_html = _markdown_to_simple_html(markdown_text, max_reviews=max_reviews)
        recipe_cards_html.append(f'<div style="flex: 1; min-width: 0;">{card_html}</div>')

    combined_html = f'''
    <div style="margin-top: 16px;">
        <h3 style="font-family: system-ui, -apple-system, sans-serif; font-size: 16px; font-weight: 600; color: #333; margin-bottom: 12px;">Retrieved Texts</h3>
        <div style="display: flex; flex-direction: row; gap: 12px; width: 100%;">
            {"".join(recipe_cards_html)}
        </div>
    </div>
    '''

    return combined_html


# ============================================================================
# Main Retrieve Function
# ============================================================================

def retrieve(
    query_text: str | None,
    query_image: Image.Image | None,
    rerank_option: str,
    generate_summary_option: str
):
    """
    Main retrieval function for the Gradio interface.

    Args:
        query_text: Text query input
        query_image: Image query input (PIL Image)
        rerank_option: "True" or "False" to enable reranking
        generate_summary_option: "True" or "False" to enable summary generation

    Returns:
        Tuple of (image_gallery, recipe_cards_html, summary, timing_dict)
    """

    embeddings_on_gpu = image_text_embeddings

    # Determine input query (prefer text over image)
    if query_text and query_text.strip():
        input_query = query_text
    elif query_image is not None:
        input_query = query_image
    else:
        raise gr.Error("Please provide either a text query or an image query.")

    # === Retrieval ===
    start_time_query_to_embed_match = time.time()
    result_sorted_scores, result_sorted_indices = match_query_to_embeddings(
        query=input_query,
        target_embeddings_to_match=embeddings_on_gpu,
        top_k=20
    )
    end_time_query_to_embed_match = time.time()

    top_dataset_results_to_show = dataset["train"].select(result_sorted_indices)

    scores_and_samples = [
        {"score": round(score.item(), 4), "sample": sample}
        for score, sample in zip(result_sorted_scores, top_dataset_results_to_show)
    ]

    output_image_gallery = [
        (item["sample"]["image"], f'{item["sample"]["name"]} | Score: {item["score"]}')
        for item in scores_and_samples[:3]
    ]

    output_recipe_cards_html = create_recipe_cards_html(
        scores_and_samples=scores_and_samples,
        num_results=3,
        text_key="recipe_markdown",
        max_reviews=1
    )

    # === Reranking (optional) ===

    # Note: the reranker scores a *text* question against documents, so only rerank text queries
    if rerank_option == "True" and not isinstance(input_query, str):
        print("[INFO] Reranking is for text queries only, skipping reranking for image query...")
        rerank_option = "False"
    if rerank_option == "True":
        start_time_reranking = time.time()
        dataset_samples_to_rerank, rerank_sorted_indicies = rerank_samples(
            sorted_indices=result_sorted_indices,
            dataset=dataset,
            dataset_split="train",
            query_text=input_query,
            num_samples_to_rerank=20,
            rerank_model=rerank_model,
            rerank_processor=rerank_processor
        )
        end_time_reranking = time.time()
        rerank_time = round(end_time_reranking - start_time_reranking, 4)

        top_dataset_results_to_show = dataset_samples_to_rerank.select(rerank_sorted_indicies)
        samples_and_rerank_changes = []
        for new_rank, (sample, original_rank) in enumerate(zip(top_dataset_results_to_show, rerank_sorted_indicies)):
            movement = new_rank - original_rank
            if movement == 0:
                movement_string = f"{movement}"
            else:
                movement_string = f"+{abs(movement)}" if movement < 0 else f"-{movement}"
            rerank_string = f"Original rank: {original_rank} | New rank: {new_rank} | Movement: {movement_string}"
            samples_and_rerank_changes.append({"sample": sample, "rerank_string": rerank_string})

        output_image_gallery = [
            (item["sample"]["image"], f'{item["sample"]["name"]} | {item["rerank_string"]}')
            for item in samples_and_rerank_changes[:3]
        ]
        output_recipe_cards_html = create_recipe_cards_html(
            scores_and_samples=samples_and_rerank_changes,
            num_results=3,
            text_key="recipe_markdown",
            max_reviews=1
        )
    else:
        rerank_time = "Reranking turned off"

    # === Generation (optional) ===
    if generate_summary_option == "True":
        start_time_generation_output = time.time()
        if rerank_option == "True":
            recipe_texts = [item["sample"]["recipe_markdown"] for item in samples_and_rerank_changes[:3]]
        else:
            recipe_texts = [item["sample"]["recipe_markdown"] for item in scores_and_samples[:3]]

        summary = generate_recipe_summary(recipe_texts)
        summary = summary.replace("```markdown", "").replace("```", "")
        end_time_generation_output = time.time()
        generation_time = round(end_time_generation_output - start_time_generation_output, 4)
    else:
        generation_time = "Generation turned off"
        summary = "Generation turned off, no summary created"

    timing_dict = {
        "query_embed_and_match_time": round(end_time_query_to_embed_match - start_time_query_to_embed_match, 4),
        "rerank_time": rerank_time,
        "generation_time": generation_time
    }

    return output_image_gallery, output_recipe_cards_html, summary, timing_dict


# ============================================================================
# Gradio Interface
# ============================================================================

with gr.Blocks(title="Multimodal RAG Demo") as demo:
    gr.Markdown("""# 👁️📑 Multimodal RAG Demo with Nemotron Embed VL and Rerank VL

Input an image or text about food and get recipe images/text back.

This is a scalable workflow that can lend itself to many use cases such as business document retrieval, technical manual look ups and more.

By default it returns the top 3 results from a database of 10,000+ recipes. We've limited it to 3 for the demo but in practice you could return as many as you like.

* **Dataset used:** https://huggingface.co/datasets/mrdbourke/recipe-synthetic-images-10k
* **Embedding model used:** https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2
    * **Note:** By default we use the image + text embeddings as we have access to image and text pairs in our dataset, and according to the [launch blog post](https://huggingface.co/blog/nvidia/llama-nemotron-vl-1b), these work the best.
* **Rerank model used:** https://huggingface.co/nvidia/llama-nemotron-rerank-vl-1b-v2
* **Generation model used:** https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct (note: you could use a larger model such as [Nemotron v3](https://huggingface.co/collections/nvidia/nvidia-nemotron-v3), however, this will require more compute resources)
""")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Query Input")

            query_text = gr.Textbox(
                label="Text Query",
                placeholder="e.g. 'dinner recipes with tomatoes'",
                lines=2
            )

            query_image = gr.Image(
                label="Image Query (optional)",
                type="pil",
                height=200
            )

            generate_summary_option = gr.Radio(
                choices=["True", "False"],
                value="False",
                label="Generate recipe summary"
            )

            rerank_option = gr.Radio(
                choices=["True", "False"],
                value="False",
                label="Rerank initial results? (note: reranking is for text queries only)"
            )

            search_btn = gr.Button("Search", variant="primary", size="lg")

        with gr.Column(scale=2):
            gr.Markdown("### Retrieved Results")

            gallery_output = gr.Gallery(
                label="Retrieved Recipe Images",
                columns=3,
                height="auto",
                object_fit="cover",
                show_label=True
            )

            recipes_html = gr.HTML(label="Retrieved Recipe Texts")

            summary_generation = gr.Markdown(
                label="Retrieved Recipe Summary (generated from top results)"
            )

            timing_output = gr.JSON(label="Timings")

    gr.Markdown("### Example Queries")

    gr.Examples(
        examples=[
            ["best omelette recipes", None, "False", "False"],
            ["best omelette recipes", None, "False", "True"],
            ["best omelette recipes", None, "True", "True"],
            ["eggplant dip", None, "True", "True"],
            [None, _space_file("kitchen_bench.png"), "False", "True"]
        ],
        inputs=[query_text, query_image, rerank_option, generate_summary_option],
        label="Example Queries"
    )

    search_btn.click(
        fn=retrieve,
        inputs=[query_text, query_image, rerank_option, generate_summary_option],
        outputs=[gallery_output, recipes_html, summary_generation, timing_output]
    )

def _print_results(query, rerank, summary):
    """Run one query from the terminal and print the top recipes."""
    output_image_gallery, _, generated_summary, timing_dict = retrieve(
        query_text=query if isinstance(query, str) else None,
        query_image=query if isinstance(query, Image.Image) else None,
        rerank_option=str(rerank),
        generate_summary_option=str(summary),
    )
    for rank, (_, caption) in enumerate(output_image_gallery, 1):
        print(f"[RESULT] #{rank} {caption}")
    if summary:
        print(generated_summary)
    print(f"[INFO] Timings: {timing_dict}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--query", help="text query to run once from the terminal (skips the web demo)")
    parser.add_argument("--image", help="path to an image query to run once from the terminal")
    parser.add_argument("--rerank", action="store_true", help="rerank the top 20 results (text queries only)")
    parser.add_argument("--summary", action="store_true", help="generate a summary of the top 3 recipes")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
    args = parser.parse_args()

    if args.query or args.image:
        query = args.query if args.query else Image.open(args.image).convert("RGB")
        _print_results(query, args.rerank, args.summary)
    else:
        demo.launch(server_name=args.host, server_port=args.port, share=args.share)