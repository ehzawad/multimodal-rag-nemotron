"""Build notebooks/multimodal_rag_tutorial.ipynb from the original tutorial notebook with bug fixes.

Usage: python scripts/build_baseline_notebook.py <original.ipynb>
"""
import json
import sys
import textwrap
from pathlib import Path

SRC = Path(sys.argv[1])
DST = Path(__file__).resolve().parent.parent / "notebooks" / "multimodal_rag_tutorial.ipynb"

nb = json.loads(SRC.read_text())
cells = nb["cells"]


def src(i):
    return "".join(cells[i]["source"])


def set_src(i, text):
    cells[i]["source"] = text.strip("\n").splitlines(keepends=True)


def replace(i, old, new, count=1):
    s = src(i)
    assert s.count(old) >= 1, f"cell {i}: pattern not found: {old[:60]!r}"
    set_src(i, s.replace(old, new, count))


def code_cell(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


SPACE_REPO = "mrdbourke/multimodal-rag-with-nemotron"

# --- [6] Image paths: the local folder from dataset creation doesn't exist anywhere else ---------------
assert "PATH_TO_IMAGES" in src(6)
set_src(6, '''
import random

# Get 100 random images
# Note: An earlier version of this cell searched a local folder created while generating the dataset
# ("2026-01-19_03-52-56-recipe_to_pdf_and_image_outputs/images"). That folder only exists on the machine
# the dataset was created on, so it found 0 images and random.sample() raised a ValueError.
# The images are already stored in the dataset's "image" column, so we sample from there instead.
random.seed(42)

all_image_indices = list(range(len(dataset["train"])))
print(f"[INFO] Number of total images: {len(all_image_indices)}")

image_indices_100 = random.sample(all_image_indices, k=min(100, len(all_image_indices)))
images_100 = dataset["train"].select(image_indices_100)["image"]
print(f"[INFO] Sampled {len(images_100)} images | Example image size: {images_100[0].size}")
''')

# --- [11] Rerank processor kwargs were built but never passed (and misspelled) -------------------------
replace(11, "rereank_processor_kwargs = {", "rerank_processor_kwargs = {")
replace(11, '''rerank_processor = AutoProcessor.from_pretrained(
    RERANK_MODEL_PATH,
    revision=RERANK_COMMIT_HASH,
    trust_remote_code=True,
)''', '''rerank_processor = AutoProcessor.from_pretrained(
    RERANK_MODEL_PATH,
    revision=RERANK_COMMIT_HASH,
    trust_remote_code=True,
    **rerank_processor_kwargs
)''')

# --- [13] Download the precomputed embeddings instead of re-embedding 10k samples; respect batch size ---
replace(13, "from safetensors.torch import save_file, load_file\n",
        "from safetensors.torch import save_file, load_file\nfrom huggingface_hub import hf_hub_download\n")
replace(13, "image_text_embeddings = None\n", f'''image_text_embeddings = None

# The precomputed image + text embeddings are hosted alongside the demo Space, download them if we don't have them
# (creating them from scratch takes a long time, see the CREATE_EMBEDDINGS branch below)
if not os.path.exists(PATH_TO_EMBEDDING_FILE):
    try:
        print(f"[INFO] Downloading precomputed embeddings from Space: {SPACE_REPO}")
        hf_hub_download(repo_id="{SPACE_REPO}",
                        repo_type="space",
                        filename=PATH_TO_EMBEDDING_FILE,
                        local_dir=".")
    except Exception as e:
        print(f"[INFO] Couldn't download precomputed embeddings ({{e}})")
''')
replace(13, '''        indicies_to_select = list(range(i, i+8))

        # Make sure indices stay lower than the total number of samples
        indicies_to_select = [item for item in indicies_to_select if item <= len(dataset["train"]) - 1]''',
        '''        # Note: stop at the end of the dataset so the last batch can be smaller than IMAGE_TEXT_BATCH_SIZE
        indicies_to_select = list(range(i, min(i + IMAGE_TEXT_BATCH_SIZE, len(dataset["train"]))))''')
replace(13, '''    save_file({"image_text_embeddings": image_text_embeddings_all}, PATH_TO_EMBEDDING_FILE)''',
        '''    save_file({"image_text_embeddings": image_text_embeddings_all.contiguous()}, PATH_TO_EMBEDDING_FILE)
    image_text_embeddings = image_text_embeddings_all.to(DEVICE)''')

# --- [20] The local "moussaka.jpeg" isn't distributed with the notebook ---------------------------------
assert 'test_image_path = "moussaka.jpeg"' in src(20)
set_src(20, f'''
from pathlib import Path
from huggingface_hub import hf_hub_download

# Test with an image
# Note: Use your own local image if you have one (e.g. "moussaka.jpeg"), otherwise download the demo's
# example image (a kitchen bench with ingredients on it)
test_image_path = "moussaka.jpeg"
if not Path(test_image_path).is_file():
    test_image_path = "kitchen_bench.png"
    if not Path(test_image_path).is_file():
        hf_hub_download(repo_id="{SPACE_REPO}", repo_type="space", filename=test_image_path, local_dir=".")

test_image_query = Image.open(test_image_path).convert("RGB")
test_image_query.resize(size=(500, 500))
''')

# --- [27] `top_n == len(scores)` was a no-op comparison and top_n was ignored ([:3] hardcoded) ----------
assert "top_n == len(scores)" in src(27)
set_src(27, '''
def get_top_n_scores_and_samples(scores, indicies, top_n=3):
    top_n = min(top_n, len(scores))

    top_samples = dataset["train"].select([int(i) for i in indicies[:top_n]])
    top_scores_and_samples = []
    for score, sample in zip(scores[:top_n], top_samples):
        top_scores_and_samples.append({"score": round(score.item(), 4),
                                       "sample": sample})
    return top_scores_and_samples
''')

# --- [52] + [64] The reranker takes a text question; image queries were passed to it as `question` ------
IMAGE_RERANK_GUARD = '''
    # Note: the reranker scores a *text* question against documents, so only rerank text queries
    if rerank_option == "True" and not isinstance(input_query, str):
        print("[INFO] Reranking is for text queries only, skipping reranking for image query...")
        rerank_option = "False"
'''
replace(52, "    ### Reranking steps ###\n", "    ### Reranking steps ###\n" + IMAGE_RERANK_GUARD)
replace(64, "    # === Reranking (optional) ===\n", "    # === Reranking (optional) ===\n" + IMAGE_RERANK_GUARD)
replace(64, "rereank_processor_kwargs = {", "rerank_processor_kwargs = {")
s64 = src(64)
old = '''rerank_processor = AutoProcessor.from_pretrained(
    RERANK_MODEL_PATH,
    revision=RERANK_COMMIT_HASH,
    trust_remote_code=True,
)'''
if old in s64:
    replace(64, old, old.replace("trust_remote_code=True,\n)", "trust_remote_code=True,\n    **rerank_processor_kwargs\n)"))

# --- [32] + [36] + [64] Reranking 20 image + text pairs in one forward pass OOMs on a 24GB GPU (L4) ----------
replace(32, "NUM_TOP_SAMPLES_TO_RERANK = 20 # Note: the more samples you try to rerank, the longer it will take (reranking is a one-to-one comparison)\n",
        "NUM_TOP_SAMPLES_TO_RERANK = 20 # Note: the more samples you try to rerank, the longer it will take (reranking is a one-to-one comparison)\n"
        "RERANK_BATCH_SIZE = 4 # Note: image + text pairs use a lot of memory, 20 pairs in one batch runs out of memory on a 24GB GPU (e.g. Colab L4)\n")
replace(32, """# Process the input samples
batch_dict_rerank = rerank_processor.process_queries_documents_crossencoder(samples_to_rerank)

# Move samples to target device
batch_dict_rerank = {
    k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch_dict_rerank.items()
}

# Run rerank inference
with torch.inference_mode():
    rerank_outputs = rerank_model(**batch_dict_rerank, return_dict=True)
""", """from transformers.modeling_outputs import SequenceClassifierOutput

# Rerank in batches to keep GPU memory bounded
rerank_logits_chunks = []
for i in range(0, len(samples_to_rerank), RERANK_BATCH_SIZE):
    # Process the input samples
    batch_dict_rerank = rerank_processor.process_queries_documents_crossencoder(samples_to_rerank[i:i + RERANK_BATCH_SIZE])

    # Move samples to target device
    batch_dict_rerank = {
        k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch_dict_rerank.items()
    }

    # Run rerank inference
    with torch.inference_mode():
        rerank_logits_chunks.append(rerank_model(**batch_dict_rerank, return_dict=True).logits)

rerank_outputs = SequenceClassifierOutput(logits=torch.cat(rerank_logits_chunks, dim=0))
""")
replace(36, """    dataset_split: str = "train",
) -> torch.Tensor:""", """    dataset_split: str = "train",
    batch_size: int = 4,
) -> torch.Tensor:""")
replace(36, """        dataset_split: Dataset split to use (e.g., "train", "test").
""", """        dataset_split: Dataset split to use (e.g., "train", "test").
        batch_size: Number of query/document pairs to rerank per forward pass (keeps GPU memory bounded).
""")
replace(36, """    # Process the input samples
    batch_dict_rerank = rerank_processor.process_queries_documents_crossencoder(
        samples_to_rerank
    )

    # Move samples to target device
    batch_dict_rerank = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch_dict_rerank.items()
    }

    # Run rerank inference
    with torch.inference_mode():
        rerank_outputs = rerank_model(**batch_dict_rerank, return_dict=True)

    # Get logits and sort
    rerank_logits = rerank_outputs.logits.squeeze(-1)""", """    # Rerank in batches to keep GPU memory bounded
    rerank_logits_chunks = []
    for i in range(0, len(samples_to_rerank), batch_size):
        # Process the input samples
        batch_dict_rerank = rerank_processor.process_queries_documents_crossencoder(
            samples_to_rerank[i:i + batch_size]
        )

        # Move samples to target device
        batch_dict_rerank = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch_dict_rerank.items()
        }

        # Run rerank inference
        with torch.inference_mode():
            rerank_logits_chunks.append(rerank_model(**batch_dict_rerank, return_dict=True).logits)

    # Get logits and sort
    rerank_logits = torch.cat(rerank_logits_chunks, dim=0).squeeze(-1)""")
replace(64, """    dataset_split: str = "train",
) -> tuple:""", """    dataset_split: str = "train",
    batch_size: int = 4,
) -> tuple:""")
replace(64, """    batch_dict_rerank = rerank_processor.process_queries_documents_crossencoder(samples_to_rerank)
    batch_dict_rerank = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch_dict_rerank.items()
    }

    with torch.inference_mode():
        rerank_outputs = rerank_model(**batch_dict_rerank, return_dict=True)

    rerank_logits = rerank_outputs.logits.squeeze(-1)""", """    # Rerank in batches to keep GPU memory bounded (20 image + text pairs at once runs out of memory on a 24GB GPU)
    rerank_logits_chunks = []
    for i in range(0, len(samples_to_rerank), batch_size):
        batch_dict_rerank = rerank_processor.process_queries_documents_crossencoder(samples_to_rerank[i:i + batch_size])
        batch_dict_rerank = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch_dict_rerank.items()
        }

        with torch.inference_mode():
            rerank_logits_chunks.append(rerank_model(**batch_dict_rerank, return_dict=True).logits)

    rerank_logits = torch.cat(rerank_logits_chunks, dim=0).squeeze(-1)""")

# --- [60] Don't block when the notebook is executed non-interactively ----------------------------------
replace(60, """demo.launch(
    share=False,
    debug=True,
    inline=True,
    show_error=True,
    height=1000
)""", """import os

# Note: set the environment variable HEADLESS=1 (e.g. when running the notebook top to bottom from a script)
# to build the demo without launching it (debug=True blocks the cell until it is interrupted)
if not os.environ.get("HEADLESS"):
    demo.launch(
        share=False,
        debug=True,
        inline=True,
        show_error=True,
        height=1000
    )
else:
    print("[INFO] HEADLESS is set, skipping demo.launch()")""")

# --- [70] Never publish to the Hub by accident when running all cells -----------------------------------
assert "create_repo(" in src(70)
set_src(70, "UPLOAD_TO_HF = False # set to True to create/update the Hugging Face Space (requires a token with write access)\n\n"
        "if UPLOAD_TO_HF:\n" + textwrap.indent(src(70), "    ") +
        '\nelse:\n    print("[INFO] UPLOAD_TO_HF=False, skipping upload to Hugging Face")')

# --- New install cell after the overview -----------------------------------------------------------------
INSTALL = code_cell('''
# Install dependencies (Google Colab already ships torch and torchvision)
# Note: transformers is pinned to the version the Nemotron VL remote code (at the commits used below) was built against
!pip install -q "transformers==4.57.3" "accelerate>=1.12.0" "datasets>=4.4.2" "safetensors>=0.7.0" "qwen-vl-utils==0.0.14" "gradio>=5"
''')
cells.insert(2, md_cell("## Install dependencies"))
cells.insert(3, INSTALL)

# --- Clean outputs / widget state so the repo diff stays readable ----------------------------------------
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None
    c.get("metadata", {}).pop("colab", None)
    c.get("metadata", {}).pop("outputId", None)
nb["metadata"].pop("widgets", None)
for n, c in enumerate(cells):
    c["id"] = f"cell-{n:02d}"
nb["nbformat_minor"] = max(nb.get("nbformat_minor", 0), 5)

DST.parent.mkdir(parents=True, exist_ok=True)
DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"Wrote {DST} ({len(cells)} cells)")
