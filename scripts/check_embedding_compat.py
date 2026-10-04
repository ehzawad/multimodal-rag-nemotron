"""Compare freshly computed Nemotron document embeddings with the precomputed ones under different settings."""
import sys
import torch, transformers
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModel

revision = sys.argv[1]
stored = load_file(hf_hub_download("mrdbourke/multimodal-rag-with-nemotron", "image_text_embeddings_10k.safetensors", repo_type="space"))["image_text_embeddings"].float()
stored = torch.nn.functional.normalize(stored, dim=-1)
ds = load_dataset("mrdbourke/recipe-synthetic-images-10k", split="train").select([0, 42, 4242])
model = AutoModel.from_pretrained("nvidia/llama-nemotron-embed-vl-1b-v2", revision=revision, dtype=torch.bfloat16,
                                  trust_remote_code=True, attn_implementation="sdpa", device_map="cuda").eval()
p = model.processor
print(f"[INFO] transformers {transformers.__version__} revision {revision[:7]} defaults: p_max_length={getattr(p, 'p_max_length', None)} max_input_tiles={getattr(p, 'max_input_tiles', None)} use_thumbnail={getattr(p, 'use_thumbnail', None)}")
settings = {"defaults": {}, "p_max_length=10240": {"p_max_length": 10240},
            "10240 + tiles6 + thumb": {"p_max_length": 10240, "max_input_tiles": 6, "use_thumbnail": True}}
original = {k: getattr(p, k, None) for k in ["p_max_length", "max_input_tiles", "use_thumbnail"]}
for name, kw in settings.items():
    for k, v in original.items():
        setattr(p, k, v)
    for k, v in kw.items():
        setattr(p, k, v)
    with torch.inference_mode():
        fresh = model.encode_documents(images=ds["image"], texts=ds["recipe_markdown"]).float()
    cos = (torch.nn.functional.normalize(fresh.cpu(), dim=-1) * stored[[0, 42, 4242]]).sum(-1)
    print(f"[INFO] {name:24s} cosine to precomputed: {[round(c, 4) for c in cos.tolist()]}")
