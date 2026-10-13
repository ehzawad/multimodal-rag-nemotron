"""Versioned, bounded public contracts, shared by transports and model backends."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import warnings
from typing import Annotated, Literal

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EMBED_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
EMBED_REVISION = "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
RERANK_MODEL = "Qwen/Qwen3-VL-Reranker-2B"
RERANK_REVISION = "4bd860ac4f15ad1897a214615cccc700f8f71818"
GEN_MODEL = "Qwen/Qwen3.5-4B"
GEN_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
QUERY_PROMPT = "Retrieve recipes (recipe card image and text) relevant to the user's search query."
RERANK_PROMPT = "Judge whether the recipe matches the user's recipe search query."
EMBED_DIM = 2048
IMAGE_MAX_PIXELS = 400 * 28 * 28
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 8_000_000
IndexID = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,62}$")]
Text = Annotated[str, Field(min_length=1, max_length=32000)]


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def embedding_identity() -> dict:
    # Changing any of these invalidates corpus embeddings; never silently reuse an old index.
    spec = dict(model=EMBED_MODEL, revision=EMBED_REVISION, dimensions=EMBED_DIM,
                dtype="bfloat16", normalization="l2", min_pixels=64 * 28 * 28,
                max_pixels=IMAGE_MAX_PIXELS, sentence_transformers="6.1.0",
                transformers="5.2.0", torch="2.11.0", torchvision="0.26.0", document_prompt="model-default", query_prompt=QUERY_PROMPT)
    return {**spec, "fingerprint": digest(spec)}


def decode_image(value: str) -> Image.Image:
    # Data URLs only: no remote fetches, local paths, SVG, or user-controlled filesystem access.
    if not isinstance(value, str):
        raise ValueError("image must be a data URL string")
    match = re.fullmatch(r"data:image/(png|jpeg|webp);base64,([A-Za-z0-9+/=]+)", value)
    if not match or len(value) > MAX_IMAGE_BYTES * 4 // 3 + 100:
        raise ValueError("image must be a PNG/JPEG/WebP data URL, at most 8 MiB decoded")
    try:
        raw = base64.b64decode(match[2], validate=True)
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError("image exceeds 8 MiB")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as im:
                if im.format not in {"PNG", "JPEG", "WEBP"} or im.width * im.height > MAX_IMAGE_PIXELS:
                    raise ValueError("unsupported image or more than 8 million pixels")
                if getattr(im, "n_frames", 1) != 1:
                    raise ValueError("animated images are not supported")
                im.load()
                return im.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError("invalid or oversized image") from exc


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Media(Contract):
    text: Text | None = None
    image: str | None = Field(default=None, max_length=MAX_IMAGE_BYTES * 4 // 3 + 100)

    @field_validator("text")
    @classmethod
    def clean_text(cls, value):
        if value is not None:
            value = value.strip()
            if not value:
                raise ValueError("text must not be blank")
        return value

    @field_validator("image")
    @classmethod
    def check_image(cls, value):
        if value is not None:
            decode_image(value)
        return value

    @model_validator(mode="after")
    def has_input(self):
        if self.text is None and self.image is None:
            raise ValueError("provide text, image, or both")
        return self

    def model_input(self):
        # Explicit typed dictionaries prevent strings resembling URLs being auto-fetched by ST.
        result = {"text": self.text} if self.text is not None else {}
        if self.image:
            result["image"] = decode_image(self.image)
        return result


class EmbedRequest(Contract):
    inputs: list[Media] = Field(min_length=1, max_length=32)
    input_type: Literal["query", "document"] = "query"
    instruction: str | None = Field(default=None, max_length=1000)


class RerankRequest(Contract):
    query: Media
    documents: list[Media] = Field(min_length=1, max_length=32)
    instruction: str = Field(default=RERANK_PROMPT, max_length=1000)
    top_k: int = Field(default=20, ge=1, le=32)


class Document(Contract):
    id: str = Field(min_length=1, max_length=128)
    content: Text
    title: str = Field(default="", max_length=1000)
    image: str | None = None
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict, max_length=32)

    @field_validator("image")
    @classmethod
    def check_image(cls, value):
        if value:
            decode_image(value)
        return value


class IndexCreate(Contract):
    index_id: IndexID
    expected_documents: int = Field(ge=1, le=100_000)
    transform: Literal["none", "recipe_v1"] = "recipe_v1"
    source: str = Field(default="custom", max_length=1000)
    query_instruction: str = Field(default=QUERY_PROMPT, min_length=1, max_length=1000)


class IngestRequest(Contract):
    documents: list[Document] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def unique_ids(self):
        if len({doc.id for doc in self.documents}) != len(self.documents):
            raise ValueError("document IDs must be unique within a batch")
        return self


class SearchRequest(Contract):
    index_id: IndexID = "recipes-v1"
    query: Media
    mode: Literal["dense", "lexical", "hybrid"] = "hybrid"
    lexical_text: str | None = Field(default=None, max_length=4000)
    top_k: int = Field(default=20, ge=1, le=100)
    candidates: int = Field(default=100, ge=1, le=500)
    bm25_min_coverage: float = Field(default=0.8, ge=0, le=1)
    exact: bool = True
    include_images: bool = False

    @model_validator(mode="after")
    def enough_candidates(self):
        if self.candidates < self.top_k:
            raise ValueError("candidates must be >= top_k")
        return self


class RagRequest(Contract):
    index_id: IndexID = "recipes-v1"
    query: Media
    strategy: Literal["direct", "agent"] = "direct"
    top_k: int = Field(default=3, ge=1, le=10)
    rerank: bool = True
    rerank_images: bool = False
    answer: bool = True
    max_tokens: int = Field(default=512, ge=1, le=2048)


class Message(Contract):
    role: Literal["system", "user", "assistant"]
    content: str | list[dict] = Field(max_length=32000)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value):
        if isinstance(value, list):
            if len(value) > 16:
                raise ValueError("too many content parts")
            for part in value:
                if part.get("type") == "text" and set(part) == {"type", "text"}:
                    if not isinstance(part["text"], str) or len(part["text"]) > 32000:
                        raise ValueError("invalid text part")
                elif part.get("type") == "image_url" and set(part) == {"type", "image_url"}:
                    if not isinstance(part["image_url"], dict) or set(part["image_url"]) != {"url"}:
                        raise ValueError("image_url requires only url")
                    decode_image(part["image_url"]["url"])
                else:
                    raise ValueError("only text and image_url data-URL parts are supported")
        return value


class ChatRequest(Contract):
    model: Literal["Qwen/Qwen3.5-4B"] = GEN_MODEL
    messages: list[Message] = Field(min_length=1, max_length=16)
    max_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.8, gt=0, le=1)
    top_k: int = Field(default=20, ge=1, le=100)
    stream: Literal[False] = False  # Durable progress SSE is available via jobs; no fake token streaming.
    tools: list[dict] | None = Field(default=None, max_length=3)
    tool_choice: Literal["auto", "none"] = "none"

    @model_validator(mode="after")
    def budget(self):
        if len(json.dumps([m.model_dump() for m in self.messages])) > 14_000_000:
            raise ValueError("message envelope too large")
        images = sum(1 for m in self.messages if isinstance(m.content, list)
                     for p in m.content if p.get("type") == "image_url")
        if images > 1:
            raise ValueError("this deployment allows one image per generation request")
        return self
