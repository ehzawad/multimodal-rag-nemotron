"""Experiment: agentic retrieval with tool calling.

Qwen3.5-4B is given a `search_recipes` tool and decides which searches to run: it rewrites the user's request into
concise English keyword queries (translating non-English requests, expanding vague ones, splitting multi-part ones).
Every tool call runs the hybrid search; the rankings are fused with the original query's ranking (RRF) and the top
results are reranked with the agent's first English query. The agent then answers from the tool results.

Optional reasoning: `think=True` lets the model think before calling tools (slower, sometimes better plans).
"""
from __future__ import annotations

import json
import re
import time

import torch
from PIL import Image

from . import pipeline as P

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_recipes",
        "description": "Search a database of ~10,000 English recipes (from Food.com) by dish name, ingredients, "
                       "cuisine, cooking method, course or occasion. Returns the best matching recipes.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string",
                                     "description": "A short ENGLISH keyword search query, e.g. 'banana bread sour cream'"}},
            "required": ["query"],
        },
    },
}

SYSTEM_PROMPT = """You are a recipe search assistant. The recipe database is in English.
For every user request, call the search_recipes tool 1 to 3 times with short English keyword queries:
- If the request is not in English, translate it to English first.
- Use the most specific dish name or key ingredients you can infer; add an alternative phrasing (synonyms, dish
  type) as a second query when the request is vague.
- Call all the searches you need in a single reply."""

_TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
_PARAM = re.compile(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.S)


def parse_tool_calls(text: str) -> list[dict]:
    """Parse Qwen3.5's XML tool calls (falls back to JSON tool calls)."""
    calls = []
    for block in _TOOL_CALL.findall(text):
        function = re.search(r"<function=([\w.-]+)>", block)
        if function:
            calls.append({"name": function.group(1), "arguments": dict(_PARAM.findall(block))})
            continue
        try:
            payload = json.loads(block)
            calls.append({"name": payload.get("name"), "arguments": payload.get("arguments", {})})
        except json.JSONDecodeError:
            pass
    return calls


class AgenticRAG:
    def __init__(self, rag: P.MultimodalRAG, think: bool = False, max_searches: int = 3):
        self.rag = rag
        self.think = think
        self.max_searches = max_searches

    @torch.inference_mode()
    def _chat(self, messages: list[dict], max_new_tokens: int, tools: list[dict] | None = None) -> str:
        processor, model = self.rag.generation_processor, self.rag.generation_model
        inputs = processor.apply_chat_template(messages, tools=tools, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt", enable_thinking=self.think)
        inputs = inputs.to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return processor.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def plan_searches(self, query_text: str | None, query_image: Image.Image | None = None) -> tuple[list[str], str]:
        content = [{"type": "image", "image": query_image.convert("RGB")}] if query_image is not None else []
        content.append({"type": "text", "text": query_text or "Find recipes for the dish or ingredients in this image."})
        messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user", "content": content}]
        raw = self._chat(messages, max_new_tokens=1024 if self.think else 160, tools=[SEARCH_TOOL])
        searches = [call["arguments"].get("query", "").strip() for call in parse_tool_calls(raw)
                    if call["name"] == "search_recipes"]
        return [s for s in searches if s][:self.max_searches], raw

    def search(self, query_text: str | None = None, query_image: Image.Image | None = None, k: int = 20,
               candidates: int = 100, rerank_top_n: int = 0, rerank_images: bool = False) -> P.RetrievalResult:
        start = time.time()
        searches, _ = self.plan_searches(query_text, query_image)
        timings = {"plan_s": round(time.time() - start, 3), "searches": searches}

        start = time.time()
        rankings = []
        if query_text or query_image is not None:  # keep the user's original query in the fusion
            rankings.append(self.rag.search(query_text=query_text, query_image=query_image, k=candidates,
                                            candidates=candidates, rewrite_image_query=False).indices)
        for search in searches:
            rankings.append(self.rag.search(query_text=search, k=candidates, candidates=candidates).indices)
        fused = P.reciprocal_rank_fusion(rankings)[:max(k, rerank_top_n)]
        timings["search_s"] = round(time.time() - start, 3)
        result = P.RetrievalResult(indices=[d for d, _ in fused], scores=[s for _, s in fused],
                                   query_text=searches[0] if searches else query_text, timings=timings)
        if rerank_top_n and result.query_text:
            result = self.rag.rerank(result.query_text, result, top_n=rerank_top_n, use_images=rerank_images)
        return result

    def answer(self, query_text: str | None, result: P.RetrievalResult, query_image: Image.Image | None = None,
               num_recipes: int = 3, max_new_tokens: int = 512) -> str:
        """Answer in the user's language, grounded in the retrieved recipes (sent back as tool results)."""
        tool_result = "\n\n".join(f"[{n}] {self.rag.texts[i].split(chr(10) + '## Reviews')[0][:4000]}"
                                  for n, i in enumerate(result.indices[:num_recipes], 1))
        content = [{"type": "image", "image": query_image.convert("RGB")}] if query_image is not None else []
        content.append({"type": "text", "text": query_text or "Find recipes for the dish or ingredients in this image."})
        searches = result.timings.get("searches") or [result.query_text or ""]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT + "\nAnswer in the user's language "
                                            "using only the search results; cite recipes as [1], [2], [3]."}]},
            {"role": "user", "content": content},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "search_recipes", "arguments": {"query": s}}} for s in searches]},
            {"role": "tool", "content": tool_result},
        ]
        text = self._chat(messages, max_new_tokens=max_new_tokens + (1024 if self.think else 0), tools=[SEARCH_TOOL])
        return text.split("</think>")[-1].strip()
