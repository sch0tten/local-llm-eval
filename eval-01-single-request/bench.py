#!/usr/bin/env python3
"""Quick tok/s benchmark for vLLM inference server."""

import json, time, os, requests

VLLM_HOST = os.environ.get("VLLM_HOST", "localhost")
API = f"http://{VLLM_HOST}:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3.5-27B-FP8"

SYSTEM = "You are a helpful assistant. Answer directly without internal reasoning or thinking steps."

PROMPTS = [
    "Write a short story about a robot learning to cook.",
    "Explain quantum entanglement to a 10-year-old.",
    "List 20 creative names for a cat cafe and explain each one.",
]

total_tok = 0
total_time = 0

for i, prompt in enumerate(PROMPTS, 1):
    print(f"--- Prompt {i}: {prompt[:60]}...")
    t0 = time.time()
    r = requests.post(API, json={
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 512,
        "temperature": 0.7,
    })
    elapsed = time.time() - t0
    u = r.json()["usage"]
    ptok, ctok = u["prompt_tokens"], u["completion_tokens"]
    total_tok += ctok
    total_time += elapsed
    print(f"  prompt: {ptok} tok  |  generated: {ctok} tok in {elapsed:.1f}s")
    print(f"  throughput: {ctok/elapsed:.1f} tok/s")
    print()

print(f"=== Overall: {total_tok} tokens in {total_time:.1f}s = {total_tok/total_time:.1f} tok/s ===")
