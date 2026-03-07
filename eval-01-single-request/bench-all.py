#!/usr/bin/env python3
"""Benchmark multiple Qwen models via vLLM.
Run each model's vLLM server first, then run this script."""

import json, time, sys, os, requests

VLLM_HOST = os.environ.get("VLLM_HOST", "localhost")
VLLM_API = f"http://{VLLM_HOST}:8000/v1/chat/completions"
OLLAMA_API = f"http://{VLLM_HOST}:11434/api/chat"

SYSTEM = "You are a helpful assistant. Answer directly without internal reasoning or thinking steps."

PROMPTS = [
    ("short", "What is the capital of France?"),
    ("medium", "Explain how a combustion engine works in 3 paragraphs."),
    ("long", "Write a detailed comparison of Python, Rust, and Go for building web services. Cover performance, ecosystem, learning curve, and concurrency."),
    ("code", "Write a Python function that finds the longest palindromic substring in a string. Include docstring and type hints."),
]

def bench_vllm(model_name=None):
    try:
        r = requests.get(f"http://{VLLM_HOST}:8000/v1/models", timeout=5)
        model = r.json()["data"][0]["id"]
    except Exception:
        print("  vLLM not running on :8000")
        return None

    print(f"\n{'='*60}")
    print(f"MODEL: {model}")
    print(f"ENGINE: vLLM")
    print(f"{'='*60}")

    results = []
    for label, prompt in PROMPTS:
        t0 = time.time()
        r = requests.post(VLLM_API, json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 512,
            "temperature": 0.7,
        }, timeout=120)
        elapsed = time.time() - t0
        u = r.json()["usage"]
        ptok, ctok = u["prompt_tokens"], u["completion_tokens"]
        tps = ctok / elapsed if elapsed > 0 else 0
        results.append((label, ptok, ctok, elapsed, tps))
        print(f"  [{label:6s}] {ctok:4d} tok in {elapsed:5.1f}s = {tps:5.1f} tok/s")

    total_tok = sum(r[2] for r in results)
    total_time = sum(r[3] for r in results)
    avg_tps = total_tok / total_time if total_time > 0 else 0
    print(f"  {'':8s} ---- avg: {avg_tps:.1f} tok/s ({total_tok} tok in {total_time:.1f}s)")
    return {"model": model, "engine": "vllm", "avg_tps": avg_tps, "results": results}

def bench_ollama(model_name):
    try:
        requests.get(f"http://{VLLM_HOST}:11434/api/tags", timeout=5)
    except Exception:
        print("  Ollama not running on :11434")
        return None

    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"ENGINE: Ollama")
    print(f"{'='*60}")

    results = []
    for label, prompt in PROMPTS:
        t0 = time.time()
        r = requests.post(OLLAMA_API, json={
            "model": model_name,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"num_predict": 512, "temperature": 0.7},
        }, timeout=300)
        elapsed = time.time() - t0
        data = r.json()
        ctok = data.get("eval_count", 0)
        ptok = data.get("prompt_eval_count", 0)
        tps = ctok / elapsed if elapsed > 0 else 0
        results.append((label, ptok, ctok, elapsed, tps))
        print(f"  [{label:6s}] {ctok:4d} tok in {elapsed:5.1f}s = {tps:5.1f} tok/s")

    total_tok = sum(r[2] for r in results)
    total_time = sum(r[3] for r in results)
    avg_tps = total_tok / total_time if total_time > 0 else 0
    print(f"  {'':8s} ---- avg: {avg_tps:.1f} tok/s ({total_tok} tok in {total_time:.1f}s)")
    return {"model": model_name, "engine": "ollama", "avg_tps": avg_tps, "results": results}

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "vllm"

    if target == "vllm":
        bench_vllm()
    elif target == "ollama":
        model = sys.argv[2] if len(sys.argv) > 2 else "qwen3-coder-next:q4_K_M"
        bench_ollama(model)
    elif target == "all":
        print("Run each model separately:")
        print("  python3 bench-all.py vllm          # whatever is loaded in vLLM")
        print("  python3 bench-all.py ollama qwen3-coder-next:q4_K_M")
    else:
        print(f"Usage: {sys.argv[0]} [vllm|ollama <model>|all]")
