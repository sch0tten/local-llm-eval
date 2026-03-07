#!/usr/bin/env python3
"""Simple chatbot for vLLM inference server."""

import json
import os
import sys
import readline
import requests

VLLM_HOST = os.environ.get("VLLM_HOST", "localhost")
API_URL = f"http://{VLLM_HOST}:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3.5-27B-FP8"

def stream_chat(messages):
    resp = requests.post(API_URL, json={
        "model": MODEL,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.7,
        "stream": True,
    }, stream=True)
    resp.raise_for_status()

    full = ""
    for line in resp.iter_lines():
        line = line.decode()
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        delta = chunk["choices"][0]["delta"].get("content", "")
        if delta:
            print(delta, end="", flush=True)
            full += delta
    print()
    return full

def main():
    print(f"Qwen3.5-27B-FP8 @ {VLLM_HOST} | /clear to reset, Ctrl+C to quit")
    print()

    messages = []
    if len(sys.argv) > 1 and sys.argv[1] == "--system":
        system_msg = sys.argv[2] if len(sys.argv) > 2 else "You are a helpful assistant. Be concise."
        messages.append({"role": "system", "content": system_msg})

    while True:
        try:
            user = input("\033[1;32myou>\033[0m ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
            break
        if not user:
            continue
        if user == "/clear":
            messages = [m for m in messages if m["role"] == "system"]
            print("(cleared)")
            continue

        messages.append({"role": "user", "content": user})
        print("\033[1;34mqwen>\033[0m ", end="", flush=True)
        try:
            reply = stream_chat(messages)
            messages.append({"role": "assistant", "content": reply})
        except requests.exceptions.ConnectionError:
            print("(connection failed — is vLLM running?)")
            messages.pop()
        except Exception as e:
            print(f"(error: {e})")
            messages.pop()

if __name__ == "__main__":
    main()
