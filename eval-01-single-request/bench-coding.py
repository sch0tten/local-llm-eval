#!/usr/bin/env python3
"""Coding-specific benchmark: MoE vs Dense across 3 GPU configs.
5 real coding prompts, AWQ-4bit only (hardware-accelerated on SM 8.6).
Runs from workstation, controls inference server via SSH."""

import json, time, sys, subprocess, requests, datetime, re, os

SERVER = os.environ.get("VLLM_SSH", "user@your-server")
VLLM_HOST = os.environ.get("VLLM_HOST", "localhost")
API = f"http://{VLLM_HOST}:8000/v1/chat/completions"
MODELS_API = f"http://{VLLM_HOST}:8000/v1/models"
MODEL_CMD = os.environ.get("VLLM_MODEL_CMD", "/path/to/model-serve.sh")

SYSTEM = "You are a helpful assistant. Answer directly without internal reasoning or thinking steps."

PROMPTS = [
    ("impl", "Write a Python function that implements an LRU cache from scratch (no functools). "
     "It should support get(key) and put(key, value) with O(1) time complexity for both operations. "
     "Use a doubly linked list and a dictionary. Include type hints and a docstring."),

    ("debug", '''Find the bug in this Python code, explain why it fails, and write the corrected version:

```python
def merge_sorted_lists(lists: list[list[int]]) -> list[int]:
    import heapq
    heap = []
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))

    result = []
    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        next_idx = elem_idx + 1
        if next_idx < len(lists[list_idx]):
            heapq.heappush(heap, (lists[list_idx][next_idx], list_idx, elem_idx))
    return result
```'''),

    ("test", "Write a comprehensive pytest test suite for a function with this signature: "
     "`def parse_cron(expr: str) -> dict[str, list[int]]` that parses a standard 5-field cron expression "
     "(minute, hour, day-of-month, month, day-of-week) and returns expanded integer lists for each field. "
     "Cover: valid expressions, wildcards, ranges, steps, comma-separated values, edge cases, and invalid input. "
     "Use parametrize where appropriate. Write at least 12 test cases."),

    ("refactor", "Refactor this Flask endpoint into a clean service-layer architecture with proper separation of concerns:\n\n"
     "```python\n"
     "@app.route('/api/orders', methods=['POST'])\n"
     "def create_order():\n"
     "    data = request.json\n"
     "    user = db.session.query(User).filter_by(id=data['user_id']).first()\n"
     "    if not user:\n"
     "        return jsonify({'error': 'User not found'}), 404\n"
     "    if user.balance < data['total']:\n"
     "        return jsonify({'error': 'Insufficient balance'}), 400\n"
     "    order = Order(user_id=user.id, total=data['total'], items=json.dumps(data['items']))\n"
     "    user.balance -= data['total']\n"
     "    db.session.add(order)\n"
     "    db.session.commit()\n"
     "    send_email(user.email, 'Order confirmed', f'Order {order.id} for ${order.total}')\n"
     "    return jsonify({'order_id': order.id}), 201\n"
     "```\n\n"
     "Show the refactored code split into: route handler, service layer, and repository layer. "
     "Include proper error handling and type hints."),

    ("design", "Design and implement a Python rate limiter class that supports:\n"
     "1. Token bucket algorithm with configurable rate and burst size\n"
     "2. Per-key rate limiting (e.g., per API client)\n"
     "3. Thread-safe operation\n"
     "4. A decorator for Flask/FastAPI routes\n\n"
     "Provide the complete implementation with type hints, docstrings, and a usage example."),
]

# 2 models x 3 configs = 6 runs
CONFIGS = [
    # --- MoE AWQ-4bit ---
    ("coder4", "TP=2 NVLink",     "--tp 2",               None),
    ("coder4", "TP=1 Single GPU", "--tp 1",               "32768"),
    ("coder4", "TP=2 PCIe",       "--tp 2 --no-nvlink",   None),
    # --- Dense AWQ-4bit ---
    ("dense",  "TP=2 NVLink",     "--tp 2",               None),
    ("dense",  "TP=1 Single GPU", "--tp 1",               "8192"),
    ("dense",  "TP=2 PCIe",       "--tp 2 --no-nvlink",   None),
]

MODEL_NAMES = {
    "coder4": "Qwen3-Coder-30B-A3B AWQ-4bit (MoE, 3.3B active)",
    "dense":  "Qwen2.5-Coder-32B-Instruct AWQ (Dense, 32B active)",
}

MODEL_SHORT = {
    "coder4": "MoE",
    "dense":  "Dense",
}


def ssh(cmd, timeout=30, retries=3):
    for attempt in range(retries):
        try:
            r = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=5", SERVER, cmd],
                capture_output=True, text=True, timeout=timeout
            )
            return r.stdout.strip()
        except (subprocess.TimeoutExpired, Exception) as e:
            if attempt < retries - 1:
                print(f"    SSH retry {attempt+1}/{retries}...")
                time.sleep(10)
            else:
                print(f"    SSH failed after {retries} attempts: {e}")
                return ""


def gpu_reset():
    """PCIe reset both GPUs and reload nvidia modules."""
    print("    PCIe reset + module reload...")
    ssh("sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null || true", timeout=20)
    time.sleep(2)
    ssh("echo 1 | sudo tee /sys/bus/pci/devices/0000:21:00.0/reset; "
        "echo 1 | sudo tee /sys/bus/pci/devices/0000:4a:00.0/reset", timeout=15)
    time.sleep(3)
    ssh("sudo modprobe nvidia nvidia_uvm nvidia_modeset nvidia_drm", timeout=20)
    time.sleep(5)


def stop_server():
    print("  Stopping server...")
    # Stop via systemd-run unit name (clean shutdown with proper signal handling)
    ssh("sudo systemctl stop vllm-bench.service 2>/dev/null; "
        "sudo systemctl stop qwen-llm qwen-coder qwen-coder4 2>/dev/null; "
        "sleep 2", timeout=30)

    # Wait for GPU memory to be released
    for attempt in range(20):
        mem = ssh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 'ERR'", timeout=15)
        if mem and "ERR" not in mem:
            used = [int(x.strip()) for x in mem.split("\n") if x.strip().isdigit()]
            if all(u < 100 for u in used):
                return
        time.sleep(5)
    print("    WARNING: GPUs may not be fully free — attempting PCIe reset")
    gpu_reset()


def start_server(model_key, args, ctx_override):
    ctx_flag = f" --ctx {ctx_override}" if ctx_override else ""
    # Use systemd-run for proper process management:
    # - Clean file descriptors (no nohup fd inheritance issues)
    # - Proper cgroup isolation
    # - Clean SIGTERM shutdown via systemctl stop
    # Dense models with --no-nvlink need full socket transport
    # (NCCL P2P disable alone causes engine init failure for dense)
    nccl_extra = ""
    if model_key == "dense" and "--no-nvlink" in args:
        nccl_extra = ("--setenv=NCCL_SHM_DISABLE=1 "
                      "--setenv=NCCL_NET=Socket ")

    run_user = os.environ.get("VLLM_USER", "nobody")
    hf_home = os.environ.get("VLLM_HF_HOME", "/tmp/hf-cache")
    hf_hub = os.environ.get("VLLM_HF_HUB", "/tmp/hf-models")
    cmd = (f"sudo systemd-run --unit=vllm-bench --remain-after-exit "
           f"--setenv=HF_HOME={hf_home} "
           f"--setenv=HUGGINGFACE_HUB_CACHE={hf_hub} "
           f"--setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
           f"--setenv=OMP_NUM_THREADS=8 "
           f"{nccl_extra}"
           f"--uid={run_user} "
           f"-- {MODEL_CMD} {model_key} {args}{ctx_flag}")
    print(f"  Starting: model-serve.sh {model_key} {args}{ctx_flag}")
    print(f"  (via systemd-run)")
    ssh(cmd, timeout=15)


def wait_for_api(timeout_secs=480):
    print("  Waiting for API", end="", flush=True)
    for _ in range(timeout_secs // 5):
        time.sleep(5)
        try:
            r = requests.get(MODELS_API, timeout=5)
            if r.status_code == 200:
                model = r.json()["data"][0]["id"]
                print(f" ready! ({model})")
                # warmup request
                requests.post(API, json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Write a hello world in Python."}],
                    "max_tokens": 50,
                }, timeout=30)
                time.sleep(2)
                return model
        except Exception:
            pass
        print(".", end="", flush=True)
    print(" TIMEOUT")
    return None


def get_gpu_mem():
    try:
        out = ssh("nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits")
        lines = out.strip().split("\n")
        gpus = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            gpus.append({"index": int(parts[0]), "used_mb": int(parts[1]), "total_mb": int(parts[2])})
        return gpus
    except Exception:
        return []


def run_bench(run_label="measurement"):
    """Run the 5 coding prompts."""
    try:
        r = requests.get(MODELS_API, timeout=5)
        model = r.json()["data"][0]["id"]
    except Exception:
        print("  ERROR: API not available")
        return None

    results = []
    for label, prompt in PROMPTS:
        t0 = time.time()
        try:
            r = requests.post(API, json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 4096,
                "temperature": 0.7,
            }, timeout=600)
        except requests.exceptions.Timeout:
            elapsed = time.time() - t0
            print(f"    [{label:8s}] TIMEOUT after {elapsed:.0f}s")
            results.append({"label": label, "prompt_tokens": 0, "completion_tokens": 0,
                            "elapsed": elapsed, "tps": 0, "finish": "timeout", "truncated": False})
            continue
        elapsed = time.time() - t0
        data = r.json()
        u = data["usage"]
        ptok, ctok = u["prompt_tokens"], u["completion_tokens"]
        tps = ctok / elapsed if elapsed > 0 else 0
        truncated = ctok >= 4090  # near max_tokens = likely cut off
        finish = data["choices"][0].get("finish_reason", "?")
        results.append({"label": label, "prompt_tokens": ptok, "completion_tokens": ctok,
                        "elapsed": elapsed, "tps": tps, "finish": finish, "truncated": truncated})
        if run_label == "measurement":
            flag = " TRUNCATED" if truncated else ""
            print(f"    [{label:8s}] {ctok:4d} tok in {elapsed:5.1f}s = {tps:5.1f} tok/s  [{finish}]{flag}")

    if run_label == "warmup":
        total_tok = sum(r["completion_tokens"] for r in results)
        total_time = sum(r["elapsed"] for r in results)
        avg = total_tok / total_time if total_time > 0 else 0
        print(f"    warmup done (avg {avg:.0f} tok/s)")

    return results


def main():
    start_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    all_results = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"{'='*70}")
    print(f"CODING BENCHMARK — {timestamp}")
    print(f"2 models x 3 GPU configs = 6 configurations")
    print(f"5 coding prompts, max_tokens=4096, temperature=0.7")
    print(f"{'='*70}")

    total = len(CONFIGS)
    for i, (model_key, config_label, args, ctx_override) in enumerate(CONFIGS):
        if i < start_idx:
            print(f"\n  Skipping [{i+1}/{total}] {MODEL_NAMES[model_key]} — {config_label}")
            continue
        print(f"\n{'='*70}")
        print(f"[{i+1}/{total}] {MODEL_NAMES[model_key]} — {config_label}")
        print(f"{'='*70}")

        stop_server()
        start_server(model_key, args, ctx_override)
        model_id = wait_for_api()

        if not model_id:
            print("  SKIPPED (server failed to start)")
            all_results[(model_key, config_label)] = None
            continue

        gpu_mem = get_gpu_mem()
        if gpu_mem:
            for g in gpu_mem:
                print(f"    GPU {g['index']}: {g['used_mb']}MB / {g['total_mb']}MB")

        # Run 1: warmup
        print("  Run 1 (warmup):")
        run_bench("warmup")

        # Run 2: measurement
        print("  Run 2 (measurement):")
        results = run_bench("measurement")

        if results:
            total_tok = sum(r["completion_tokens"] for r in results)
            total_time = sum(r["elapsed"] for r in results)
            avg_tps = total_tok / total_time if total_time > 0 else 0
            print(f"    ---- Average: {avg_tps:.1f} tok/s ({total_tok} tok in {total_time:.1f}s)")

            all_results[(model_key, config_label)] = {
                "results": results,
                "gpu_mem": gpu_mem,
                "avg_tps": avg_tps,
                "total_tokens": total_tok,
                "total_time": total_time,
                "per_prompt": {r["label"]: round(r["tps"], 1) for r in results},
            }
        else:
            all_results[(model_key, config_label)] = None

    # Cleanup
    stop_server()

    # Print summary tables
    print(f"\n\n{'='*70}")
    print("RESULTS — CODING BENCHMARK")
    print(f"{'='*70}")

    prompt_labels = [p[0] for p in PROMPTS]
    header = f"{'Configuration':<22s}" + "".join(f" {l:>8s}" for l in prompt_labels) + f" {'Average':>9s}"

    for model_key in ["coder4", "dense"]:
        print(f"\n### {MODEL_NAMES[model_key]}")
        print(header)
        print("-" * len(header))
        for config_label in ["TP=2 NVLink", "TP=1 Single GPU", "TP=2 PCIe"]:
            data = all_results.get((model_key, config_label))
            if data is None:
                print(f"{config_label:<22s} {'SKIPPED':>8s}")
                continue
            pp = data["per_prompt"]
            cols = "".join(f" {pp.get(l, 0):>8.1f}" for l in prompt_labels)
            print(f"{config_label:<22s}{cols} {data['avg_tps']:>8.1f}")

    # Cross-model comparison
    print(f"\n### MoE vs Dense — Head to Head")
    print(f"{'Configuration':<22s} {'MoE':>9s} {'Dense':>9s} {'Speedup':>9s}")
    print("-" * 52)
    for config_label in ["TP=2 NVLink", "TP=1 Single GPU", "TP=2 PCIe"]:
        moe = all_results.get(("coder4", config_label))
        dense = all_results.get(("dense", config_label))
        moe_tps = f"{moe['avg_tps']:.0f}" if moe else "-"
        dense_tps = f"{dense['avg_tps']:.0f}" if dense else "-"
        if moe and dense and dense["avg_tps"] > 0:
            ratio = f"{moe['avg_tps'] / dense['avg_tps']:.1f}x"
        else:
            ratio = "-"
        print(f"{config_label:<22s} {moe_tps:>9s} {dense_tps:>9s} {ratio:>9s}")

    # NVLink impact
    print(f"\n### NVLink Impact")
    for model_key in ["coder4", "dense"]:
        nvlink = all_results.get((model_key, "TP=2 NVLink"))
        pcie = all_results.get((model_key, "TP=2 PCIe"))
        if nvlink and pcie and pcie["avg_tps"] > 0:
            boost = ((nvlink["avg_tps"] / pcie["avg_tps"]) - 1) * 100
            print(f"  {MODEL_SHORT[model_key]:>6s}: NVLink gives {boost:+.1f}% over PCIe")

    # Second GPU impact
    print(f"\n### Second GPU Impact")
    for model_key in ["coder4", "dense"]:
        tp2 = all_results.get((model_key, "TP=2 NVLink"))
        tp1 = all_results.get((model_key, "TP=1 Single GPU"))
        if tp2 and tp1 and tp1["avg_tps"] > 0:
            boost = ((tp2["avg_tps"] / tp1["avg_tps"]) - 1) * 100
            print(f"  {MODEL_SHORT[model_key]:>6s}: Second GPU gives {boost:+.1f}%")

    # Truncation report
    truncations = []
    for (mk, cl), v in all_results.items():
        if v:
            for r in v["results"]:
                if r.get("truncated"):
                    truncations.append(f"  {MODEL_SHORT[mk]} / {cl} / {r['label']}: {r['completion_tokens']} tok ({r['finish']})")
    if truncations:
        print(f"\n### Truncated Responses (hit max_tokens=4096)")
        for t in truncations:
            print(t)
    else:
        print(f"\n### All responses completed naturally (none hit max_tokens)")

    # Save JSON
    json_data = {
        "timestamp": timestamp,
        "prompts": [(l, p[:80] + "...") for l, p in PROMPTS],
        "max_tokens": 4096,
        "system_prompt": SYSTEM,
        "results": {},
    }
    for (mk, cl), v in all_results.items():
        if v:
            json_data["results"][f"{mk}|{cl}"] = v
    with open("bench-coding-results.json", "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"\nResults saved to bench-coding-results.json")


if __name__ == "__main__":
    main()
