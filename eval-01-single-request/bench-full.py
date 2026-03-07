#!/usr/bin/env python3
"""Full 9-config benchmark: 3 models x 3 GPU configs.
Runs from workstation, controls inference server via SSH."""

import json, time, sys, subprocess, requests, datetime, os

SERVER = os.environ.get("VLLM_SSH", "user@your-server")
VLLM_HOST = os.environ.get("VLLM_HOST", "localhost")
API = f"http://{VLLM_HOST}:8000/v1/chat/completions"
MODELS_API = f"http://{VLLM_HOST}:8000/v1/models"
MODEL_CMD = os.environ.get("VLLM_MODEL_CMD", "/path/to/model-serve.sh")

SYSTEM = "You are a helpful assistant. Answer directly without internal reasoning or thinking steps."

PROMPTS = [
    ("short", "What is the capital of France?"),
    ("medium", "Explain how a combustion engine works in 3 paragraphs."),
    ("long", "Write a detailed comparison of Python, Rust, and Go for building web services, covering performance, developer experience, ecosystem maturity, and deployment complexity."),
    ("code", "Write a Python function that finds the longest palindromic substring in a given string. Include type hints, docstring, and handle edge cases. Then write 5 unit tests using pytest."),
]

# 3 models x 3 configs = 9 runs
# Each config: (model_key, label, ssh_args, ctx_override)
CONFIGS = [
    # --- coder4: MoE AWQ-4bit ---
    ("coder4", "TP=2 NVLink",     "--tp 2",               None),
    ("coder4", "TP=1 Single GPU", "--tp 1",               "32768"),
    ("coder4", "TP=2 PCIe",       "--tp 2 --no-nvlink",   None),
    # --- coder: MoE FP8 (30GB — TP=1 won't fit on 24GB GPU) ---
    ("coder",  "TP=2 NVLink",     "--tp 2",               None),
    ("coder",  "TP=2 PCIe",       "--tp 2 --no-nvlink",   None),
    # --- dense: Dense AWQ-4bit ---
    ("dense",  "TP=2 NVLink",     "--tp 2",               None),
    ("dense",  "TP=1 Single GPU", "--tp 1",               "8192"),
    ("dense",  "TP=2 PCIe",       "--tp 2 --no-nvlink",   None),
]

MODEL_NAMES = {
    "coder4": "Qwen3-Coder-30B-A3B AWQ-4bit",
    "coder":  "Qwen3-Coder-30B-A3B FP8",
    "dense":  "Qwen2.5-Coder-32B-Instruct AWQ",
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
    # Graceful SIGTERM first, wait for clean NVLink teardown
    ssh("sudo systemctl stop qwen-llm qwen-coder qwen-coder4 2>/dev/null; "
        "pkill -f 'vllm serve' 2>/dev/null || true", timeout=15)
    time.sleep(10)
    # Check if process died
    out = ssh("pgrep -f 'vllm serve' 2>/dev/null || echo 'clean'", timeout=10)
    if "clean" not in out:
        # Force kill if still alive
        ssh("pkill -9 -f 'vllm serve' 2>/dev/null || true", timeout=10)
        time.sleep(5)

    # Wait for GPU memory to be released
    for attempt in range(10):
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
    cmd = f"nohup {MODEL_CMD} {model_key} {args}{ctx_flag} > /tmp/vllm-bench.log 2>&1 &"
    print(f"  Starting: model-serve.sh {model_key} {args}{ctx_flag}")
    ssh(cmd, timeout=15)


def wait_for_api(timeout_secs=450):
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
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 10,
                }, timeout=30)
                time.sleep(2)
                return model
        except Exception:
            pass
        print(".", end="", flush=True)
    print(" TIMEOUT")
    return None


def get_gpu_mem():
    """Get GPU memory usage from nvidia-smi."""
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


def get_context_length():
    """Get max_model_len from vLLM server log."""
    try:
        out = ssh("grep -m1 'max_model_len' /tmp/vllm-bench.log 2>/dev/null || echo ''")
        if "max_model_len" in out:
            import re
            m = re.search(r'max_model_len[=: ]+(\d+)', out)
            if m:
                return int(m.group(1))
    except Exception:
        pass
    return None


def run_bench(run_label="measurement"):
    """Run the 4 prompts, return list of (label, ptok, ctok, elapsed, tps)."""
    try:
        r = requests.get(MODELS_API, timeout=5)
        model = r.json()["data"][0]["id"]
    except Exception:
        print("  ERROR: API not available")
        return None

    results = []
    for label, prompt in PROMPTS:
        t0 = time.time()
        r = requests.post(API, json={
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
        if run_label == "measurement":
            print(f"    [{label:6s}] {ctok:4d} tok in {elapsed:5.1f}s = {tps:5.1f} tok/s")

    if run_label == "warmup":
        avg = sum(r[2] for r in results) / sum(r[3] for r in results)
        print(f"    warmup done (avg {avg:.0f} tok/s)")

    return results


def main():
    all_results = {}  # (model_key, config_label) -> {results, gpu_mem, context, avg_tps}

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"{'='*70}")
    print(f"FULL BENCHMARK SUITE — {timestamp}")
    print(f"3 models x 3 GPU configs = 8 configurations (FP8 skips TP=1)")
    print(f"{'='*70}")

    total = len(CONFIGS)
    for i, (model_key, config_label, args, ctx_override) in enumerate(CONFIGS):
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

        # Get GPU memory at steady state
        gpu_mem = get_gpu_mem()
        if gpu_mem:
            for g in gpu_mem:
                print(f"    GPU {g['index']}: {g['used_mb']}MB / {g['total_mb']}MB")

        # Get max context from server log
        ctx = get_context_length()

        # Run 1: warmup
        print("  Run 1 (warmup):")
        run_bench("warmup")

        # Run 2: measurement
        print("  Run 2 (measurement):")
        results = run_bench("measurement")

        if results:
            total_tok = sum(r[2] for r in results)
            total_time = sum(r[3] for r in results)
            avg_tps = total_tok / total_time if total_time > 0 else 0
            print(f"    ---- Average: {avg_tps:.1f} tok/s ({total_tok} tok in {total_time:.1f}s)")

            all_results[(model_key, config_label)] = {
                "results": results,
                "gpu_mem": gpu_mem,
                "context": ctx,
                "avg_tps": avg_tps,
                "per_prompt": {r[0]: round(r[4], 1) for r in results},
            }
        else:
            all_results[(model_key, config_label)] = None

    # Cleanup
    stop_server()

    # Print summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for model_key in ["coder4", "coder", "dense"]:
        print(f"\n### {MODEL_NAMES[model_key]}")
        print(f"{'Configuration':<22s} {'Short':>6s} {'Medium':>7s} {'Long':>6s} {'Code':>6s} {'Average':>9s} {'Context':>9s}")
        print("-" * 70)
        for config_label in ["TP=2 NVLink", "TP=1 Single GPU", "TP=2 PCIe"]:
            data = all_results.get((model_key, config_label))
            if data is None:
                print(f"{config_label:<22s} {'SKIPPED':>6s}")
                continue
            pp = data["per_prompt"]
            ctx = f"{data['context']//1024}K" if data.get("context") else "?"
            print(f"{config_label:<22s} {pp.get('short',0):>6.0f} {pp.get('medium',0):>7.0f} {pp.get('long',0):>6.0f} {pp.get('code',0):>6.0f} {data['avg_tps']:>8.1f}  {ctx:>8s}")

    # Cross-model comparison
    print(f"\n### MoE vs Dense Comparison")
    print(f"{'Configuration':<22s} {'MoE AWQ':>9s} {'MoE FP8':>9s} {'Dense':>9s} {'MoE/Dense':>10s}")
    print("-" * 65)
    for config_label in ["TP=2 NVLink", "TP=1 Single GPU", "TP=2 PCIe"]:
        moe4 = all_results.get(("coder4", config_label))
        moefp8 = all_results.get(("coder", config_label))
        dense = all_results.get(("dense", config_label))

        moe4_tps = f"{moe4['avg_tps']:.0f}" if moe4 else "-"
        moefp8_tps = f"{moefp8['avg_tps']:.0f}" if moefp8 else "-"
        dense_tps = f"{dense['avg_tps']:.0f}" if dense else "-"

        if moe4 and dense and dense["avg_tps"] > 0:
            ratio = f"{moe4['avg_tps'] / dense['avg_tps']:.1f}x"
        else:
            ratio = "-"

        print(f"{config_label:<22s} {moe4_tps:>9s} {moefp8_tps:>9s} {dense_tps:>9s} {ratio:>10s}")

    # Save JSON results
    json_data = {
        "timestamp": timestamp,
        "results": {f"{k[0]}|{k[1]}": v for k, v in all_results.items() if v},
    }
    with open("bench-full-results.json", "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"\nResults saved to bench-full-results.json")


if __name__ == "__main__":
    main()
