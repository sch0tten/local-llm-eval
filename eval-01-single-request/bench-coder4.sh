#!/bin/bash
# Benchmark Qwen3-Coder-30B-A3B AWQ-4bit on inference server
# Tests: TP=2 NVLink, TP=1 single GPU, TP=2 PCIe-only
set -euo pipefail

SERVER="${VLLM_SSH:-user@your-server}"
VLLM_HOST="${VLLM_HOST:-localhost}"
API="http://${VLLM_HOST}:8000/v1/chat/completions"
MODEL_CMD="${VLLM_MODEL_CMD:-/path/to/model-serve.sh}"

wait_for_api() {
    echo -n "  Waiting for API"
    for i in $(seq 1 90); do
        sleep 5
        if curl -s "http://${VLLM_HOST}:8000/v1/models" >/dev/null 2>&1; then
            echo " ready!"
            # warmup request
            curl -s "$API" -d '{"model":"cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit","messages":[{"role":"user","content":"hi"}],"max_tokens":10}' >/dev/null 2>&1
            sleep 2
            return 0
        fi
        echo -n "."
    done
    echo " TIMEOUT"
    return 1
}

run_bench() {
    local label="$1"
    echo ""
    echo "============================================================"
    echo "BENCHMARK: $label"
    echo "============================================================"
    python3 bench-all.py vllm
}

stop_server() {
    ssh "$SERVER" "sudo systemctl stop qwen-llm qwen-coder qwen-coder4 2>/dev/null; pkill -f 'vllm serve' 2>/dev/null || true; sleep 3"
}

# --- Test 1: TP=2 with NVLink ---
echo ">>> TEST 1: TP=2 NVLink (default)"
stop_server
ssh "$SERVER" "nohup $MODEL_CMD coder4 --tp 2 > /tmp/vllm-bench.log 2>&1 &"
wait_for_api && run_bench "TP=2 NVLink"

# --- Test 2: TP=1 single GPU ---
echo ""
echo ">>> TEST 2: TP=1 Single GPU"
stop_server
ssh "$SERVER" "nohup $MODEL_CMD coder4 --tp 1 > /tmp/vllm-bench.log 2>&1 &"
wait_for_api && run_bench "TP=1 Single GPU"

# --- Test 3: TP=2 PCIe only (NVLink disabled) ---
echo ""
echo ">>> TEST 3: TP=2 PCIe (NVLink disabled)"
stop_server
ssh "$SERVER" "nohup $MODEL_CMD coder4 --tp 2 --no-nvlink > /tmp/vllm-bench.log 2>&1 &"
wait_for_api && run_bench "TP=2 PCIe (no NVLink)"

# Cleanup
echo ""
echo ">>> Stopping server..."
stop_server
echo "Done! All benchmarks complete."
