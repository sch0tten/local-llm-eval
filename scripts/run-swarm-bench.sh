#!/bin/bash
# Swarm Benchmark Suite: 2 models × 3 GPU configs × 4 concurrency levels
#
# Architecture: workstation (this script) ──LAN──> inference server (inference server)
#
# Between each GPU config change, the inference server is REBOOTED for clean
# NVIDIA driver state. This eliminates driver corruption from model unloading
# as a confounding variable — faster and more reliable than PCIe reset.
#
# Dense model notes (vLLM 0.17 nightly):
#   - Requires --enforce-eager (CUDA graph capture crashes on V1 engine)
#   - TP=1 needs --ctx 12288 + gpu-memory-utilization 0.95 (18GB model on 24GB GPU)
#   - TP=2 PCIe BROKEN: NCCL_P2P_DISABLE=1 + enforce-eager causes SIGBUS during
#     weight loading. This is a vLLM nightly regression — no workaround found.
#
# Requires: VLLM_SSH, VLLM_HOST, VLLM_MODEL_CMD, VLLM_VENV environment variables
# Usage: source ../.env && ./run-swarm-bench.sh

set -euo pipefail

SERVER="${VLLM_SSH:?Set VLLM_SSH (e.g. user@your-server)}"
VLLM_HOST="${VLLM_HOST:?Set VLLM_HOST (e.g. 192.168.1.100)}"
MODEL_CMD="${VLLM_MODEL_CMD:?Set VLLM_MODEL_CMD (path to model-serve.sh on server)}"
VLLM_VENV="${VLLM_VENV:?Set VLLM_VENV (venv path on server, e.g. /opt/vllm/.venv)}"
RESULTS_DIR="${RESULTS_DIR:-./results}"

mkdir -p "$RESULTS_DIR"

reboot_server() {
    echo "  Rebooting inference server for clean GPU state..."
    ssh "$SERVER" "sudo reboot" 2>/dev/null || true
    sleep 10

    # Wait for SSH to come back (up to 2 minutes)
    echo -n "  Waiting for server"
    for i in $(seq 1 24); do
        sleep 5
        if ssh -o ConnectTimeout=3 -o BatchMode=yes "$SERVER" "true" 2>/dev/null; then
            echo " up!"
            sleep 5  # let services settle
            return 0
        fi
        echo -n "."
    done
    echo " TIMEOUT — server did not come back"
    exit 1
}

start_server() {
    local model_key="$1"
    local args="$2"
    local ctx_override="${3:-}"

    local ctx_flag=""
    [[ -n "$ctx_override" ]] && ctx_flag=" --ctx $ctx_override"

    echo "  Starting: model-serve.sh $model_key $args$ctx_flag"
    ssh "$SERVER" "sudo systemd-run --unit=vllm-bench --remain-after-exit \
        --setenv=HF_HOME=${VLLM_HF_HOME:-/tmp/hf-cache} \
        --setenv=HUGGINGFACE_HUB_CACHE=${VLLM_HF_HUB:-/tmp/hf-models} \
        --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        --setenv=OMP_NUM_THREADS=8 \
        --uid=${VLLM_USER:-nobody} \
        -- $MODEL_CMD $model_key $args$ctx_flag" 2>/dev/null
}

# Dense model requires --enforce-eager and explicit python -m vllm invocation
# (vllm binary shebang may point to wrong venv)
start_dense_server() {
    local tp="$1"
    local extra_args="${2:-}"
    local ctx="${3:-32768}"
    local gpu_util="${4:-0.92}"

    local cuda_env=""
    [[ "$tp" == "1" ]] && cuda_env="export CUDA_VISIBLE_DEVICES=0 && "

    local nccl_env=""
    [[ -n "$extra_args" && "$extra_args" == *"no-nvlink"* ]] && nccl_env="export NCCL_P2P_DISABLE=1 && "

    echo "  Starting Dense: TP=$tp ctx=$ctx gpu_util=$gpu_util enforce-eager"
    ssh "$SERVER" "sudo systemd-run --unit=vllm-bench --remain-after-exit \
        --uid=${VLLM_USER:-nobody} \
        -- bash -c '${cuda_env}${nccl_env}export HF_HOME=${VLLM_HF_HOME:-/tmp/hf-cache} && \
        export HUGGINGFACE_HUB_CACHE=${VLLM_HF_HUB:-/tmp/hf-models} && \
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
        export OMP_NUM_THREADS=8 && \
        ${VLLM_VENV}/bin/python3 -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-Coder-32B-Instruct-AWQ \
        --tensor-parallel-size $tp \
        --trust-remote-code --enable-chunked-prefill --max-num-seqs 4 \
        --disable-custom-all-reduce --host 0.0.0.0 --port 8000 \
        --gpu-memory-utilization $gpu_util --max-model-len $ctx \
        --dtype float16 --enforce-eager'" 2>/dev/null
}

wait_for_api() {
    echo -n "  Waiting for API"
    for i in $(seq 1 96); do  # 8 minutes
        sleep 5
        if curl -s "http://${VLLM_HOST}:8000/v1/models" >/dev/null 2>&1; then
            MODEL=$(curl -s "http://${VLLM_HOST}:8000/v1/models" | \
                python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || echo "unknown")
            echo " ready! ($MODEL)"
            return 0
        fi
        echo -n "."
    done
    echo " TIMEOUT"
    return 1
}

run_bench() {
    local model_name="$1"
    local config_label="$2"
    local output_csv="$3"

    echo ""
    echo "========================================================================"
    echo "SWARM BENCHMARK: $config_label"
    echo "  Model:       $model_name"
    echo "  Concurrency: 1 → 2 → 3 → 4 agents"
    echo "  Prompts:     16 unique (4 types × 4 variants)"
    echo "========================================================================"

    python3 "$(dirname "$0")/bench-swarm.py" \
        --base-url "http://${VLLM_HOST}:8000/v1" \
        --model "$model_name" \
        --max-tokens 8192 \
        --temperature 0.7 \
        --warmup 1 \
        --runs 2 \
        --concurrency 1 2 3 4 \
        --output-csv "$output_csv"
}

TOTAL=6
N=0
START_TIME=$(date +%s)

echo "========================================================================"
echo "SWARM BENCHMARK SUITE — $(date '+%Y-%m-%d %H:%M')"
echo "========================================================================"
echo "  Architecture: workstation ──LAN──> inference server"
echo "  Matrix:       2 models × 3 GPU configs × 4 concurrency levels (1 config broken)"
echo "  Prompts:      16 unique coding tasks (4 types × 4 variants)"
echo "  max_tokens:   8192 (single-module scope, targeting natural completion)"
echo "  Measurement:  warmup=1, runs=2 per concurrency level"
echo "  GPU reset:    full server reboot between each config change"
echo "========================================================================"

# ==========================================================================
# MoE: Qwen3-Coder-30B-A3B AWQ-4bit (3.3B active params per token)
# ==========================================================================

# --- Test 1: MoE TP=2 NVLink ---
N=$((N+1))
echo ""
echo ">>> TEST $N/$TOTAL: MoE — TP=2 NVLink"
reboot_server
start_server "coder4" "--tp 2"
wait_for_api
run_bench "cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit" "MoE TP=2 NVLink" \
    "$RESULTS_DIR/swarm-moe-tp2-nvlink.csv"

# --- Test 2: MoE TP=1 Single GPU ---
N=$((N+1))
echo ""
echo ">>> TEST $N/$TOTAL: MoE — TP=1 Single GPU"
reboot_server
start_server "coder4" "--tp 1" "16384"
wait_for_api
run_bench "cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit" "MoE TP=1 Single GPU" \
    "$RESULTS_DIR/swarm-moe-tp1.csv"

# --- Test 3: MoE TP=2 PCIe ---
N=$((N+1))
echo ""
echo ">>> TEST $N/$TOTAL: MoE — TP=2 PCIe (NVLink disabled)"
reboot_server
start_server "coder4" "--tp 2 --no-nvlink"
wait_for_api
run_bench "cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit" "MoE TP=2 PCIe" \
    "$RESULTS_DIR/swarm-moe-tp2-pcie.csv"

# ==========================================================================
# Dense: Qwen2.5-Coder-32B-Instruct AWQ (32B active params per token)
# ==========================================================================

# --- Test 4: Dense TP=2 NVLink ---
N=$((N+1))
echo ""
echo ">>> TEST $N/$TOTAL: Dense — TP=2 NVLink"
reboot_server
start_dense_server "2"
wait_for_api
run_bench "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ" "Dense TP=2 NVLink" \
    "$RESULTS_DIR/swarm-dense-tp2-nvlink.csv"

# --- Test 5: Dense TP=1 Single GPU ---
# Note: 18GB model on 24GB GPU — needs reduced context (12288) and higher util (0.95)
N=$((N+1))
echo ""
echo ">>> TEST $N/$TOTAL: Dense — TP=1 Single GPU"
reboot_server
start_dense_server "1" "" "12288" "0.95"
wait_for_api
run_bench "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ" "Dense TP=1 Single GPU" \
    "$RESULTS_DIR/swarm-dense-tp1.csv"

# --- Test 6: Dense TP=2 PCIe ---
# KNOWN BROKEN: NCCL_P2P_DISABLE=1 + enforce-eager causes SIGBUS during weight
# loading on vLLM 0.17 nightly. Without enforce-eager, CUDA graph capture crashes.
# Both paths fail — Dense TP=2 PCIe is not functional on this vLLM version.
N=$((N+1))
echo ""
echo ">>> TEST $N/$TOTAL: Dense — TP=2 PCIe (NVLink disabled) [EXPECTED TO FAIL]"
reboot_server
start_dense_server "2" "no-nvlink"
if wait_for_api; then
    run_bench "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ" "Dense TP=2 PCIe" \
        "$RESULTS_DIR/swarm-dense-tp2-pcie.csv"
else
    echo "  SKIPPED — Dense TP=2 PCIe failed to start (known vLLM 0.17 bug)"
fi

# --- Done ---
END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "========================================================================"
echo "ALL $TOTAL TESTS COMPLETE — ${ELAPSED} minutes total"
echo "Results in $RESULTS_DIR/swarm-*.csv"
echo "========================================================================"
