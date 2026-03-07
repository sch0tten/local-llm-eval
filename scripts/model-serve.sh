#!/bin/bash
# Multi-model vLLM serve script (2x RTX 3090 NVLink)
# Usage: ./model-serve.sh <model> [--tp 1|2] [--no-nvlink] [--ctx <tokens>]

set -euo pipefail

VENV="${VLLM_VENV:-$HOME/.venv}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores

source "${VENV}/bin/activate"

MODEL_KEY="${1:-27b}"
shift || true

# Parse optional flags
TP=2
NO_NVLINK=0
CTX_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tp) TP="$2"; shift 2 ;;
        --no-nvlink) NO_NVLINK=1; shift ;;
        --ctx) CTX_OVERRIDE="$2"; shift 2 ;;
        *) shift ;;
    esac
done

if [[ "$NO_NVLINK" == "1" ]]; then
    export NCCL_P2P_DISABLE=1
    echo "NVLink DISABLED — using PCIe for all-reduce"
fi

case "$MODEL_KEY" in
    27b)
        MODEL="Qwen/Qwen3.5-27B-FP8"
        EXTRA_ARGS="--gpu-memory-utilization 0.92 --max-model-len 92000 --kv-cache-dtype fp8"
        ;;
    27b-int4)
        MODEL="Qwen/Qwen3.5-27B-GPTQ-Int4"
        EXTRA_ARGS="--gpu-memory-utilization 0.92 --max-model-len 131072 --quantization gptq --dtype float16"
        ;;
    9b)
        MODEL="Qwen/Qwen3.5-9B"
        EXTRA_ARGS="--gpu-memory-utilization 0.92 --max-model-len 131072"
        ;;
    coder)
        MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
        EXTRA_ARGS="--gpu-memory-utilization 0.92 --max-model-len 65536 --kv-cache-dtype fp8 --tool-call-parser qwen3_coder"
        ;;
    coder4)
        MODEL="cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit"
        EXTRA_ARGS="--gpu-memory-utilization 0.92 --max-model-len 131072 --tool-call-parser qwen3_coder"
        ;;
    dense)
        MODEL="Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"
        EXTRA_ARGS="--gpu-memory-utilization 0.92 --max-model-len 32768 --dtype float16"
        ;;
    *)
        echo "Usage: $0 <model> [--tp 1|2] [--no-nvlink] [--ctx <tokens>]"
        echo ""
        echo "  27b-int4  Qwen3.5-27B-GPTQ-Int4 (general purpose)"
        echo "  coder     Qwen3-Coder-30B-A3B-Instruct-FP8 (agent coding)"
        echo "  coder4    Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit (agent coding, faster)"
        echo "  dense     Qwen2.5-Coder-32B-Instruct-AWQ (dense baseline)"
        echo "  27b       Qwen3.5-27B-FP8"
        echo "  9b        Qwen3.5-9B"
        echo ""
        echo "Options:"
        echo "  --tp 1|2      tensor parallel size (default: 2)"
        echo "  --no-nvlink   disable NVLink, force PCIe communication"
        echo "  --ctx <n>     override max context length"
        exit 1
        ;;
esac

# Apply context length override if specified
if [[ -n "$CTX_OVERRIDE" ]]; then
    EXTRA_ARGS=$(echo "$EXTRA_ARGS" | sed "s/--max-model-len [0-9]*/--max-model-len $CTX_OVERRIDE/")
    echo "Context override: $CTX_OVERRIDE"
fi

echo "Starting $MODEL (TP=$TP)..."
exec vllm serve "${MODEL}" \
    --tensor-parallel-size "$TP" \
    --trust-remote-code \
    --enable-chunked-prefill \
    --max-num-seqs 4 \
    --disable-custom-all-reduce \
    --host 0.0.0.0 \
    --port 8000 \
    $EXTRA_ARGS
