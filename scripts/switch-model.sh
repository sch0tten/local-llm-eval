#!/bin/bash
# Switch the vLLM model on inference server
# Usage: ./switch-model.sh [llm|coder|coder4|stop|status]

set -euo pipefail

SERVER="${VLLM_SSH:-user@your-server}"

get_model() {
    ssh -o ConnectTimeout=5 "$SERVER" \
        "curl -s http://localhost:8000/v1/models 2>/dev/null" 2>/dev/null \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true
}

case "${1:-status}" in
    llm)
        echo "Starting Qwen3.5-27B-GPTQ-Int4..."
        ssh "$SERVER" "sudo systemctl start qwen-llm"
        echo "Waiting for API..."
        for i in $(seq 1 60); do
            sleep 5
            MODEL=$(get_model)
            if [[ -n "$MODEL" ]]; then
                echo "Ready: $MODEL"
                exit 0
            fi
            echo -n "."
        done
        echo -e "\nTimeout. Check: ssh $VLLM_SSH journalctl -u qwen-llm -n 20"
        exit 1
        ;;
    coder)
        echo "Starting Qwen3-Coder-30B-A3B..."
        ssh "$SERVER" "sudo systemctl start qwen-coder"
        echo "Waiting for API..."
        for i in $(seq 1 60); do
            sleep 5
            MODEL=$(get_model)
            if [[ -n "$MODEL" ]]; then
                echo "Ready: $MODEL"
                exit 0
            fi
            echo -n "."
        done
        echo -e "\nTimeout. Check: ssh $VLLM_SSH journalctl -u qwen-coder -n 20"
        exit 1
        ;;
    coder4)
        echo "Starting Qwen3-Coder-30B-A3B AWQ-4bit..."
        ssh "$SERVER" "sudo systemctl start qwen-coder4"
        echo "Waiting for API..."
        for i in $(seq 1 60); do
            sleep 5
            MODEL=$(get_model)
            if [[ -n "$MODEL" ]]; then
                echo "Ready: $MODEL"
                exit 0
            fi
            echo -n "."
        done
        echo -e "\nTimeout. Check: ssh $VLLM_SSH journalctl -u qwen-coder4 -n 20"
        exit 1
        ;;
    stop)
        echo "Stopping all models..."
        ssh "$SERVER" "sudo systemctl stop qwen-llm qwen-coder qwen-coder4 2>/dev/null; echo done"
        ;;
    status|"")
        echo "Models:"
        echo "  llm     Qwen3.5-27B-GPTQ-Int4 (50 tok/s, 131K ctx)"
        echo "  coder   Qwen3-Coder-30B-A3B-Instruct-FP8 (154 tok/s, 65K ctx)"
        echo "  coder4  Qwen3-Coder-30B-A3B AWQ-4bit (173 tok/s, 131K ctx)"
        echo "  stop    Stop all"
        echo ""
        MODEL=$(get_model)
        if [[ -n "$MODEL" ]]; then
            echo "Currently serving: $MODEL"
        else
            LLM=$(ssh -o ConnectTimeout=5 "$SERVER" "systemctl is-active qwen-llm 2>/dev/null" 2>/dev/null || echo "inactive")
            CODER=$(ssh -o ConnectTimeout=5 "$SERVER" "systemctl is-active qwen-coder 2>/dev/null" 2>/dev/null || echo "inactive")
            CODER4=$(ssh -o ConnectTimeout=5 "$SERVER" "systemctl is-active qwen-coder4 2>/dev/null" 2>/dev/null || echo "inactive")
            if [[ "$LLM" == "activating" || "$CODER" == "activating" || "$CODER4" == "activating" ]]; then
                echo "Loading..."
            else
                echo "No model running"
            fi
        fi
        ;;
    *)
        echo "Usage: $0 [llm|coder|coder4|stop|status]"
        exit 1
        ;;
esac
