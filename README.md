# MoE vs Dense: Local LLM Benchmarks for Agentic Coding

**How fast can a single $800 GPU run a coding agent swarm?**

This repository contains the benchmark code, scripts, and raw results behind the article:
**[Best Local LLM for Agentic Coding](https://ure.us/articles/best-local-llm-agentic-coding/)**

We systematically compared **Mixture-of-Experts (MoE) vs Dense** coding models across every GPU configuration on a dual RTX 3090 NVLink system — answering questions the community has been guessing at:

- Does NVLink actually matter?
- Is a second GPU worth it?
- Can MoE models really replace dense models for coding?
- AWQ vs FP8 — which is faster on consumer GPUs?

## TL;DR

### Single-Request Throughput

| Configuration | MoE (3.3B active) | Dense (32B active) | MoE Advantage |
|---------------|-------------------|-------------------|---------------|
| **2x RTX 3090 NVLink** | 166 tok/s | 31 tok/s | **5.4x** |
| **1x RTX 3090** | 167 tok/s | 39 tok/s | **4.3x** |

### 4-Agent Swarm (Effective Throughput)

| Configuration | MoE (3.3B active) | Dense (32B active) | MoE Advantage |
|---------------|-------------------|-------------------|---------------|
| **2x RTX 3090 NVLink** | **399 eff. tok/s** | 82 eff. tok/s | **4.9x** |
| **1x RTX 3090** | **336 eff. tok/s** | 127 eff. tok/s | **2.6x** |

A single RTX 3090 runs the MoE coding model at **336 effective tok/s with 4 concurrent agents** — completing 4 coding tasks in 28 seconds. The dense model on two GPUs with NVLink maxes out at 82 eff. tok/s.

## Models Tested

| Model | Architecture | Quantization | Active Params | Kernel |
|-------|-------------|-------------|---------------|--------|
| [Qwen3-Coder-30B-A3B](https://huggingface.co/cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit) | MoE | AWQ 4-bit | 3.3B of 30B | Marlin |
| [Qwen2.5-Coder-32B](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-AWQ) | Dense | AWQ 4-bit | 32B | Marlin |

Both models use identical **AWQ 4-bit quantization** with Marlin kernels — hardware-accelerated on Ampere Tensor Cores. This ensures an apples-to-apples comparison where the only variable is MoE vs Dense architecture.

> **Why not FP8?** The RTX 3090 (SM 8.6) lacks native FP8 Tensor Cores. FP8 models work via software decompression but run **13% slower** than AWQ 4-bit. See the [full article](https://ure.us/articles/best-local-llm-agentic-coding/) for details.

## Hardware

| Component | Spec |
|-----------|------|
| GPUs | 2x NVIDIA RTX 3090 24GB (Ampere, SM 8.6) |
| Interconnect | NV3 NVLink (3 lanes, ~112 GB/s bidirectional) |
| CPU | AMD Threadripper (TRX40) |
| RAM | 64GB DDR4 |
| Storage | PM1735 NVMe 5.4TB (ZFS) |
| OS | Ubuntu 24.04, CUDA 12.8, driver 570.133.20 |
| Inference | vLLM 0.17.0rc1 |

## Benchmark Prompts

Five real coding tasks — the kind of work coding agents actually perform:

| # | Task | Type |
|---|------|------|
| 1 | Implement LRU Cache (O(1), doubly linked list) | Algorithm implementation |
| 2 | Find & fix bug in merge-sorted-lists | Debug & fix |
| 3 | Comprehensive pytest suite for cron parser | Test generation |
| 4 | Refactor Flask route → service layer | Architecture refactoring |
| 5 | Design rate limiter with token bucket | System design |

Each prompt: `max_tokens=4096`, `temperature=0.7`, thinking mode disabled. Two passes per configuration (warmup + measurement).

## GPU Configurations

| Config | What | How |
|--------|------|-----|
| **TP=2 NVLink** | Both GPUs, NVLink active | `--tensor-parallel-size 2` |
| **TP=1 Single** | One GPU only | `--tensor-parallel-size 1`, `CUDA_VISIBLE_DEVICES=0` |
| **TP=2 PCIe** | Both GPUs, NVLink disabled | `--tensor-parallel-size 2`, `NCCL_P2P_DISABLE=1` |

## Key Findings

### MoE doesn't care about your GPU topology

163–170 tok/s regardless of single GPU, dual GPU, NVLink, or PCIe. The 3.3B active parameters fit comfortably in one GPU's memory bandwidth.

### Single GPU is where MoE dominates hardest

**4.1x faster** than dense on a single RTX 3090. At 168 tok/s, a 1000-token function completes in ~6 seconds. Three parallel agents can each generate a module while you review the first.

### NVLink is a luxury, not a necessity

MoE: 3.8% NVLink boost. Dense: 2.7%. Neither model significantly benefits from NVLink at batch size 1. Save the money.

### AWQ 4-bit beats FP8 by 13% on Ampere

RTX 3090 lacks native FP8 compute — AWQ has mature, optimized Marlin kernels. Always use AWQ/GPTQ on consumer GPUs.

## Repository Structure

```
├── eval-01-single-request/          # Eval 1: MoE vs Dense at concurrency=1
│   ├── article-moe-benchmark.md     #   Full MoE vs Dense analysis
│   ├── article-coding-benchmark.md  #   Single-request coding benchmark
│   ├── bench-coding.py              #   5 coding prompts benchmark
│   ├── bench-full.py                #   Full matrix (MoE + FP8 + Dense)
│   ├── bench-all.py                 #   Quick benchmark (4 general prompts)
│   ├── bench.py                     #   Minimal single-model benchmark
│   └── results/                     #   bench-coding-results.json
├── eval-02-swarm-concurrency/       # Eval 2: MoE vs Dense, concurrency 1→4
│   ├── article-swarm-benchmark.md   #   Swarm concurrency analysis
│   ├── generate-charts.py           #   Chart generation
│   ├── charts/                      #   Generated PNG charts
│   └── results/                     #   swarm-{moe,dense}-*.csv
├── eval-03-scaling-c8/              # Eval 3: Scaling MoE to 8 concurrent agents
│   ├── article-swarm-scaling.md     #   Scaling analysis article
│   ├── generate-charts-c8.py        #   Chart generation
│   ├── charts/                      #   Generated PNG charts
│   └── results/                     #   swarm-coder4-*.csv
├── scripts/                         # Shared infrastructure
│   ├── bench-swarm.py               #   Swarm benchmark engine (used by eval-02, eval-03)
│   ├── run-swarm-bench.sh           #   Full suite orchestrator
│   ├── model-serve.sh               #   vLLM multi-model launcher (inference server)
│   ├── switch-model.sh              #   Remote model switcher (workstation → inference server)
│   └── chat.py                      #   Interactive chat client
└── README.md
```

## Quick Start

### Configuration

All scripts use environment variables for server connection. Copy the example and fill in your values:

```bash
cp .env.example .env
# Edit .env with your server IP, SSH target, and paths
source .env
```

| Variable | Description | Example |
|----------|-------------|---------|
| `VLLM_HOST` | vLLM server hostname/IP | `192.168.1.100` |
| `VLLM_SSH` | SSH target for remote management | `user@192.168.1.100` |
| `VLLM_MODEL_CMD` | Path to model-serve.sh on server | `/home/user/model-serve.sh` |
| `VLLM_VENV` | Python venv path (model-serve.sh) | `/opt/vllm/.venv` |

### Benchmark topology

```
┌──────────────────────┐        LAN        ┌──────────────────────┐
│  Workstation          │ ─── 1 Gbps ────> │  Inference Server    │
│  (benchmark scripts)  │                   │  vLLM on :8000       │
│                       │ <── API resp ──── │  2x RTX 3090 NVLink  │
└──────────────────────┘                   └──────────────────────┘
```

All benchmarks run on the workstation and send OpenAI-compatible API requests to the inference server over the local network. The network hop is part of the measurement.

### Run your own benchmarks

```bash
# 1. Start vLLM with a model
./scripts/model-serve.sh coder4 --tp 2

# 2. Run the coding benchmark
python3 eval-01-single-request/bench-coding.py

# 3. Quick benchmark (4 general prompts)
python3 eval-01-single-request/bench-all.py vllm
```

### Switch models remotely

```bash
./scripts/switch-model.sh coder4    # MoE AWQ-4bit (173 tok/s)
./scripts/switch-model.sh llm       # Qwen3.5-27B general
./scripts/switch-model.sh stop      # Stop all
./scripts/switch-model.sh           # Show status
```

## Critical vLLM Flags for RTX 3090

```bash
--disable-custom-all-reduce  # REQUIRED: custom all-reduce crashes SM 8.6
--enable-chunked-prefill     # Better memory efficiency for long contexts
--gpu-memory-utilization 0.92  # Max safe value for TP=2
--dtype float16              # Required for GPTQ models
--tool-call-parser qwen3_coder  # Native tool calling for Qwen3-Coder
```

## The Bigger Picture

These benchmarks support a specific architecture for AI-assisted development:

```
Claude Code (Opus) — Orchestrator
  ├── Subagent: code generation  ─┐
  ├── Subagent: test writing     ─┤──→ LiteLLM Proxy ──→ vLLM
  └── Subagent: refactoring      ─┘    (API translation)   (local model)
                                                            1x RTX 3090
                                                            168 tok/s
```

The orchestrator (Claude, GPT-4) handles planning and review. The local MoE model handles volume: boilerplate, tests, single-function implementations — at 168 tok/s with zero API costs.

## Related

- **Full article:** [Best Local LLM for Agentic Coding](https://ure.us/articles/best-local-llm-agentic-coding/)
- **Qwen3-Coder:** [Model card](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)
- **vLLM:** [GitHub](https://github.com/vllm-project/vllm)

## License

MIT — see [LICENSE](LICENSE).

---

*Benchmarked March 2026 by [Stefano Schotten](https://github.com/sch0tten). Built with Claude Code.*
