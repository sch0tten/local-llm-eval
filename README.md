# Local LLM Eval: MoE vs Dense for Agentic Coding

**How fast can consumer GPUs run a coding agent swarm?**

Three evaluations comparing **Mixture-of-Experts vs Dense** LLM architectures for agentic coding workloads on dual RTX 3090 NVLink. Benchmark code, raw data, and full analysis articles included.

## Evaluations

### [Eval 1: Single-Request Throughput](eval-01-single-request/)

MoE vs Dense at concurrency=1 across three GPU configurations (NVLink, single GPU, PCIe-only).

| Configuration | MoE (3.3B active) | Dense (32B active) | MoE Advantage |
|---------------|-------------------|-------------------|---------------|
| **2x RTX 3090 NVLink** | 166 tok/s | 31 tok/s | **5.4x** |
| **1x RTX 3090** | 167 tok/s | 39 tok/s | **4.3x** |

MoE doesn't care about GPU topology. NVLink is a luxury, not a necessity.

### [Eval 2: Swarm Concurrency (1→4 Agents)](eval-02-swarm-concurrency/)

What happens when you dispatch 2, 3, or 4 coding tasks simultaneously to the same GPU?

| | Serial (1 agent) | Swarm (4 agents) | Speedup |
|---|---|---|---|
| **MoE** (TP=2 NVLink) | 166 tok/s | **399 eff. tok/s** | 2.83x |
| **MoE** (1x GPU) | 167 tok/s | **336 eff. tok/s** | 2.06x |
| **Dense** (TP=2 NVLink) | 31 tok/s | 82 eff. tok/s | 3.07x |
| **Dense** (1x GPU) | 39 tok/s | 127 eff. tok/s | 5.53x |

Single GPU MoE (336 eff. tok/s) beats dual-GPU Dense with NVLink (82 eff. tok/s) by 4x.

### [Eval 3: Scaling to 8 Concurrent Agents](eval-03-scaling-c8/)

Does throughput keep degrading past 4 agents, or is there a floor?

| Concurrent Agents | Per-task tok/s | Effective tok/s |
|:-----------------:|:--------------:|:---------------:|
| 1 | 169 | 169 |
| 4 | 125 | 404 |
| 8 | 123 | 388 |

Per-task throughput plateaus at C=4 and stays flat through C=8. Agents 5–8 are free.

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

## Models

| Model | Architecture | Quantization | Active Params | Kernel |
|-------|-------------|-------------|---------------|--------|
| [Qwen3-Coder-30B-A3B](https://huggingface.co/cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit) | MoE | AWQ 4-bit | 3.3B of 30B | Marlin |
| [Qwen2.5-Coder-32B](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-AWQ) | Dense | AWQ 4-bit | 32B | Marlin |

Both use identical AWQ 4-bit quantization with Marlin kernels — hardware-accelerated on Ampere Tensor Cores.

> **Why not FP8?** The RTX 3090 (SM 8.6) lacks native FP8 Tensor Cores. FP8 runs **13% slower** than AWQ 4-bit via software decompression.

## Benchmark Topology

```
┌──────────────────────┐        LAN        ┌──────────────────────┐
│  Workstation          │ ─── 1 Gbps ────> │  Inference Server    │
│  (benchmark scripts)  │                   │  vLLM on :8000       │
│                       │ <── API resp ──── │  2x RTX 3090 NVLink  │
└──────────────────────┘                   └──────────────────────┘
```

All benchmarks run on the workstation and send OpenAI-compatible API requests to the inference server over the local network. The network hop is part of the measurement.

## Quick Start

```bash
# Configure server connection
cp .env.example .env
# Edit .env with your server IP, SSH target, and paths
source .env

# Start a model
./scripts/model-serve.sh coder4 --tp 2

# Run benchmarks
python3 eval-01-single-request/bench-coding.py        # Single-request eval
python3 scripts/bench-swarm.py --concurrency 1 2 3 4  # Swarm eval
```

| Variable | Description | Example |
|----------|-------------|---------|
| `VLLM_HOST` | vLLM server hostname/IP | `192.168.1.100` |
| `VLLM_SSH` | SSH target for remote management | `user@192.168.1.100` |
| `VLLM_MODEL_CMD` | Path to model-serve.sh on server | `/home/user/model-serve.sh` |
| `VLLM_VENV` | Python venv path (model-serve.sh) | `/opt/vllm/.venv` |

## Repository Structure

```
├── eval-01-single-request/          # MoE vs Dense at concurrency=1
├── eval-02-swarm-concurrency/       # MoE vs Dense, concurrency 1→4
├── eval-03-scaling-c8/              # Scaling MoE to 8 concurrent agents
├── scripts/                         # Shared: serving, benchmarking, model switching
└── README.md
```

Each eval directory contains its own README, benchmark scripts, raw results, charts, and full analysis articles.

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

## Critical vLLM Flags for RTX 3090

```bash
--disable-custom-all-reduce  # REQUIRED: custom all-reduce crashes SM 8.6
--enable-chunked-prefill     # Better memory efficiency for long contexts
--gpu-memory-utilization 0.92  # Max safe value for TP=2
--dtype float16              # Required for GPTQ models
--tool-call-parser qwen3_coder  # Native tool calling for Qwen3-Coder
```

## Related

- **Articles:** [Part 1](https://ure.us/articles/best-local-llm-agentic-coding/) · [Part 2](https://ure.us/articles/best-local-llm-coding-agent-swarm/) · [Part 3](https://ure.us/articles/scaling-coding-agent-swarms/)
- **Qwen3-Coder:** [Model card](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)
- **vLLM:** [GitHub](https://github.com/vllm-project/vllm)

## License

MIT — see [LICENSE](LICENSE).

---

*Benchmarked March 2026 by [Stefano Schotten](https://github.com/sch0tten). Built with Claude Code.*
