# Eval 1: MoE vs Dense — Single-Request Throughput

**Can a MoE coding model beat a dense model on consumer GPUs at batch size 1?**

Full analysis: [Best Local LLM for Agentic Coding](https://ure.us/articles/best-local-llm-agentic-coding/)

## Results

| Configuration | MoE (3.3B active) | Dense (32B active) | MoE Advantage |
|---------------|-------------------|-------------------|---------------|
| **2x RTX 3090 NVLink** | 166 tok/s | 31 tok/s | **5.4x** |
| **1x RTX 3090** | 167 tok/s | 39 tok/s | **4.3x** |

## Models

| Model | Architecture | Quantization | Active Params | Kernel |
|-------|-------------|-------------|---------------|--------|
| [Qwen3-Coder-30B-A3B](https://huggingface.co/cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit) | MoE | AWQ 4-bit | 3.3B of 30B | Marlin |
| [Qwen2.5-Coder-32B](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-AWQ) | Dense | AWQ 4-bit | 32B | Marlin |

Both use identical AWQ 4-bit quantization with Marlin kernels — hardware-accelerated on Ampere Tensor Cores. The only variable is MoE vs Dense architecture.

> **Why not FP8?** The RTX 3090 (SM 8.6) lacks native FP8 Tensor Cores. FP8 runs **13% slower** than AWQ 4-bit via software decompression.

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

- **MoE doesn't care about GPU topology**: 163–170 tok/s regardless of single GPU, dual GPU, NVLink, or PCIe.
- **Single GPU is where MoE dominates hardest**: 4.1x faster than dense on a single RTX 3090.
- **NVLink is a luxury**: MoE gets 3.8% boost, Dense gets 2.7%. Neither benefits significantly at batch size 1.
- **AWQ 4-bit beats FP8 by 13%** on Ampere — always use AWQ/GPTQ on consumer GPUs.

## Scripts

| Script | What |
|--------|------|
| `bench-coding.py` | 5 coding prompts, MoE vs Dense across 3 GPU configs |
| `bench-full.py` | Full 9-config matrix (MoE + FP8 + Dense) |
| `bench-all.py` | Quick benchmark — 4 general prompts |
| `bench.py` | Minimal single-model benchmark |

## Articles

- [`article-moe-benchmark.md`](article-moe-benchmark.md) — Full MoE vs Dense analysis
- [`article-coding-benchmark.md`](article-coding-benchmark.md) — Single-request coding benchmark
