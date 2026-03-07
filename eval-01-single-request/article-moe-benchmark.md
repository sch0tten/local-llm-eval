# MoE vs Dense for Local Coding Agents: Why NVLink Doesn't Matter and One GPU Is Enough

## Objective

The AI coding agent ecosystem is evolving fast. Tools like Claude Code, Cursor, and Windsurf can orchestrate sub-agents to parallelize coding tasks — but running those agents against cloud APIs gets expensive at scale. This study aims to help the community answer a practical question: **what's the cheapest, fastest way to run a local LLM coding swarm on consumer hardware?**

We systematically benchmarked two coding models — a Mixture-of-Experts (MoE) and an equivalent-class dense model — across every GPU configuration available on a dual RTX 3090 NVLink system: single GPU, dual GPU with NVLink, and dual GPU over PCIe only. The results challenge common assumptions about multi-GPU inference and show that MoE architectures fundamentally change the hardware calculus for local agent coding.

**All benchmark code, scripts, and configuration are open.** We hope this data helps others building local inference infrastructure for agentic coding workflows.

## TL;DR

| | MoE (Qwen3-Coder-30B-A3B) | Dense (Qwen2.5-Coder-32B) |
|---|---|---|
| TP=2 NVLink | **173 tok/s** | 66 tok/s |
| TP=1 Single GPU | **169 tok/s** | 41 tok/s |
| TP=2 PCIe (no NVLink) | **167 tok/s** | 44 tok/s |
| NVLink benefit | 4% | 33% |
| Second GPU benefit | 2% | 60% |

The MoE model is **2.6–4.1x faster** than the dense model across all configs. NVLink gives a 33% boost for dense but only 4% for MoE. A single RTX 3090 runs the MoE coder at 169 tok/s — fast enough for a real-time agent swarm.

## Why This Matters: Local Coding Agent Swarms

The emerging pattern in AI-assisted development is **orchestrator + swarm**: a powerful model (Claude Opus, GPT-4) plans and decomposes work, then delegates parallelizable tasks to faster, cheaper models. Code generation, test writing, refactoring, documentation — these are all embarrassingly parallel.

Running those swarm agents locally has three advantages:
1. **Cost**: Zero API costs for the bulk generation work
2. **Latency**: No network round-trip, sub-second time-to-first-token
3. **Privacy**: Code never leaves your machine

For this to be practical, the local model needs:
- **>100 tok/s** for interactive agent loops (generate → review → iterate)
- **Native tool calling** for agentic workflows (file I/O, shell, search)
- **Good code quality** to avoid constant correction by the orchestrator
- **Affordable hardware** — ideally a single consumer GPU

As we'll show, MoE models hit all four requirements on an $800 used RTX 3090.

## The Models

### MoE: Qwen3-Coder-30B-A3B-Instruct

[Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) — Alibaba's purpose-built agentic coding model.

| Property | Value |
|----------|-------|
| Architecture | Mixture-of-Experts (MoE) |
| Total parameters | 30.5B |
| Active parameters per token | **3.3B** (8 of 128 experts) |
| Context | 256K native |
| Tool calling | Native support |
| Designed for | Agentic coding, SWE-bench, terminal-bench |

We tested two quantizations:
- **FP8** (official, 30GB on disk) — [Qwen3-Coder-30B-A3B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8)
- **AWQ 4-bit** (community, 16.9GB on disk) — [cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit](https://huggingface.co/cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit)

### Dense: Qwen2.5-Coder-32B-Instruct

[Qwen2.5-Coder-32B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-AWQ) — Alibaba's dense coding model, AWQ 4-bit quantized.

| Property | Value |
|----------|-------|
| Architecture | Dense transformer |
| Total parameters | 32.5B |
| Active parameters per token | **32.5B** (all weights active) |
| Context | 128K (configured at 32K for VRAM) |
| Quantization | AWQ 4-bit (official) |

This is the **control group** — a similarly-sized coding model with the same quantization, but dense architecture. It lets us isolate the impact of MoE vs dense on throughput and GPU interconnect sensitivity.

## The Hardware

| Component | Spec |
|-----------|------|
| GPUs | 2x NVIDIA RTX 3090 24GB (Ampere, SM 8.6) |
| Interconnect | NV3 NVLink (3 lanes, ~112 GB/s bidirectional) |
| PCIe | Gen 4.0 x16 (~25 GB/s per GPU) |
| CPU | AMD Threadripper (Gigabyte TRX40 Aorus Master) |
| RAM | 64GB DDR4 |
| Storage | PM1735 enterprise NVMe (5.4TB ZFS pool) |
| OS | Ubuntu 24.04, CUDA 12.8, driver 570.133.20 |

## The Benchmark

Four prompts designed to simulate the range of agent coding tasks:

| Prompt | Type | Simulates |
|--------|------|-----------|
| "What is the capital of France?" | Short factual | Quick validation, tool result parsing |
| "Explain how a combustion engine works in 3 paragraphs." | Medium prose | Documentation generation |
| "Write a detailed comparison of Python, Rust, and Go for web services..." | Long analysis | Architecture analysis, code review |
| "Write a Python function that finds the longest palindromic substring..." | Code generation | The core agent coding workload |

**Methodology:**
- Each prompt capped at 512 output tokens
- Served via vLLM's OpenAI-compatible chat completions API
- System prompt: "Answer directly without internal reasoning or thinking steps"
- Two runs per config: first for warmup (torch.compile cache), second for measurement
- Tokens/second computed from API-reported `completion_tokens` / wall-clock time
- vLLM 0.17.0 nightly with `--enable-chunked-prefill`, `--max-num-seqs 4`

**Three GPU configurations per model:**
1. **TP=2, NVLink** — both GPUs, NVLink active (the "best case")
2. **TP=1, Single GPU** — one GPU only (the budget case)
3. **TP=2, PCIe only** — both GPUs, NVLink disabled via `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NET=Socket` (simulates systems without NVLink)

## Results

### MoE: Qwen3-Coder-30B-A3B AWQ-4bit (3.3B active)

| Configuration | Short | Medium | Long | Code | **Average** | Context |
|---------------|-------|--------|------|------|-------------|---------|
| TP=2, NVLink | 137 | 172 | 174 | 174 | **173.0 tok/s** | 131K |
| TP=1, Single GPU | 141 | 169 | 170 | 170 | **169.1 tok/s** | 32K |
| TP=2, PCIe only | 141 | 166 | 167 | 167 | **166.7 tok/s** | 131K |

### MoE: Qwen3-Coder-30B-A3B FP8 (3.3B active)

| Configuration | Short | Medium | Long | Code | **Average** | Context |
|---------------|-------|--------|------|------|-------------|---------|
| TP=2, NVLink | 101 | 153 | 153 | 157 | **153.7 tok/s** | 65K |

### Dense: Qwen2.5-Coder-32B AWQ-4bit (32B active)

| Configuration | Short | Medium | Long | Code | **Average** | Context |
|---------------|-------|--------|------|------|-------------|---------|
| TP=2, NVLink | 52 | 66 | 66 | 66 | **65.6 tok/s** | 32K |
| TP=1, Single GPU | 35 | 41 | 41 | 41 | **41.1 tok/s** | 8K |
| TP=2, PCIe only | 23 | 44 | 44 | 44 | **43.9 tok/s** | 32K |

## Analysis

### 1. NVLink: Critical for Dense, Irrelevant for MoE

This is the headline finding.

| Interconnect impact | MoE | Dense |
|---------------------|-----|-------|
| NVLink vs PCIe | **4% faster** | **33% faster** |

In tensor-parallel inference, GPUs synchronize activations after each layer via all-reduce operations. The payload is the same size regardless of architecture — the full hidden state (~5120 × fp16 = 10KB per token per layer). What changes is the **compute-to-communication ratio**.

For the dense 32B model, each GPU processes 16B parameters worth of matrix multiplications between all-reduce calls. This takes time, and the GPU can overlap some communication with compute. But when you slow down the interconnect from 112 GB/s (NVLink) to ~25 GB/s (PCIe), the communication becomes the bottleneck — hence the 33% penalty.

For the MoE model, each GPU processes only ~1.65B active parameters (half of 3.3B at TP=2) between all-reduce calls. The compute finishes so fast that even PCIe can deliver the activation tensors in time. The model is compute-bound, not communication-bound — **NVLink has nothing to speed up**.

### 2. Second GPU: Essential for Dense, Optional for MoE

| Second GPU impact | MoE | Dense |
|-------------------|-----|-------|
| TP=1 vs TP=2 NVLink | **2% faster** | **60% faster** |

The dense model nearly doubles throughput going from 1→2 GPUs (41→66 tok/s) because it's memory-bandwidth-bound: more GPUs = more aggregate memory bandwidth to feed the 32B of active weights. The MoE model barely notices the second GPU because 3.3B active weights can be served at near-maximum throughput from a single GPU's 936 GB/s memory bandwidth.

The only reason to use two GPUs with MoE is **context length**: a single 24GB GPU can only fit the 17GB model + ~7GB of KV cache (~32K context). Two GPUs give you 131K context.

### 3. AWQ-4bit Beats FP8 by 13%

| Quantization | tok/s | Model size | Context |
|-------------|-------|-----------|---------|
| AWQ 4-bit | **173** | 16.9 GB | 131K |
| FP8 | 154 | 30 GB | 65K |

Counterintuitive — fewer bits should mean less memory bandwidth, but FP8 loses because:
- The RTX 3090 (SM 8.6) **lacks native FP8 compute**. vLLM uses the Marlin kernel for weight-only FP8 decompression, adding overhead.
- AWQ-4bit with Marlin kernels is a more mature, better-optimized code path.
- Smaller weights (17GB vs 30GB) leave more VRAM for KV cache, reducing cache pressure.

Quality impact is negligible: perplexity increases from 1.616 to 1.628 (0.7%).

### 4. MoE Delivers 2.6–4.1x the Throughput of Dense

| Config | MoE | Dense | Speedup |
|--------|-----|-------|---------|
| TP=2 NVLink | 173 | 66 | **2.6x** |
| TP=1 Single GPU | 169 | 41 | **4.1x** |
| TP=2 PCIe | 167 | 44 | **3.8x** |

The MoE advantage grows as you **remove hardware**: with NVLink and two GPUs, it's 2.6x. With a single GPU, it's 4.1x. MoE models are the great equalizer — they make expensive interconnects and multi-GPU setups optional rather than required.

Both models are ~30B total parameters, both are AWQ-4bit quantized, both are coding-specialized. The only difference is architecture. The MoE model activates 3.3B of its 30B parameters per token; the dense model activates all 32B. That 10:1 ratio in active parameters translates to a 2.6–4.1x throughput advantage depending on how much hardware you throw at it.

## The Agent Architecture

```
┌──────────────────────────────────────────────────────┐
│  Claude Code (Opus) — Orchestrator                    │
│  Plans, decomposes tasks, reviews results              │
│                                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ Subagent │  │ Subagent │  │ Subagent │  ...        │
│  │ code gen │  │ tests    │  │ refactor │             │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘            │
└───────┼──────────────┼──────────────┼─────────────────┘
        │              │              │
        ▼              ▼              ▼
   ┌──────────────────────────────────────────┐
   │  LiteLLM Proxy                           │
   │  Anthropic API  ←→  OpenAI API           │
   └─────────────────┬────────────────────────┘
                     │
                     ▼
   ┌──────────────────────────────────────────┐
   │  vLLM                                    │
   │  Qwen3-Coder-30B-A3B · AWQ-4bit         │
   │  1x RTX 3090 · 169 tok/s · 32K ctx      │
   │  or 2x RTX 3090 · 173 tok/s · 131K ctx  │
   └──────────────────────────────────────────┘
```

At 170 tok/s, a 500-token code generation completes in **~3 seconds**. Three parallel subagents can each produce a complete file in the time it takes to review the output. This enables interactive agent loops — dispatch, generate, review, iterate — entirely on local hardware, with zero API costs for the bulk generation work.

The orchestrator (Claude Code) handles what it does best: planning, architectural reasoning, complex multi-file refactors, and quality review. The local swarm handles the volume: boilerplate generation, test scaffolding, documentation, single-function implementations. The result is faster iteration with lower costs.

## Practical Recommendations

### For single-GPU builders ($800 budget)

Buy a used RTX 3090. Run the AWQ-4bit MoE model at TP=1. You get 169 tok/s with 32K context — enough for most coding agent tasks. This is the best performance-per-dollar option for local inference.

### For dual-GPU builders ($1600 budget)

**Skip NVLink.** Two RTX 3090s without NVLink (PCIe only) deliver 167 tok/s with 131K context. The 4% NVLink speedup doesn't justify the cost and complexity of NVLink bridges and compatible motherboards. Use the savings for more RAM or faster storage.

The only reason to go dual-GPU is if you need long context windows (>32K tokens). For typical agent coding tasks (function generation, test writing, code review), 32K on a single GPU is plenty.

### For quantization choices

AWQ-4bit is the sweet spot. It's faster than FP8 (+13%), smaller on disk (17GB vs 30GB), fits on a single 24GB GPU, and quality loss is negligible (0.7% perplexity increase). FP8 only makes sense on GPUs with native FP8 compute (Ada Lovelace / Hopper).

### Dense models still have their place

If you need the absolute best code quality and have two GPUs with NVLink, a dense 32B model at 66 tok/s is still viable for non-interactive workloads (background batch generation, overnight code review runs). But for interactive agent loops where latency matters, MoE is the clear winner.

## The Stack

| Component | What | Why |
|-----------|------|-----|
| [vLLM 0.17.0 nightly](https://github.com/vllm-project/vllm) | Inference engine | Required for Qwen3 MoE support |
| `--disable-custom-all-reduce` | vLLM flag | Custom all-reduce crashes on SM 8.6 (RTX 3090) |
| `--tool-call-parser qwen3_coder` | vLLM flag | Enables native tool calling for agent workflows |
| `NCCL_P2P_DISABLE=1` | Environment variable | Forces PCIe transport (for benchmarking or non-NVLink systems) |
| [LiteLLM](https://github.com/BerriAI/litellm) | API proxy | Translates Anthropic API ↔ OpenAI API for Claude Code integration |

**Note:** Qwen3.5 MoE models (e.g., Qwen3.5-35B-A3B) are currently broken in vLLM 0.17 nightly — different architecture from Qwen3 MoE. The Qwen3-Coder MoE works fine.

## Conclusion

MoE architectures fundamentally change the economics of local LLM inference. The traditional playbook — buy multiple GPUs, get NVLink, maximize memory bandwidth — was written for dense models. With MoE, a single $800 RTX 3090 serves a 30B-parameter coding model at 170 tokens per second. That's fast enough to power a swarm of coding agents that can generate, test, and refactor code in parallel — all running locally.

The implications extend beyond our specific setup. As the industry moves toward larger MoE models (Qwen3-Coder-Next at 80B/3B active, DeepSeek-V3 at 671B/37B active), the NVLink premium matters less and less. PCIe bandwidth is sufficient for the all-reduce operations in MoE inference. NVLink remains valuable for dense models and prefill-heavy workloads, but for autoregressive generation — the bread and butter of coding agents — it's a luxury, not a necessity.

**The bottom line:** If you're building local infrastructure for AI coding agents, start with one RTX 3090, an AWQ-4bit MoE model, and vLLM. You'll get 170 tok/s, native tool calling, and a capable coding model — for the price of two months of API credits.

---

*Benchmarked on March 6, 2026. Hardware: 2x NVIDIA RTX 3090 24GB, NV3 NVLink, AMD Threadripper, 64GB RAM, PM1735 NVMe. Software: vLLM 0.17.0rc1, Ubuntu 24.04, CUDA 12.8, driver 570.133.20. All benchmark code available at [github link].*
