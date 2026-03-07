<!--
ARTICLE WRITER INSTRUCTIONS:

This is a self-contained benchmark section ready to paste into a larger article
about running local LLMs for coding agent swarms.

Context: The audience is developers and ML engineers building local inference
infrastructure for agentic coding workflows (Claude Code subagents, Cursor,
Windsurf, etc). They want to know: what hardware do I actually need?

This section answers that with hard numbers from real coding prompts — not
synthetic benchmarks or "capital of France" questions.

Key editorial notes:
- All numbers are from a single measurement run after a warmup pass
- Both models use AWQ 4-bit quantization (hardware-accelerated Marlin kernels
  on Ampere SM 8.6) — apples-to-apples comparison
- The PCIe test uses NCCL_P2P_DISABLE=1 to disable NVLink peer-to-peer.
  No socket transport hacks — clean PCIe fallback
- "Thinking mode" (chain-of-thought reasoning) was disabled for both models.
  A follow-up article will cover the thinking mode quality/speed trade-off
- The MoE model (Qwen3-Coder) is one generation newer than the dense model
  (Qwen2.5-Coder). This is intentional: these are the best available coding
  models in each architecture class. The comparison isolates the practical
  question "which model should I run?" not the theoretical "is MoE faster
  at equal quality?"

Tone: Technical, direct, data-driven. No hype. Let the numbers speak.
-->

## Coding Benchmark: MoE vs Dense on Real Tasks

### The Test

We threw five real coding tasks at both models — the kind of work that coding agents actually do in production:

| Prompt | Task Type | What It Tests |
|--------|-----------|---------------|
| **Implement LRU Cache** | Algorithm implementation | Data structure design, O(1) complexity, type hints |
| **Debug Merge Sort** | Bug finding & fixing | Code comprehension, error analysis, corrected output |
| **Pytest Suite** | Test generation | Parametrize, edge cases, 12+ test cases for a cron parser |
| **Refactor Flask Route** | Architecture refactoring | Service layer, repository pattern, separation of concerns |
| **Rate Limiter Design** | System design + implementation | Token bucket, thread safety, decorator pattern |

Each prompt was run with `max_tokens=4096`, `temperature=0.7`, and a system prompt disabling thinking mode. Two passes per configuration: warmup (torch.compile cache), then measurement.

### The Models

| | MoE | Dense |
|---|---|---|
| **Model** | Qwen3-Coder-30B-A3B-Instruct | Qwen2.5-Coder-32B-Instruct |
| **Quantization** | AWQ 4-bit (Marlin kernels) | AWQ 4-bit (Marlin kernels) |
| **Total parameters** | 30B | 32B |
| **Active parameters/token** | **3.3B** (8 of 128 experts) | **32B** (all weights) |
| **Model size on disk** | 16.9 GB | 19.5 GB |
| **Native tool calling** | Yes | No |

Both use identical Marlin AWQ kernels — the only variable is architecture.

### Results

#### Qwen3-Coder-30B-A3B AWQ-4bit (MoE, 3.3B active)

| Configuration | Impl | Debug | Test | Refactor | Design | **Average** |
|---------------|------|-------|------|----------|--------|-------------|
| TP=2 NVLink | 170.0 | 167.3 | 169.6 | 168.5 | 170.7 | **169.7 tok/s** |
| TP=1 Single GPU | 169.2 | 168.8 | 168.0 | 168.1 | 168.5 | **168.4 tok/s** |
| TP=2 PCIe only | 165.9 | 162.9 | 163.2 | 160.7 | 164.5 | **163.5 tok/s** |

#### Qwen2.5-Coder-32B-Instruct AWQ (Dense, 32B active)

| Configuration | Impl | Debug | Test | Refactor | Design | **Average** |
|---------------|------|-------|------|----------|--------|-------------|
| TP=2 NVLink | 65.6 | 65.1 | 65.5 | 64.9 | 65.6 | **65.4 tok/s** |
| TP=1 Single GPU | 41.0 | 40.9 | 40.9 | 41.0 | 41.0 | **41.0 tok/s** |
| TP=2 PCIe only | 63.9 | 63.5 | 63.7 | 63.4 | 63.8 | **63.7 tok/s** |

### Head-to-Head

| Configuration | MoE | Dense | MoE Advantage |
|---------------|-----|-------|---------------|
| **TP=2 NVLink** | 169.7 | 65.4 | **2.6x** |
| **TP=1 Single GPU** | 168.4 | 41.0 | **4.1x** |
| **TP=2 PCIe only** | 163.5 | 63.7 | **2.6x** |

### What the Numbers Tell Us

#### 1. MoE doesn't care about your GPU topology

The MoE model delivers **163–170 tok/s regardless of configuration**. Single GPU, dual GPU, NVLink, PCIe — the spread is only 3.8%. The model activates 3.3B parameters per token; a single RTX 3090's 936 GB/s memory bandwidth can serve that without breaking a sweat.

The only reason to add a second GPU is context length: one 24GB GPU fits the 17GB model plus ~7GB of KV cache (~32K context). Two GPUs give you 131K.

#### 2. Dense models need two GPUs — and benefit from NVLink

The dense model tells a different story. Going from one GPU to two with NVLink gives a **59% throughput boost** (41 → 65 tok/s). That's expected: 32B active parameters are memory-bandwidth-bound, and two GPUs double the available bandwidth.

Interestingly, PCIe-only dual GPU (63.7 tok/s) comes within 2.7% of NVLink (65.4 tok/s) for these coding prompts. The all-reduce payload at this batch size is small enough that PCIe Gen 4 handles it without becoming a bottleneck. NVLink's advantage would show up more at higher concurrency.

#### 3. Single GPU is where MoE dominates hardest

With one RTX 3090 — an $800 card — the MoE model runs at **168 tok/s**. The dense model manages **41 tok/s**. That's a **4.1x advantage**.

At 168 tok/s, a 1000-token function implementation completes in **~6 seconds**. Three parallel coding agents can each generate a complete module while you review the first output. At 41 tok/s, the same task takes 24 seconds — too slow for interactive agent loops.

#### 4. Throughput is remarkably consistent across task types

Both models show near-flat throughput regardless of prompt complexity. The MoE model varies from 160.7 to 170.7 tok/s (6% spread); the dense model varies even less. This means you can reliably predict completion times for agent task scheduling — the model won't suddenly slow down on harder prompts.

### Hardware Implications

| If you have... | MoE tok/s | Dense tok/s | Recommendation |
|----------------|-----------|-------------|----------------|
| **1x RTX 3090** | 168 | 41 | MoE. No contest. |
| **2x RTX 3090 (no NVLink)** | 164 | 64 | MoE for speed, Dense if you need its quality. |
| **2x RTX 3090 (NVLink)** | 170 | 65 | NVLink adds <4% for MoE. Save the money. |

### A Note on Thinking Mode

Both Qwen3-Coder and Qwen2.5-Coder were benchmarked with thinking mode **disabled** — the system prompt instructed direct answers without chain-of-thought reasoning. Qwen3-Coder supports an optional thinking mode that generates `<think>...</think>` reasoning tokens before the response, which may improve code quality at the cost of additional token generation.

The throughput numbers above represent **raw generation speed**. With thinking mode enabled, the model generates the same tok/s, but a portion of those tokens are internal reasoning that the agent doesn't use — reducing effective (useful) throughput. The trade-off between thinking mode quality gains and speed cost is the subject of a follow-up benchmark.

### Methodology

**Hardware:** 2x NVIDIA RTX 3090 24GB (Ampere, SM 8.6), NV3 NVLink, AMD Threadripper, 64GB DDR4, PM1735 NVMe. Ubuntu 24.04, CUDA 12.8, driver 570.133.20.

**Software:** vLLM 0.17.0rc1.dev119, `--disable-custom-all-reduce`, `--enable-chunked-prefill`, `--max-num-seqs 4`, `--gpu-memory-utilization 0.92`.

**Quantization:** Both models use AWQ 4-bit with Marlin kernels — hardware-accelerated on Ampere Tensor Cores (FP16/INT4). No FP8 (RTX 3090 lacks native FP8 compute; see quantization note below).

**PCIe isolation:** `NCCL_P2P_DISABLE=1` to force NCCL all-reduce over PCIe instead of NVLink. Shared memory transport remains enabled — this simulates a real dual-GPU system without NVLink, not an artificially handicapped one.

**Protocol:** Each configuration: start vLLM via `systemd-run` (clean process management), wait for API health check, run 5 prompts as warmup, run 5 prompts for measurement. Tokens/second = `completion_tokens` from API response / wall-clock time. All responses completed naturally (`finish_reason: stop`) except one MoE test-generation prompt that hit the 4096-token cap.

### Why AWQ 4-bit, Not FP8?

The RTX 3090 (Ampere, SM 8.6) has Tensor Cores that natively accelerate FP16, BF16, INT8, and INT4 operations. It does **not** have native FP8 Tensor Core instructions — those were introduced in Ada Lovelace (SM 8.9) and Hopper (SM 9.0).

When you load an FP8 model on a 3090, inference engines like vLLM use Marlin kernels to decompress FP8 weights to FP16 on the fly before feeding them to the Tensor Cores. This works, but the decompression adds a compute tax on every token. Our earlier testing showed FP8 running **13% slower** than AWQ 4-bit on the same model (154 vs 173 tok/s).

AWQ 4-bit with Marlin kernels is a mature, heavily optimized code path for Ampere hardware. It produces smaller model files (17GB vs 30GB for FP8), leaves more VRAM for KV cache, and delivers higher throughput. The quality impact is negligible — perplexity increases by ~0.7%.

**Bottom line:** On RTX 3090/4090 and similar consumer GPUs, always choose AWQ or GPTQ 4-bit over FP8. FP8 only makes sense on hardware with native FP8 compute (H100, A100 next-gen, RTX 5090).

---

*Benchmarked March 7, 2026. All code, scripts, and configuration available at [repo link].*
