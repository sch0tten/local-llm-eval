# Eval 2: MoE vs Dense — Swarm Concurrency (1→4 Agents)

**What happens to throughput when you run multiple coding agents on the same GPU?**

Full analysis: [Best Local LLM for Coding Agent Swarms](https://ure.us/articles/best-local-llm-coding-agent-swarm/)

## Results

| | Serial (1 agent) | Swarm (4 agents) | Speedup |
|---|---|---|---|
| **MoE** (TP=2 NVLink) | 166 tok/s | **399 eff. tok/s** | 2.83x |
| **MoE** (1x GPU) | 167 tok/s | **336 eff. tok/s** | 2.06x |
| **Dense** (TP=2 NVLink) | 31 tok/s | 82 eff. tok/s | 3.07x |
| **Dense** (1x GPU) | 39 tok/s | 127 eff. tok/s | 5.53x |

MoE delivers **399 effective tok/s** with 4 concurrent agents on two GPUs — completing 4 coding tasks in 26 seconds instead of 75. On a single GPU: **336 eff. tok/s**. Dense maxes out at 127 eff. tok/s.

**MoE is 4.9x faster than Dense under swarm load.**

![Effective Throughput](charts/effective-throughput.png)

## Method

- 16 unique coding prompts (4 types × 4 variants)
- Concurrency levels: 1, 2, 3, 4 simultaneous agents
- `max_tokens=8192`, `temperature=0.7`
- Full server reboot between GPU config changes for clean driver state
- 5 GPU configurations: MoE (TP=2 NVLink, TP=1, TP=2 PCIe) + Dense (TP=2 NVLink, TP=1)

## Key Findings

- **Contention is real but manageable**: Per-task throughput drops ~27% from C=1 to C=4 for MoE, but total effective throughput still scales 2.8x.
- **MoE advantage holds at every concurrency level**: The 4.9x ratio over Dense is consistent from 1 to 4 agents.
- **Single GPU MoE beats dual-GPU Dense**: 336 eff. tok/s (MoE, 1 GPU) vs 82 eff. tok/s (Dense, 2 GPUs + NVLink).
- **NVLink matters more under concurrency**: 19% boost for MoE swarm vs 3.8% at serial — but still not essential.

## Charts

| Chart | What it shows |
|-------|-------------|
| ![](charts/effective-throughput.png) | Effective throughput vs concurrent agents |
| ![](charts/contention.png) | Per-task throughput under contention |
| ![](charts/speedup.png) | Wall-clock speedup |
| ![](charts/moe-advantage.png) | MoE advantage ratio over Dense |
| ![](charts/serial-vs-swarm.png) | Serial vs 4-agent swarm comparison |

## Scripts & Data

- **Benchmark engine**: [`../scripts/bench-swarm.py`](../scripts/bench-swarm.py)
- **Orchestrator**: [`../scripts/run-swarm-bench.sh`](../scripts/run-swarm-bench.sh)
- **Chart generation**: [`generate-charts.py`](generate-charts.py)
- **Raw data**: `results/swarm-{moe,dense}-*.csv`
- **Article**: [`article-swarm-benchmark.md`](article-swarm-benchmark.md)
