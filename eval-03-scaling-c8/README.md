# Eval 3: Scaling MoE to 8 Concurrent Agents

**Does throughput keep degrading past 4 agents, or is there a floor?**

Full analysis: [Scaling Coding Agent Swarms to 8 Concurrent Agents](https://ure.us/articles/scaling-coding-agent-swarms/)

## Results

| Concurrent Agents | Per-task tok/s | Effective tok/s | Marginal cost |
|:-----------------:|:--------------:|:---------------:|:-------------:|
| 1 | 169 | 169 | — |
| 2 | 154 | 220 | -9% per task |
| 3 | 137 | 247 | -11% per task |
| 4 | 125 | 404 | -9% per task |
| 5 | 130 | 316 | +4% per task |
| 6 | 124 | 372 | -5% per task |
| 7 | 126 | 369 | +2% per task |
| 8 | 123 | 388 | -2% per task |

**Per-task throughput plateaus at C=4 and stays flat through C=8.** Agents 5 through 8 are free — the contention penalty (169 → 125 tok/s, ~27%) happens between C=1 and C=4, then the floor is reached.

![Per-Task Throughput Plateau](charts/per-task-plateau.png)

## Key Findings

- **Contention floor at ~123 tok/s**: The 27% drop happens C=1→C=4, then per-task throughput stabilizes. Adding agents 5–8 costs nothing.
- **Engine throughput ceiling at ~500 tok/s**: vLLM's aggregate output saturates at ~500 tok/s on GDDR6X, regardless of concurrency. This is a memory bandwidth limit, not a model limit.
- **Prefix cache inflates C=1 baselines**: vLLM's automatic prefix caching gives the second measurement run a ~16% boost at C=1 (shared prefixes). At C=4+, cache eviction negates this — the plateau is real throughput.
- **8 agents on a single GPU**: 123 tok/s per task × 8 = 388 effective tok/s aggregate. The MoE architecture makes this possible because only 3.3B parameters are active per token.

![Effective Throughput C=1 to C=8](charts/effective-throughput-c8.png)

## Charts

| Chart | What it shows |
|-------|-------------|
| ![](charts/per-task-plateau.png) | Per-task throughput plateau at C=4 |
| ![](charts/effective-throughput-c8.png) | Effective throughput scaling to C=8 |
| ![](charts/engine-ceiling.png) | vLLM engine throughput ceiling |
| ![](charts/contention-floor.png) | Contention floor analysis |

## Scripts & Data

- **Benchmark engine**: [`../scripts/bench-swarm.py`](../scripts/bench-swarm.py)
- **Chart generation**: [`generate-charts-c8.py`](generate-charts-c8.py)
- **Raw data**: `results/swarm-coder4-*.csv`
- **Article**: [`article-swarm-scaling.md`](article-swarm-scaling.md)
