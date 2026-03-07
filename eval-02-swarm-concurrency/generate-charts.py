#!/usr/bin/env python3
"""Generate charts for the swarm benchmark article.

Reads CSV results from ../results/swarm-*.csv and produces PNG charts in charts/.
"""

import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS = Path(__file__).parent / "results"
CHARTS = Path(__file__).parent / "charts"
CHARTS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Color palette — professional, accessible
# ---------------------------------------------------------------------------
COLORS = {
    "MoE TP=2 NVLink":  "#2563eb",  # blue
    "MoE TP=1 Single":  "#7c3aed",  # purple
    "MoE TP=2 PCIe":    "#0891b2",  # teal
    "Dense TP=2 NVLink": "#dc2626", # red
    "Dense TP=1 Single": "#ea580c", # orange
}

MARKERS = {
    "MoE TP=2 NVLink":  "o",
    "MoE TP=1 Single":  "s",
    "MoE TP=2 PCIe":    "D",
    "Dense TP=2 NVLink": "^",
    "Dense TP=1 Single": "v",
}

FILES = {
    "MoE TP=2 NVLink":  "swarm-moe-tp2-nvlink.csv",
    "MoE TP=1 Single":  "swarm-moe-tp1.csv",
    "MoE TP=2 PCIe":    "swarm-moe-tp2-pcie.csv",
    "Dense TP=2 NVLink": "swarm-dense-tp2-nvlink.csv",
    "Dense TP=1 Single": "swarm-dense-tp1.csv",
}

# ---------------------------------------------------------------------------
# Parse CSVs
# ---------------------------------------------------------------------------
def load_results():
    """Return dict[config_name] -> list of row dicts."""
    data = {}
    for name, fname in FILES.items():
        path = RESULTS / fname
        if not path.exists():
            continue
        with open(path) as f:
            rows = list(csv.DictReader(f))
        data[name] = rows
    return data


def aggregate(data):
    """Compute per-config, per-concurrency averages across runs.

    Returns dict[config] -> dict with keys:
        concurrency: [1,2,3,4]
        eff_tps: [avg effective tok/s at each C]
        avg_wall: [avg wall seconds at each C]
        per_task_tps: {C: avg per-task tok/s}
    """
    results = {}
    for config, rows in data.items():
        by_c = defaultdict(list)
        per_task = defaultdict(list)
        for r in rows:
            c = int(r["concurrency"])
            by_c[c].append({
                "eff": float(r["run_effective_tok_per_sec"]),
                "wall": float(r["run_wall_seconds"]),
                "tps": float(r["tok_per_sec"]),
                "run": int(r["run"]),
            })
            per_task[c].append(float(r["tok_per_sec"]))

        concurrencies = sorted(by_c.keys())
        eff_tps = []
        avg_wall = []
        pt_tps = {}

        for c in concurrencies:
            # Average effective tok/s across runs (each run has 4 tasks,
            # but run_effective_tok_per_sec is per-run, so deduplicate)
            runs = {}
            walls = {}
            for entry in by_c[c]:
                run = entry["run"]
                runs[run] = entry["eff"]
                walls[run] = entry["wall"]
            eff_tps.append(sum(runs.values()) / len(runs))
            avg_wall.append(sum(walls.values()) / len(walls))
            pt_tps[c] = sum(per_task[c]) / len(per_task[c])

        results[config] = {
            "concurrency": concurrencies,
            "eff_tps": eff_tps,
            "avg_wall": avg_wall,
            "per_task_tps": pt_tps,
        }
    return results


# ---------------------------------------------------------------------------
# Chart styling
# ---------------------------------------------------------------------------
def style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Chart 1: Effective Throughput vs Concurrency
# ---------------------------------------------------------------------------
def chart_effective_throughput(agg):
    fig, ax = plt.subplots(figsize=(10, 6))

    for config in FILES:
        if config not in agg:
            continue
        d = agg[config]
        ax.plot(d["concurrency"], d["eff_tps"],
                color=COLORS[config], marker=MARKERS[config],
                linewidth=2.5, markersize=8, label=config)
        # Label the C=4 point
        ax.annotate(f'{d["eff_tps"][-1]:.0f}',
                     (d["concurrency"][-1], d["eff_tps"][-1]),
                     textcoords="offset points", xytext=(8, 0),
                     fontsize=9, color=COLORS[config], fontweight="bold")

    style_ax(ax,
             "Effective Throughput vs Concurrent Agents",
             "Concurrent Agents", "Effective tok/s (total output / wall time)")
    ax.set_xticks([1, 2, 3, 4])
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.set_xlim(0.8, 4.5)

    fig.tight_layout()
    fig.savefig(CHARTS / "effective-throughput.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> effective-throughput.png")


# ---------------------------------------------------------------------------
# Chart 2: Per-Task Throughput Degradation (contention)
# ---------------------------------------------------------------------------
def chart_contention(agg):
    fig, ax = plt.subplots(figsize=(10, 6))

    for config in FILES:
        if config not in agg:
            continue
        d = agg[config]
        cs = d["concurrency"]
        tps = [d["per_task_tps"][c] for c in cs]
        ax.plot(cs, tps,
                color=COLORS[config], marker=MARKERS[config],
                linewidth=2.5, markersize=8, label=config)
        # Annotate contention % at C=4
        if len(tps) >= 2:
            drop = (1 - tps[-1] / tps[0]) * 100
            ax.annotate(f'{tps[-1]:.0f} tok/s\n(-{drop:.0f}%)',
                         (cs[-1], tps[-1]),
                         textcoords="offset points", xytext=(8, -5),
                         fontsize=8, color=COLORS[config])

    style_ax(ax,
             "Per-Task Throughput Under Contention",
             "Concurrent Agents",
             "Per-task tok/s (individual task speed)")
    ax.set_xticks([1, 2, 3, 4])
    ax.legend(loc="lower left", framealpha=0.9, fontsize=9)
    ax.set_xlim(0.8, 4.8)

    fig.tight_layout()
    fig.savefig(CHARTS / "contention.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> contention.png")


# ---------------------------------------------------------------------------
# Chart 3: Wall-Clock Speedup at C=4
# ---------------------------------------------------------------------------
def chart_speedup(agg):
    fig, ax = plt.subplots(figsize=(8, 5))

    configs = [c for c in FILES if c in agg]
    speedups = []
    colors = []

    for config in configs:
        d = agg[config]
        s = d["avg_wall"][0] / d["avg_wall"][-1]  # C=1 wall / C=4 wall
        speedups.append(s)
        colors.append(COLORS[config])

    bars = ax.barh(range(len(configs)), speedups, color=colors, height=0.6,
                   edgecolor="white", linewidth=1.5)

    for i, (bar, s) in enumerate(zip(bars, speedups)):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f'{s:.2f}x', va="center", fontsize=11, fontweight="bold",
                color=colors[i])

    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs, fontsize=10)
    ax.axvline(x=1, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=4, color="gray", linestyle=":", alpha=0.3, label="Linear (4x)")
    style_ax(ax,
             "Wall-Clock Speedup: 4 Agents vs Serial",
             "Speedup (C=1 wall / C=4 wall)", "")
    ax.set_xlabel("Speedup factor", fontsize=11)
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(CHARTS / "speedup.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> speedup.png")


# ---------------------------------------------------------------------------
# Chart 4: MoE Advantage Ratio at Each Concurrency Level
# ---------------------------------------------------------------------------
def chart_moe_advantage(agg):
    """Compare best MoE (TP=2 NVLink) vs best Dense (TP=2 NVLink) at each C."""
    moe = agg.get("MoE TP=2 NVLink")
    dense = agg.get("Dense TP=2 NVLink")
    if not moe or not dense:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    cs = moe["concurrency"]
    ratios = [m / d for m, d in zip(moe["eff_tps"], dense["eff_tps"])]

    bars = ax.bar(cs, ratios, color=["#2563eb", "#3b82f6", "#60a5fa", "#93c5fd"],
                  width=0.6, edgecolor="white", linewidth=1.5)

    for bar, r in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{r:.1f}x', ha="center", fontsize=12, fontweight="bold",
                color="#1e40af")

    style_ax(ax,
             "MoE Advantage Over Dense (TP=2 NVLink)",
             "Concurrent Agents", "MoE / Dense effective tok/s ratio")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(0, max(ratios) + 1)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="Parity")

    fig.tight_layout()
    fig.savefig(CHARTS / "moe-advantage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> moe-advantage.png")


# ---------------------------------------------------------------------------
# Chart 5: Absolute Throughput Comparison — stacked context
# ---------------------------------------------------------------------------
def chart_absolute_comparison(agg):
    """Side-by-side: MoE vs Dense effective tok/s at C=1 and C=4."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    configs_moe = [c for c in FILES if "MoE" in c and c in agg]
    configs_dense = [c for c in FILES if "Dense" in c and c in agg]

    def plot_bars(ax, configs, title):
        x = range(len(configs))
        c1_vals = [agg[c]["eff_tps"][0] for c in configs]
        c4_vals = [agg[c]["eff_tps"][-1] for c in configs]
        labels = [c.replace("MoE ", "").replace("Dense ", "") for c in configs]

        w = 0.35
        b1 = ax.bar([i - w/2 for i in x], c1_vals, w, label="Serial (C=1)",
                     color="#94a3b8", edgecolor="white")
        b2 = ax.bar([i + w/2 for i in x], c4_vals, w, label="4 Agents (C=4)",
                     color="#2563eb" if "MoE" in title else "#dc2626",
                     edgecolor="white")

        for bar, v in zip(b1, c1_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{v:.0f}', ha="center", fontsize=9, color="#475569")
        for bar, v in zip(b2, c4_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{v:.0f}', ha="center", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        style_ax(ax, title, "", "Effective tok/s")
        ax.legend(fontsize=9)

    plot_bars(ax1, configs_moe, "MoE (3.3B active)")
    plot_bars(ax2, configs_dense, "Dense (32B active)")

    fig.suptitle("Serial vs 4-Agent Swarm: Effective Throughput",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(CHARTS / "serial-vs-swarm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> serial-vs-swarm.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading results...")
    data = load_results()
    print(f"  Loaded {len(data)} configs: {', '.join(data.keys())}")

    print("Computing aggregates...")
    agg = aggregate(data)

    # Print summary table
    print("\n  Config                  C=1 eff   C=4 eff   Speedup  Contention")
    print("  " + "-" * 70)
    for config in FILES:
        if config not in agg:
            continue
        d = agg[config]
        speedup = d["avg_wall"][0] / d["avg_wall"][-1]
        contention = (1 - d["per_task_tps"][4] / d["per_task_tps"][1]) * 100
        print(f"  {config:<24s} {d['eff_tps'][0]:7.1f}   {d['eff_tps'][-1]:7.1f}   "
              f"{speedup:5.2f}x   {contention:5.1f}%")

    print("\nGenerating charts...")
    chart_effective_throughput(agg)
    chart_contention(agg)
    chart_speedup(agg)
    chart_moe_advantage(agg)
    chart_absolute_comparison(agg)
    print(f"\nDone! Charts saved to {CHARTS}/")
