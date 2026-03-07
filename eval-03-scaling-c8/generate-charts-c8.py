#!/usr/bin/env python3
"""Generate charts for the Part 3 swarm scaling article (C=1 to C=8).

Reads CSV from ../results/swarm-coder4-tp2-nvlink-c8.csv and produces PNGs in charts/.
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS = Path(__file__).parent / "results"
CHARTS = Path(__file__).parent / "charts"
CHARTS.mkdir(exist_ok=True)

CSV_FILE = RESULTS / "swarm-coder4-tp2-nvlink-c8.csv"

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
BLUE = "#2563eb"
BLUE_LIGHT = "#93c5fd"
PURPLE = "#7c3aed"
GRAY = "#64748b"
RED = "#dc2626"
GREEN = "#059669"

# ---------------------------------------------------------------------------
# Parse CSV
# ---------------------------------------------------------------------------
def load_results():
    with open(CSV_FILE) as f:
        return list(csv.DictReader(f))


def aggregate(rows):
    """Compute per-concurrency averages."""
    by_c = defaultdict(list)
    per_task = defaultdict(list)

    for r in rows:
        c = int(r["concurrency"])
        by_c[c].append({
            "eff": float(r["run_effective_tok_per_sec"]),
            "wall": float(r["run_wall_seconds"]),
            "tps": float(r["tok_per_sec"]),
            "run": int(r["run"]),
            "task_type": r["task_type"],
        })
        per_task[c].append(float(r["tok_per_sec"]))

    concurrencies = sorted(by_c.keys())
    eff_tps = []
    avg_wall = []
    pt_tps = {}
    per_type = defaultdict(dict)  # type -> {c: avg_tps}

    for c in concurrencies:
        # Deduplicate runs for effective tok/s
        runs = {}
        walls = {}
        for entry in by_c[c]:
            runs[entry["run"]] = entry["eff"]
            walls[entry["run"]] = entry["wall"]
        eff_tps.append(sum(runs.values()) / len(runs))
        avg_wall.append(sum(walls.values()) / len(walls))
        pt_tps[c] = sum(per_task[c]) / len(per_task[c])

        # Per task type
        type_vals = defaultdict(list)
        for entry in by_c[c]:
            type_vals[entry["task_type"]].append(entry["tps"])
        for tt, vals in type_vals.items():
            per_type[tt][c] = sum(vals) / len(vals)

    return {
        "concurrency": concurrencies,
        "eff_tps": eff_tps,
        "avg_wall": avg_wall,
        "per_task_tps": pt_tps,
        "per_type": dict(per_type),
    }


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
# Chart 1: Per-Task Throughput Plateau (the key finding)
# ---------------------------------------------------------------------------
def chart_per_task_plateau(agg):
    fig, ax = plt.subplots(figsize=(10, 6))
    cs = agg["concurrency"]
    tps = [agg["per_task_tps"][c] for c in cs]

    ax.plot(cs, tps, color=BLUE, marker="o", linewidth=2.5, markersize=9,
            label="Per-task tok/s", zorder=5)

    # Shade the plateau region
    ax.axhspan(120, 132, xmin=0, xmax=1, alpha=0.08, color=BLUE)
    ax.axvline(x=4, color=GRAY, linestyle=":", alpha=0.5, linewidth=1)
    ax.annotate("Plateau: C=4 to C=8",
                xy=(6, 127), fontsize=11, color=BLUE, fontweight="bold",
                ha="center")
    ax.annotate("Contention\nramp",
                xy=(2.5, 150), fontsize=10, color=GRAY, ha="center",
                fontstyle="italic")

    # Annotate each point
    for c, t in zip(cs, tps):
        offset = (0, 12) if c <= 3 else (0, -18)
        ax.annotate(f'{t:.0f}', (c, t), textcoords="offset points",
                    xytext=offset, fontsize=9, ha="center",
                    fontweight="bold" if c >= 4 else "normal",
                    color=BLUE if c >= 4 else GRAY)

    style_ax(ax,
             "Per-Task Throughput: The Plateau After C=4",
             "Concurrent Agents",
             "Per-task tok/s (individual task speed)")
    ax.set_xticks(cs)
    ax.set_ylim(100, 185)
    ax.set_xlim(0.5, 8.5)

    fig.tight_layout()
    fig.savefig(CHARTS / "per-task-plateau.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> per-task-plateau.png")


# ---------------------------------------------------------------------------
# Chart 2: Effective Throughput C=1 to C=8
# ---------------------------------------------------------------------------
def chart_effective_throughput_c8(agg):
    fig, ax = plt.subplots(figsize=(10, 6))
    cs = agg["concurrency"]
    eff = agg["eff_tps"]

    ax.plot(cs, eff, color=BLUE, marker="o", linewidth=2.5, markersize=9,
            label="Effective tok/s", zorder=5)

    # Engine ceiling reference line
    ax.axhline(y=500, color=RED, linestyle="--", alpha=0.4, linewidth=1.5)
    ax.annotate("Engine ceiling (~500 tok/s)",
                xy=(5, 505), fontsize=9, color=RED, fontstyle="italic")

    # Annotate each point
    for c, e in zip(cs, eff):
        offset = (8, -5) if c != 4 else (8, 5)
        ax.annotate(f'{e:.0f}', (c, e), textcoords="offset points",
                    xytext=offset, fontsize=9,
                    fontweight="bold" if c == 4 else "normal",
                    color=BLUE)

    style_ax(ax,
             "Effective Throughput: C=1 to C=8",
             "Concurrent Agents",
             "Effective tok/s (total output / wall time)")
    ax.set_xticks(cs)
    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(100, 550)

    fig.tight_layout()
    fig.savefig(CHARTS / "effective-throughput-c8.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> effective-throughput-c8.png")


# ---------------------------------------------------------------------------
# Chart 3: Engine Throughput Ceiling (aggregate gen tok/s from logs)
# ---------------------------------------------------------------------------
def chart_engine_ceiling(agg):
    """Shows the ~500 tok/s engine ceiling with concurrent request count."""
    fig, ax = plt.subplots(figsize=(10, 6))

    cs = agg["concurrency"]
    # Approximate engine aggregate: per_task_tps * concurrency
    # (capped by actual engine observations)
    engine_agg = [agg["per_task_tps"][c] * c for c in cs]

    ax.bar(cs, engine_agg, color=BLUE, width=0.6, edgecolor="white",
           linewidth=1.5, alpha=0.8, label="Per-task × C (theoretical aggregate)")

    # Actual ceiling line
    ax.axhline(y=500, color=RED, linestyle="--", linewidth=2, alpha=0.7,
               label="Observed engine ceiling (~500 tok/s)")

    # Per-task tok/s as secondary line on twin axis
    ax2 = ax.twinx()
    ax2.plot(cs, [agg["per_task_tps"][c] for c in cs], color=PURPLE,
             marker="s", linewidth=2, markersize=7, label="Per-task tok/s")
    ax2.set_ylabel("Per-task tok/s", fontsize=11, color=PURPLE)
    ax2.tick_params(axis="y", labelcolor=PURPLE)
    ax2.set_ylim(80, 200)
    ax2.spines["top"].set_visible(False)

    for c, e in zip(cs, engine_agg):
        ax.text(c, e + 15, f'{e:.0f}', ha="center", fontsize=8, color=BLUE)

    style_ax(ax,
             "Engine Throughput Budget vs Concurrent Agents",
             "Concurrent Agents",
             "Aggregate generation tok/s")
    ax.set_xticks(cs)
    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(0, 1200)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
              framealpha=0.9, fontsize=9)

    fig.tight_layout()
    fig.savefig(CHARTS / "engine-ceiling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> engine-ceiling.png")


# ---------------------------------------------------------------------------
# Chart 4: Contention Floor
# ---------------------------------------------------------------------------
def chart_contention_floor(agg):
    fig, ax = plt.subplots(figsize=(10, 6))
    cs = agg["concurrency"]
    baseline = agg["per_task_tps"][1]
    contention = [(1 - agg["per_task_tps"][c] / baseline) * 100 for c in cs]

    ax.plot(cs, contention, color=BLUE, marker="o", linewidth=2.5,
            markersize=9, zorder=5)

    # Shade the floor region
    ax.axhspan(22, 30, alpha=0.08, color=BLUE)
    ax.annotate("Contention floor: ~27%",
                xy=(6, 28.5), fontsize=11, color=BLUE, fontweight="bold",
                ha="center")

    for c, ct in zip(cs, contention):
        offset = (0, 10) if c <= 3 else (0, -15)
        ax.annotate(f'{ct:.0f}%', (c, ct), textcoords="offset points",
                    xytext=offset, fontsize=9, ha="center",
                    fontweight="bold" if c >= 4 else "normal",
                    color=BLUE if c >= 4 else GRAY)

    style_ax(ax,
             "Contention Floor: Penalty Stabilizes After C=4",
             "Concurrent Agents",
             "Per-task contention vs serial (%)")
    ax.set_xticks(cs)
    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(-2, 35)

    fig.tight_layout()
    fig.savefig(CHARTS / "contention-floor.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> contention-floor.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Loading {CSV_FILE}...")
    rows = load_results()
    print(f"  {len(rows)} rows loaded")

    print("Computing aggregates...")
    agg = aggregate(rows)

    # Summary
    print(f"\n  {'C':>3s}  {'Eff tok/s':>10s}  {'Per-task':>9s}  {'Wall (s)':>9s}  {'Contention':>11s}")
    print("  " + "-" * 50)
    baseline = agg["per_task_tps"][1]
    for i, c in enumerate(agg["concurrency"]):
        ct = (1 - agg["per_task_tps"][c] / baseline) * 100
        print(f"  {c:>3d}  {agg['eff_tps'][i]:>10.1f}  {agg['per_task_tps'][c]:>9.1f}"
              f"  {agg['avg_wall'][i]:>9.1f}  {ct:>10.1f}%")

    print("\nGenerating charts...")
    chart_per_task_plateau(agg)
    chart_effective_throughput_c8(agg)
    chart_engine_ceiling(agg)
    chart_contention_floor(agg)
    print(f"\nDone! Charts saved to {CHARTS}/")
