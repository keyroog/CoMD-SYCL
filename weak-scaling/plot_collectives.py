#!/usr/bin/env python3
"""
Weak-scaling plot for commReduce collective timer
comparing cuda-mpi-gpuaware, cuda-nccl-mpi, sycl-occl variants.

Uses MaxTime across ranks as the main metric (the collective blocks all ranks
until the slowest finishes). The shaded band shows MinTime..MaxTime to expose
rank imbalance.
"""

import re
import glob
import os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size":          70,
    "axes.labelsize":     76,
    "xtick.labelsize":    66,
    "ytick.labelsize":    66,
    "legend.fontsize":    66,
    "xtick.major.pad":    16,
    "ytick.major.pad":    16,
    "xtick.major.size":   16,
    "ytick.major.size":   16,
    "xtick.major.width":   3,
    "ytick.major.width":   3,
    "axes.linewidth":      3,
    "axes.labelpad":      20,
    "legend.framealpha":  0.9,
    "lines.linewidth":     6,
    "lines.markersize":   28,
})

# ── helpers ────────────────────────────────────────────────────────────────────

VARIANT_LABEL = {
    "cuda-mpi-gpuaware": "MPI (GPU-Aware)",
    "cuda-nccl-mpi": "NCCL",
    "sycl-occl":     "oneCCL",
}

VARIANT_COLOR = {
    "cuda-mpi-gpuaware": "#9467bd",   # blue
    "cuda-nccl-mpi": "#2ca02c",   # red
    "sycl-occl":     "#1f77b4",   # green
}

VARIANT_MARKER = {
    "cuda-mpi-gpuaware": "o",
    "cuda-nccl-mpi": "s",
    "sycl-occl":     "^",
}

TIMERS = ["commReduce"]

TIMER_TITLE = {
    "commReduce": "commReduce  (global reductions)",
}


def detect_variant(filename):
    """Extract variant key from filename, e.g. CoMD-cuda-nccl-mpi.*.yaml -> cuda-nccl-mpi"""
    base = os.path.basename(filename)
    # strip leading "CoMD-" and trailing ".TIMESTAMP.yaml"
    m = re.match(r"CoMD-(.+?)\.\d{4}", base)
    if m:
        return m.group(1)
    return None


def parse_yaml(path):
    """
    Returns a dict:
      {
        "nranks": int,
        "commReduce": {"avg": float, "min": float, "max": float, "stdev": float},
      }
    or None on parse error.
    """
    with open(path) as f:
        text = f.read()

    # TotalRanks
    m = re.search(r"TotalRanks:\s*(\d+)", text)
    if not m:
        return None
    nranks = int(m.group(1))

    # Locate the "Across Ranks" block
    across_match = re.search(r"Performance Results Across Ranks:(.*)", text, re.DOTALL)
    if not across_match:
        return None
    across_text = across_match.group(1)

    result = {"nranks": nranks}

    for timer_name in TIMERS:
        # Find the timer block inside the Across Ranks section
        # Pattern: "  Timer: commHalo\n" followed by the stat lines
        block_pattern = rf"Timer:\s+{timer_name}\s*\n((?:\s+\w.*\n)*)"
        bm = re.search(block_pattern, across_text)
        if not bm:
            result[timer_name] = None
            continue
        block = bm.group(1)

        def extract(key):
            km = re.search(rf"{key}:\s+([\d.eE+\-]+)", block)
            return float(km.group(1)) if km else float("nan")

        result[timer_name] = {
            "avg":   extract("AvgTime"),
            "min":   extract("MinTime"),
            "max":   extract("MaxTime"),
            "stdev": extract("StdevTime"),
        }

    return result


# ── collect data ───────────────────────────────────────────────────────────────

yaml_dir = os.path.dirname(os.path.abspath(__file__))
files = glob.glob(os.path.join(yaml_dir, "CoMD-*.yaml"))

# data[variant][nranks] = list of per-file parsed results
data = defaultdict(lambda: defaultdict(list))

for path in sorted(files):
    variant = detect_variant(path)
    if variant not in VARIANT_LABEL:
        print(f"  [skip] unknown variant in: {os.path.basename(path)}")
        continue
    parsed = parse_yaml(path)
    if parsed is None:
        print(f"  [skip] parse error: {os.path.basename(path)}")
        continue
    data[variant][parsed["nranks"]].append(parsed)
    print(f"  {variant:20s}  ranks={parsed['nranks']:3d}  "
          f"commReduce max={parsed['commReduce']['max']:.4f}s")

def summarise(runs, timer):
    """Median over multiple runs at the same (variant, nranks) point."""
    mins = [r[timer]["min"] for r in runs if r[timer] is not None]
    avgs = [r[timer]["avg"] for r in runs if r[timer] is not None]
    maxs = [r[timer]["max"] for r in runs if r[timer] is not None]
    if not avgs:
        return float("nan"), float("nan"), float("nan")
    return np.median(mins), np.median(avgs), np.median(maxs)


# ── plot ───────────────────────────────────────────────────────────────────────

timer = "commReduce"
fig, ax = plt.subplots(1, 1, figsize=(40, 20))

for variant in sorted(VARIANT_LABEL.keys()):
    if variant not in data:
        continue
    nranks_sorted = sorted(data[variant].keys())
    xs, ys_min, ys_avg, ys_max = [], [], [], []
    for nr in nranks_sorted:
        min_t, avg_t, max_t = summarise(data[variant][nr], timer)
        xs.append(nr)
        ys_min.append(min_t)
        ys_avg.append(avg_t)
        ys_max.append(max_t)

    xs = np.array(xs)
    ys_min = np.array(ys_min)
    ys_avg = np.array(ys_avg)
    ys_max = np.array(ys_max)

    label = VARIANT_LABEL[variant]
    color = VARIANT_COLOR[variant]
    marker = VARIANT_MARKER[variant]

    ax.plot(xs, ys_max, marker=marker, color=color, label=label, zorder=3)
    ax.fill_between(xs, ys_min, ys_max, color=color, alpha=0.18, zorder=2)
    ax.plot(xs, ys_avg, linestyle="--", color=color, linewidth=4, alpha=0.9, zorder=4)

ax.set_title(TIMER_TITLE[timer])
ax.set_xlabel("Number of MPI ranks")
ax.set_ylabel("Time [s]")
ax.set_xticks(sorted({nr for v in data.values() for nr in v}))
ax.xaxis.set_major_formatter(plt.ScalarFormatter())
ax.set_xscale("log", base=2)
ax.set_ylim(bottom=0.0)
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
ax.legend()

plt.tight_layout()
out_path = os.path.join(yaml_dir, "commReduce_weak_scaling.pdf")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_path}")

# also save PNG for quick preview
out_png = out_path.replace(".pdf", ".png")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved: {out_png}")
