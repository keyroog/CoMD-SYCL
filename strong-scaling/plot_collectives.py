#!/usr/bin/env python3
"""
Strong-scaling plot for collective timer (commReduce)
comparing cuda-mpi, cuda-nccl-mpi, sycl-occl variants.

Scientific note:
For collectives, the step completion is limited by the slowest rank, so the
main curve uses MaxTime across ranks. The shaded band shows MinTime..MaxTime
across ranks to expose rank imbalance.
"""

import re
import glob
import os
from collections import defaultdict
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# helpers

VARIANT_LABEL = {
    "cuda-mpi": "MPI",
    "cuda-nccl-mpi": "NCCL",
    "sycl-occl": "oneCCL",
}

VARIANT_COLOR = {
    "cuda-mpi": "#1f77b4",  # blue
    "cuda-nccl-mpi": "#d62728",  # red
    "sycl-occl": "#2ca02c",  # green
}

VARIANT_MARKER = {
    "cuda-mpi": "o",
    "cuda-nccl-mpi": "s",
    "sycl-occl": "^",
}

TIMERS = ["commReduce"]

TIMER_TITLE = {
    "commReduce": "commReduce  (global reductions)",
}


def detect_variant(filename):
    """Extract variant key from filename, e.g. CoMD-cuda-nccl-mpi.*.yaml -> cuda-nccl-mpi"""
    base = os.path.basename(filename)
    # strip leading "CoMD-" and trailing ".TIMESTAMP.yaml"
    match = re.match(r"CoMD-(.+?)\.\d{4}", base)
    if match:
        return match.group(1)
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
    with open(path) as file_obj:
        text = file_obj.read()

    # TotalRanks
    match = re.search(r"TotalRanks:\s*(\d+)", text)
    if not match:
        return None
    nranks = int(match.group(1))

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
        block_match = re.search(block_pattern, across_text)
        if not block_match:
            result[timer_name] = None
            continue
        block = block_match.group(1)

        def extract(key):
            key_match = re.search(rf"{key}:\s+([\d.eE+\-]+)", block)
            return float(key_match.group(1)) if key_match else float("nan")

        result[timer_name] = {
            "avg": extract("AvgTime"),
            "min": extract("MinTime"),
            "max": extract("MaxTime"),
            "stdev": extract("StdevTime"),
        }

    return result


# collect data

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
    print(
        f"  {variant:20s}  ranks={parsed['nranks']:3d}  "
        f"commReduce={parsed['commReduce']['avg']:.4f}s"
    )


def summarise(runs, timer):
    """Average over multiple runs at the same (variant, nranks) point."""
    mins = [run[timer]["min"] for run in runs if run[timer] is not None]
    avgs = [run[timer]["avg"] for run in runs if run[timer] is not None]
    maxs = [run[timer]["max"] for run in runs if run[timer] is not None]
    if not avgs:
        return float("nan"), float("nan"), float("nan")
    return np.mean(mins), np.mean(avgs), np.mean(maxs)


# plot

timer = "commReduce"
fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharey=False)
fig.suptitle(
    "CoMD strong-scaling — commReduce "
    "(line: Across-Ranks MaxTime, band: MinTime..MaxTime)",
    fontsize=13,
    fontweight="bold",
)

for variant in sorted(VARIANT_LABEL.keys()):
    if variant not in data:
        continue
    nranks_sorted = sorted(data[variant].keys())
    xs, ys_max, ys_min, ys_avg = [], [], [], []
    for nranks in nranks_sorted:
        min_time, avg_time, max_time = summarise(data[variant][nranks], timer)
        xs.append(nranks)
        ys_min.append(min_time)
        ys_avg.append(avg_time)
        ys_max.append(max_time)

    xs = np.array(xs)
    ys_min = np.array(ys_min)
    ys_avg = np.array(ys_avg)
    ys_max = np.array(ys_max)

    label = VARIANT_LABEL[variant]
    color = VARIANT_COLOR[variant]
    marker = VARIANT_MARKER[variant]

    ax.plot(
        xs,
        ys_max,
        marker=marker,
        color=color,
        linewidth=1.8,
        markersize=7,
        label=label,
        zorder=3,
    )
    ax.fill_between(xs, ys_min, ys_max, color=color, alpha=0.18, zorder=2)
    ax.plot(xs, ys_avg, linestyle="--", color=color, linewidth=1.0, alpha=0.9, zorder=4)

ax.set_title(TIMER_TITLE[timer], fontsize=11)
ax.set_xlabel("Number of MPI ranks", fontsize=10)
ax.set_ylabel("Time [s]", fontsize=10)
ax.set_xticks(sorted({nranks for variant_data in data.values() for nranks in variant_data}))
ax.xaxis.set_major_formatter(plt.ScalarFormatter())
ax.set_xscale("log", base=2)
ax.set_ylim(bottom=0.0)
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
ax.legend(fontsize=10)

plt.tight_layout()
out_path = os.path.join(yaml_dir, "collectives_strong_scaling.pdf")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_path}")

# also save PNG for quick preview
out_png = out_path.replace(".pdf", ".png")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved: {out_png}")
