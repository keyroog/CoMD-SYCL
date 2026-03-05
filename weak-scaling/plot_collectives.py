#!/usr/bin/env python3
"""
Weak-scaling plot for collective timers (commHalo, commReduce)
comparing cuda-mpi, cuda-nccl-mpi, sycl-occl variants.

Extracts AvgTime and StdevTime from the "Performance Results Across Ranks"
section of each CoMD YAML output file.
"""

import re
import glob
import os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── helpers ────────────────────────────────────────────────────────────────────

VARIANT_LABEL = {
    "cuda-mpi":      "MPI",
    "cuda-nccl-mpi": "NCCL",
    "sycl-occl":     "oneCCL",
}

VARIANT_COLOR = {
    "cuda-mpi":      "#1f77b4",   # blue
    "cuda-nccl-mpi": "#d62728",   # red
    "sycl-occl":     "#2ca02c",   # green
}

VARIANT_MARKER = {
    "cuda-mpi":      "o",
    "cuda-nccl-mpi": "s",
    "sycl-occl":     "^",
}

TIMERS = ["commHalo", "commReduce"]

TIMER_TITLE = {
    "commHalo":   "commHalo  (halo send/receive)",
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
        "commHalo":   {"avg": float, "min": float, "max": float, "stdev": float},
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
          f"commHalo={parsed['commHalo']['avg']:.4f}s  "
          f"commReduce={parsed['commReduce']['avg']:.4f}s")

# For each (variant, nranks) average across repeated runs (if any)
def summarise(runs, timer):
    """Average over multiple runs at the same (variant, nranks) point."""
    avgs = [r[timer]["avg"] for r in runs if r[timer] is not None]
    stdevs = [r[timer]["stdev"] for r in runs if r[timer] is not None]
    if not avgs:
        return float("nan"), float("nan")
    return np.mean(avgs), np.mean(stdevs)   # stdev: mean of per-run stdevs


# ── plot ───────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
fig.suptitle("CoMD weak-scaling — collective timers (Across Ranks AvgTime ± StdevTime)",
             fontsize=13, fontweight="bold")

for ax, timer in zip(axes, TIMERS):
    for variant in sorted(VARIANT_LABEL.keys()):
        if variant not in data:
            continue
        nranks_sorted = sorted(data[variant].keys())
        xs, ys, errs = [], [], []
        for nr in nranks_sorted:
            avg, stdev = summarise(data[variant][nr], timer)
            xs.append(nr)
            ys.append(avg)
            errs.append(stdev)

        xs = np.array(xs)
        ys = np.array(ys)
        errs = np.array(errs)

        label = VARIANT_LABEL[variant]
        color = VARIANT_COLOR[variant]
        marker = VARIANT_MARKER[variant]

        ax.plot(xs, ys, marker=marker, color=color, linewidth=1.8,
                markersize=7, label=label, zorder=3)
        ax.fill_between(xs, ys - errs, ys + errs,
                        color=color, alpha=0.18, zorder=2)
        ax.errorbar(xs, ys, yerr=errs, fmt="none",
                    ecolor=color, elinewidth=1.2, capsize=4, zorder=4)

    ax.set_title(TIMER_TITLE[timer], fontsize=11)
    ax.set_xlabel("Number of MPI ranks", fontsize=10)
    ax.set_ylabel("Time [s]", fontsize=10)
    ax.set_xticks(sorted({nr for v in data.values() for nr in v}))
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    ax.set_xscale("log", base=2)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(fontsize=10)

plt.tight_layout()
out_path = os.path.join(yaml_dir, "collectives_weak_scaling.pdf")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_path}")

# also save PNG for quick preview
out_png = out_path.replace(".pdf", ".png")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved: {out_png}")
