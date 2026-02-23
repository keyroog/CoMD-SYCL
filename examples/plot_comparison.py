#!/usr/bin/env python3
# Run with: /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 plot_comparison.py
"""
CoMD benchmark comparison: CUDA+MPI vs SYCL+oneCCL
=====================================================
Parses YAML output files in strong-scaling/ and weak-scaling/,
extracts per-timer data, and produces comparison plots.

Usage:
    python plot_comparison.py

Output:
    comd_comparison.pdf  (or .png if matplotlib PDF backend is unavailable)

Timer legend
------------
  loop        : total simulation loop time (primary metric)
  force       : GPU force kernel
  redistribute: atom link-cell redistribution (NOTE: SYCL values include
                async GPU accounting artefacts — PercentLoop > 100%,
                use with caution)
  atomHalo    : atom ghost-cell exchange (MPI point-to-point, both versions)
  eamHalo     : EAM force halo exchange  (MPI point-to-point, both versions)
  commHalo    : low-level MPI sendrecv inside halo exchange
  commReduce  : allreduce for energy sums  ← oneCCL (SYCL) vs MPI (CUDA)
"""

import re
import os
import glob
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent.resolve()
STRONG_DIR = SCRIPT_DIR / "strong-scaling"
WEAK_DIR   = SCRIPT_DIR / "weak-scaling"
OUTPUT_PDF = SCRIPT_DIR / "comd_comparison.pdf"

TIMERS_OF_INTEREST = [
    "loop", "force", "redistribute", "atomHalo",
    "eamHalo", "commHalo", "commReduce",
]

# Colours
C_CUDA = "#2196F3"   # blue  — CUDA+MPI
C_SYCL = "#FF5722"   # orange — SYCL+oneCCL

# ---------------------------------------------------------------------------
# YAML parser
# ---------------------------------------------------------------------------

def parse_yaml(path: Path) -> dict:
    """
    Parse a CoMD YAML output file.

    Returns a dict with keys:
        variant     : "cuda" | "sycl"
        ranks       : int
        atoms       : int
        atom_rate   : float   atoms/μs  (global)
        loop_time   : float   s
        init_time   : float   s  (total - loop)
        timers      : dict[name -> {"total": float, "avg": float,
                                     "min": float, "max": float,
                                     "percent_loop": float}]
    """
    text = path.read_text(errors="replace")

    def _find(pattern, cast=str, default=None):
        m = re.search(pattern, text)
        return cast(m.group(1).strip()) if m else default

    name    = _find(r"Mini-Application Name\s*:\s*(.+)")
    ranks   = _find(r"TotalRanks\s*:\s*(\d+)", int, 1)
    atoms   = _find(r"Total atoms\s*:\s*(\d+)", int, 0)
    rate    = _find(r"AtomRate:\s*\n\s*AverageRate:\s*([\d.]+)", float, 0.0)

    # Determine variant
    if name and "sycl" in name.lower():
        variant = "sycl"
    else:
        variant = "cuda"

    # Parse timer blocks from "Performance Results For Rank 0" section
    # We take Rank-0 values for single-rank runs and the AvgTime for multi-rank.
    timers: dict = {}

    # --- Rank-0 individual timers (Total + PercentLoop) ---
    rank0_section = re.search(
        r"Performance Results For Rank 0:(.*?)Performance Results Across Ranks:",
        text, re.DOTALL
    )
    if rank0_section:
        block = rank0_section.group(1)
        for m in re.finditer(
            r"Timer:\s+(\w+)\s+CallCount:\s+\d+\s+AvgPerCall:\s+([\d.]+)\s+"
            r"Total:\s+([\d.]+)\s+PercentLoop:\s+([\d.]+)",
            block
        ):
            tname, avg_per, total, pct = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))
            timers.setdefault(tname, {})["total"]        = total
            timers.setdefault(tname, {})["avg_per_call"] = avg_per
            timers.setdefault(tname, {})["percent_loop"] = pct

    # --- Across-Ranks section: grab AvgTime, MinTime, MaxTime ---
    across_section = re.search(
        r"Performance Results Across Ranks:(.*?)Performance Global",
        text, re.DOTALL
    )
    if across_section:
        block = across_section.group(1)
        for m in re.finditer(
            r"Timer:\s+(\w+)\s+MinRank:\s+\d+\s+MinTime:\s+([\d.]+)\s+"
            r"MaxRank:\s+\d+\s+MaxTime:\s+([\d.]+)\s+AvgTime:\s+([\d.]+)\s+"
            r"StdevTime:\s+([\d.]+)",
            block
        ):
            tname = m.group(1)
            timers.setdefault(tname, {})["min"]   = float(m.group(2))
            timers.setdefault(tname, {})["max"]   = float(m.group(3))
            timers.setdefault(tname, {})["avg"]   = float(m.group(4))
            timers.setdefault(tname, {})["stdev"] = float(m.group(5))

    loop_time = timers.get("loop", {}).get("avg", timers.get("loop", {}).get("total", 0.0))
    total_time = timers.get("total", {}).get("avg", timers.get("total", {}).get("total", 0.0))
    init_time  = total_time - loop_time

    return {
        "variant":   variant,
        "ranks":     ranks,
        "atoms":     atoms,
        "atom_rate": rate,
        "loop_time": loop_time,
        "init_time": init_time,
        "timers":    timers,
    }


def load_directory(directory: Path) -> dict:
    """
    Load all YAML files in *directory*.

    Returns:
        {
          "cuda": {ranks: record, ...},
          "sycl": {ranks: record, ...},
        }
    """
    data = {"cuda": {}, "sycl": {}}
    for path in sorted(directory.glob("*.yaml")):
        try:
            rec = parse_yaml(path)
            variant = rec["variant"]
            ranks   = rec["ranks"]
            if ranks in data[variant]:
                warnings.warn(
                    f"Duplicate entry for {variant} {ranks} ranks in {directory.name}; "
                    f"keeping {path.name}"
                )
            data[variant][ranks] = rec
        except Exception as exc:
            warnings.warn(f"Could not parse {path.name}: {exc}")
    return data


def _timer_avg(rec: dict, name: str) -> float:
    """Return AvgTime across ranks (or Rank-0 total if only one rank)."""
    t = rec["timers"].get(name, {})
    return t.get("avg", t.get("total", 0.0))


def _timer_stdev(rec: dict, name: str) -> float:
    t = rec["timers"].get(name, {})
    return t.get("stdev", 0.0)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _sorted_ranks(d: dict) -> list[int]:
    return sorted(d.keys())


def _bar_positions(ranks: list[int], width: float, offset: float) -> np.ndarray:
    x = np.arange(len(ranks))
    return x + offset * width


def annotate_anomaly(ax, x, y, text="⚠", fontsize=8, color="red"):
    ax.annotate(text, xy=(x, y), xytext=(x, y * 1.08),
                ha="center", fontsize=fontsize, color=color,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8))


# ---------------------------------------------------------------------------
# Figure 1 — Strong scaling overview
# ---------------------------------------------------------------------------

def fig_strong_scaling(strong: dict, ax_loop, ax_rate, ax_reduce, ax_breakdown):
    cuda = strong["cuda"]
    sycl = strong["sycl"]

    all_ranks = sorted(set(_sorted_ranks(cuda)) | set(_sorted_ranks(sycl)))

    def _get(d, ranks_list, key, func=_timer_avg):
        return [func(d[r], key) if r in d else np.nan for r in ranks_list]

    # ---- (A) Loop time ----
    cuda_loop  = [cuda[r]["loop_time"] if r in cuda else np.nan for r in all_ranks]
    sycl_loop  = [sycl[r]["loop_time"] if r in sycl else np.nan for r in all_ranks]

    ax_loop.plot(all_ranks, cuda_loop, "o-", color=C_CUDA, label="CUDA + MPI", lw=2, ms=7)
    ax_loop.plot(all_ranks, sycl_loop, "s--", color=C_SYCL, label="SYCL + oneCCL", lw=2, ms=7)
    # ideal scaling reference
    ref_loop = cuda_loop[0]
    ideal_x  = np.array(all_ranks)
    ax_loop.plot(ideal_x, ref_loop / ideal_x * all_ranks[0],
                 "k:", lw=1, label="Ideal scaling")
    ax_loop.set_xscale("log", base=2)
    ax_loop.set_yscale("log")
    ax_loop.set_xlabel("Number of ranks (GPUs)")
    ax_loop.set_ylabel("Loop time (s)  ↓ better")
    ax_loop.set_title("(A) Strong scaling — loop time (256 K atoms)")
    ax_loop.set_xticks(all_ranks)
    ax_loop.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax_loop.legend(fontsize=8)
    ax_loop.grid(True, which="both", ls=":", alpha=0.5)

    # ---- (B) Atom rate ----
    cuda_rate = [cuda[r]["atom_rate"] if r in cuda else np.nan for r in all_ranks]
    sycl_rate = [sycl[r]["atom_rate"] if r in sycl else np.nan for r in all_ranks]

    ax_rate.plot(all_ranks, cuda_rate, "o-", color=C_CUDA, label="CUDA + MPI", lw=2, ms=7)
    ax_rate.plot(all_ranks, sycl_rate, "s--", color=C_SYCL, label="SYCL + oneCCL", lw=2, ms=7)
    ref_rate = cuda_rate[0]
    ax_rate.plot(ideal_x, ref_rate * ideal_x / all_ranks[0],
                 "k:", lw=1, label="Ideal scaling")
    ax_rate.set_xscale("log", base=2)
    ax_rate.set_xlabel("Number of ranks (GPUs)")
    ax_rate.set_ylabel("Atom update rate (atoms/μs)  ↑ better")
    ax_rate.set_title("(B) Strong scaling — throughput")
    ax_rate.set_xticks(all_ranks)
    ax_rate.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax_rate.legend(fontsize=8)
    ax_rate.grid(True, which="both", ls=":", alpha=0.5)

    # ---- (C) commReduce (collective) overhead ----
    cuda_red  = _get(cuda, all_ranks, "commReduce")
    sycl_red  = _get(sycl, all_ranks, "commReduce")
    cuda_red_e = [_timer_stdev(cuda[r], "commReduce") if r in cuda else 0 for r in all_ranks]
    sycl_red_e = [_timer_stdev(sycl[r], "commReduce") if r in sycl else 0 for r in all_ranks]

    ax_reduce.errorbar(all_ranks, cuda_red, yerr=cuda_red_e,
                       fmt="o-", color=C_CUDA, label="CUDA + MPI (MPI_Allreduce)", lw=2, ms=7, capsize=4)
    ax_reduce.errorbar(all_ranks, sycl_red, yerr=sycl_red_e,
                       fmt="s--", color=C_SYCL, label="SYCL + oneCCL (ccl::allreduce)", lw=2, ms=7, capsize=4)
    ax_reduce.set_xscale("log", base=2)
    ax_reduce.set_xlabel("Number of ranks (GPUs)")
    ax_reduce.set_ylabel("commReduce time (s)  ↓ better")
    ax_reduce.set_title("(C) Collective (allreduce) time — oneCCL vs MPI")
    ax_reduce.set_xticks(all_ranks)
    ax_reduce.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax_reduce.legend(fontsize=8)
    ax_reduce.grid(True, which="both", ls=":", alpha=0.5)

    # ---- (D) Stacked bar — timer breakdown ----
    bar_timers = ["force", "atomHalo", "eamHalo", "commHalo", "commReduce"]
    bar_colors = ["#4CAF50", "#9C27B0", "#FF9800", "#607D8B", "#F44336"]
    bar_labels = ["force (GPU)", "atomHalo (MPI p2p)", "eamHalo (MPI p2p)",
                  "commHalo (MPI p2p)", "commReduce (collective)"]

    width = 0.35
    x = np.arange(len(all_ranks))

    cuda_bottoms = np.zeros(len(all_ranks))
    sycl_bottoms = np.zeros(len(all_ranks))

    for i, (tname, col, lab) in enumerate(zip(bar_timers, bar_colors, bar_labels)):
        cuda_vals = np.array(_get(cuda, all_ranks, tname))
        sycl_vals = np.array(_get(sycl, all_ranks, tname))

        kw_label = dict(label=lab, color=col) if i == 0 else dict(color=col)
        # CUDA bars (left)
        ax_breakdown.bar(x - width/2, cuda_vals, width,
                         bottom=cuda_bottoms, **kw_label,
                         hatch="", alpha=0.85)
        # SYCL bars (right) — hatched
        ax_breakdown.bar(x + width/2, sycl_vals, width,
                         bottom=sycl_bottoms, color=col,
                         hatch="///", alpha=0.65)
        np.add.at(cuda_bottoms, range(len(all_ranks)), np.nan_to_num(cuda_vals))
        np.add.at(sycl_bottoms, range(len(all_ranks)), np.nan_to_num(sycl_vals))

    # Legend for bars (dummy patches)
    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(color=col, label=lab)
                      for col, lab in zip(bar_colors, bar_labels)]
    cuda_patch = mpatches.Patch(facecolor="gray", label="solid = CUDA+MPI")
    sycl_patch = mpatches.Patch(facecolor="gray", hatch="///", label="hatched = SYCL+oneCCL")
    ax_breakdown.legend(handles=legend_patches + [cuda_patch, sycl_patch],
                        fontsize=7, loc="upper left", ncol=2)

    ax_breakdown.set_xticks(x)
    ax_breakdown.set_xticklabels([f"{r}R" for r in all_ranks])
    ax_breakdown.set_xlabel("Number of ranks")
    ax_breakdown.set_ylabel("Time (s)")
    ax_breakdown.set_title("(D) Timer breakdown — strong scaling (AvgTime across ranks)")
    ax_breakdown.grid(True, axis="y", ls=":", alpha=0.5)


# ---------------------------------------------------------------------------
# Figure 2 — Weak scaling overview
# ---------------------------------------------------------------------------

def fig_weak_scaling(weak: dict, ax_loop, ax_rate, ax_reduce, ax_init):
    cuda = weak["cuda"]
    sycl = weak["sycl"]

    all_ranks = sorted(set(_sorted_ranks(cuda)) | set(_sorted_ranks(sycl)))

    cuda_loop  = [cuda[r]["loop_time"]  if r in cuda else np.nan for r in all_ranks]
    sycl_loop  = [sycl[r]["loop_time"]  if r in sycl else np.nan for r in all_ranks]
    cuda_rate  = [cuda[r]["atom_rate"]  if r in cuda else np.nan for r in all_ranks]
    sycl_rate  = [sycl[r]["atom_rate"]  if r in sycl else np.nan for r in all_ranks]

    def _get(d, ranks_list, key):
        return [_timer_avg(d[r], key) if r in d else np.nan for r in ranks_list]

    # ---- (E) Loop time — ideal is flat ----
    ax_loop.axhline(cuda_loop[0], color=C_CUDA, ls=":", lw=1, label="Ideal (CUDA base)")
    ax_loop.axhline(sycl_loop[0], color=C_SYCL, ls=":", lw=1, label="Ideal (SYCL base)")
    ax_loop.plot(all_ranks, cuda_loop, "o-", color=C_CUDA, label="CUDA + MPI", lw=2, ms=7)
    ax_loop.plot(all_ranks, sycl_loop, "s--", color=C_SYCL, label="SYCL + oneCCL", lw=2, ms=7)
    ax_loop.set_xscale("log", base=2)
    ax_loop.set_xlabel("Number of ranks (GPUs)")
    ax_loop.set_ylabel("Loop time (s)  — ideal = flat")
    ax_loop.set_title("(E) Weak scaling — loop time")
    ax_loop.set_xticks(all_ranks)
    ax_loop.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax_loop.legend(fontsize=8)
    ax_loop.grid(True, which="both", ls=":", alpha=0.5)
    # Annotate non-isogranular point
    if 2 in all_ranks:
        idx = all_ranks.index(2)
        ax_loop.annotate("64K atoms/rank\n(non-isogranular)",
                         xy=(2, max(cuda_loop[idx] or 0, sycl_loop[idx] or 0)),
                         xytext=(2.2, max(cuda_loop[idx] or 0, sycl_loop[idx] or 0) * 1.1),
                         fontsize=7, color="gray",
                         arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    # ---- (F) Atom rate ----
    ax_rate.plot(all_ranks, cuda_rate, "o-", color=C_CUDA, label="CUDA + MPI", lw=2, ms=7)
    ax_rate.plot(all_ranks, sycl_rate, "s--", color=C_SYCL, label="SYCL + oneCCL", lw=2, ms=7)
    ax_rate.set_xscale("log", base=2)
    ax_rate.set_xlabel("Number of ranks (GPUs)")
    ax_rate.set_ylabel("Atom update rate (atoms/μs)")
    ax_rate.set_title("(F) Weak scaling — throughput")
    ax_rate.set_xticks(all_ranks)
    ax_rate.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax_rate.legend(fontsize=8)
    ax_rate.grid(True, which="both", ls=":", alpha=0.5)

    # ---- (G) commReduce ----
    cuda_red = _get(cuda, all_ranks, "commReduce")
    sycl_red = _get(sycl, all_ranks, "commReduce")
    cuda_red_e = [_timer_stdev(cuda[r], "commReduce") if r in cuda else 0 for r in all_ranks]
    sycl_red_e = [_timer_stdev(sycl[r], "commReduce") if r in sycl else 0 for r in all_ranks]

    ax_reduce.errorbar(all_ranks, cuda_red, yerr=cuda_red_e,
                       fmt="o-", color=C_CUDA, label="CUDA + MPI", lw=2, ms=7, capsize=4)
    ax_reduce.errorbar(all_ranks, sycl_red, yerr=sycl_red_e,
                       fmt="s--", color=C_SYCL, label="SYCL + oneCCL", lw=2, ms=7, capsize=4)
    ax_reduce.set_xscale("log", base=2)
    ax_reduce.set_xlabel("Number of ranks (GPUs)")
    ax_reduce.set_ylabel("commReduce time (s)")
    ax_reduce.set_title("(G) Collective overhead — weak scaling")
    ax_reduce.set_xticks(all_ranks)
    ax_reduce.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax_reduce.legend(fontsize=8)
    ax_reduce.grid(True, which="both", ls=":", alpha=0.5)

    # ---- (H) Initialisation overhead (total - loop) ----
    cuda_init = [cuda[r]["init_time"] if r in cuda else np.nan for r in all_ranks]
    sycl_init = [sycl[r]["init_time"] if r in sycl else np.nan for r in all_ranks]

    ax_init.plot(all_ranks, cuda_init, "o-", color=C_CUDA, label="CUDA + MPI", lw=2, ms=7)
    ax_init.plot(all_ranks, sycl_init, "s--", color=C_SYCL, label="SYCL + oneCCL", lw=2, ms=7)
    ax_init.set_xscale("log", base=2)
    ax_init.set_xlabel("Number of ranks (GPUs)")
    ax_init.set_ylabel("Init time (total − loop, s)")
    ax_init.set_title("(H) Startup overhead (SYCL/NCCL init)")
    ax_init.set_xticks(all_ranks)
    ax_init.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax_init.legend(fontsize=8)
    ax_init.grid(True, which="both", ls=":", alpha=0.5)


# ---------------------------------------------------------------------------
# Figure 3 — Collective deep-dive (both strong + weak)
# ---------------------------------------------------------------------------

def fig_collective_deepdive(strong: dict, weak: dict, axes):
    """
    Two rows × two cols:
      (I)  strong: commReduce absolute + as % of loop
      (J)  weak:   commReduce absolute + as % of loop
    """
    ax_s_abs, ax_s_pct, ax_w_abs, ax_w_pct = axes

    for label, data, (ax_abs, ax_pct) in [
        ("Strong scaling (256 K atoms fixed)", strong, (ax_s_abs, ax_s_pct)),
        ("Weak scaling (~32 K atoms/rank)",    weak,   (ax_w_abs, ax_w_pct)),
    ]:
        cuda = data["cuda"]
        sycl = data["sycl"]
        all_ranks = sorted(set(_sorted_ranks(cuda)) | set(_sorted_ranks(sycl)))

        cuda_red  = [_timer_avg(cuda[r], "commReduce") if r in cuda else np.nan for r in all_ranks]
        sycl_red  = [_timer_avg(sycl[r], "commReduce") if r in sycl else np.nan for r in all_ranks]
        cuda_loop = [cuda[r]["loop_time"] if r in cuda else np.nan for r in all_ranks]
        sycl_loop = [sycl[r]["loop_time"] if r in sycl else np.nan for r in all_ranks]

        cuda_pct = [cr / cl * 100 if cl else np.nan
                    for cr, cl in zip(cuda_red, cuda_loop)]
        sycl_pct = [cr / cl * 100 if cl else np.nan
                    for cr, cl in zip(sycl_red, sycl_loop)]

        ax_abs.plot(all_ranks, cuda_red, "o-", color=C_CUDA, lw=2, ms=7, label="CUDA + MPI")
        ax_abs.plot(all_ranks, sycl_red, "s--", color=C_SYCL, lw=2, ms=7, label="SYCL + oneCCL")
        ax_abs.set_xscale("log", base=2)
        ax_abs.set_xlabel("Ranks")
        ax_abs.set_ylabel("commReduce (s)")
        ax_abs.set_title(f"Collective time — {label}")
        ax_abs.set_xticks(all_ranks)
        ax_abs.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax_abs.legend(fontsize=8)
        ax_abs.grid(True, which="both", ls=":", alpha=0.5)

        ax_pct.plot(all_ranks, cuda_pct, "o-", color=C_CUDA, lw=2, ms=7, label="CUDA + MPI")
        ax_pct.plot(all_ranks, sycl_pct, "s--", color=C_SYCL, lw=2, ms=7, label="SYCL + oneCCL")
        ax_pct.set_xscale("log", base=2)
        ax_pct.set_xlabel("Ranks")
        ax_pct.set_ylabel("commReduce / loop  (%)")
        ax_pct.set_title(f"Collective % of loop — {label}")
        ax_pct.set_xticks(all_ranks)
        ax_pct.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax_pct.legend(fontsize=8)
        ax_pct.grid(True, which="both", ls=":", alpha=0.5)


# ---------------------------------------------------------------------------
# Summary table (printed to stdout)
# ---------------------------------------------------------------------------

def print_summary_table(strong: dict, weak: dict):
    print("\n" + "="*80)
    print("STRONG SCALING SUMMARY (256 K atoms)")
    print("="*80)
    print(f"{'Ranks':>6}  {'Version':>14}  {'loop(s)':>9}  {'atoms/μs':>9}  "
          f"{'commReduce(s)':>14}  {'commReduce%':>12}  {'init(s)':>8}")
    print("-"*80)
    for variant, name, data in [("cuda","CUDA+MPI",strong["cuda"]),
                                  ("sycl","SYCL+oneCCL",strong["sycl"])]:
        for r in sorted(data):
            rec  = data[r]
            loop = rec["loop_time"]
            red  = _timer_avg(rec, "commReduce")
            pct  = red / loop * 100 if loop else 0
            print(f"{r:>6}  {name:>14}  {loop:>9.4f}  {rec['atom_rate']:>9.2f}  "
                  f"{red:>14.4f}  {pct:>11.1f}%  {rec['init_time']:>8.2f}")

    print("\n" + "="*80)
    print("WEAK SCALING SUMMARY")
    print("="*80)
    print(f"{'Ranks':>6}  {'Version':>14}  {'atoms':>8}  {'loop(s)':>9}  "
          f"{'atoms/μs':>9}  {'commReduce(s)':>14}  {'commReduce%':>12}")
    print("-"*80)
    for variant, name, data in [("cuda","CUDA+MPI",weak["cuda"]),
                                  ("sycl","SYCL+oneCCL",weak["sycl"])]:
        for r in sorted(data):
            rec  = data[r]
            loop = rec["loop_time"]
            red  = _timer_avg(rec, "commReduce")
            pct  = red / loop * 100 if loop else 0
            print(f"{r:>6}  {name:>14}  {rec['atoms']:>8}  {loop:>9.4f}  "
                  f"{rec['atom_rate']:>9.2f}  {red:>14.4f}  {pct:>11.1f}%")

    print("\nNOTES:")
    print("  • commReduce = allreduce timer (oneCCL in SYCL, MPI in CUDA).")
    print("  • CUDA compiled with -O5, SYCL with -O2 — host-code opt differs.")
    print("  • 'redistribute' PercentLoop > 100% in SYCL is a GPU-async timing")
    print("    artefact; do not interpret as real simulation time.")
    print("  • Weak scaling at 2 ranks uses 64K atoms/rank (non-isogranular).")
    print("  • init = total - loop; SYCL incurs extra SYCL/NCCL init overhead.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading strong-scaling data from {STRONG_DIR}")
    strong = load_directory(STRONG_DIR)
    print(f"  CUDA: {sorted(strong['cuda'].keys())} ranks")
    print(f"  SYCL: {sorted(strong['sycl'].keys())} ranks")

    print(f"Loading weak-scaling data from {WEAK_DIR}")
    weak = load_directory(WEAK_DIR)
    print(f"  CUDA: {sorted(weak['cuda'].keys())} ranks")
    print(f"  SYCL: {sorted(weak['sycl'].keys())} ranks")

    print_summary_table(strong, weak)

    # ---- Build figures ----
    # Figure 1 — Strong scaling
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle(
        "CoMD: CUDA+MPI vs SYCL+oneCCL — Strong Scaling (EAM, 256 K atoms, thread_atom)",
        fontsize=13, fontweight="bold"
    )
    fig_strong_scaling(strong, axes1[0, 0], axes1[0, 1], axes1[1, 0], axes1[1, 1])
    fig1.tight_layout(rect=[0, 0, 1, 0.96])

    # Figure 2 — Weak scaling
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle(
        "CoMD: CUDA+MPI vs SYCL+oneCCL — Weak Scaling (EAM, ~32 K atoms/rank, thread_atom)",
        fontsize=13, fontweight="bold"
    )
    fig_weak_scaling(weak, axes2[0, 0], axes2[0, 1], axes2[1, 0], axes2[1, 1])
    fig2.tight_layout(rect=[0, 0, 1, 0.96])

    # Figure 3 — Collective deep-dive
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
    fig3.suptitle(
        "CoMD: oneCCL vs MPI collective overhead — allreduce deep-dive",
        fontsize=13, fontweight="bold"
    )
    fig_collective_deepdive(strong, weak,
                             [axes3[0, 0], axes3[0, 1], axes3[1, 0], axes3[1, 1]])
    fig3.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(OUTPUT_PDF) as pdf:
        pdf.savefig(fig1)
        pdf.savefig(fig2)
        pdf.savefig(fig3)
        # PDF metadata
        d = pdf.infodict()
        d["Title"]   = "CoMD SYCL+oneCCL vs CUDA+MPI benchmark"
        d["Subject"] = "Strong and weak scaling, collective overhead analysis"

    print(f"\nSaved → {OUTPUT_PDF}")

    plt.close("all")


if __name__ == "__main__":
    main()
