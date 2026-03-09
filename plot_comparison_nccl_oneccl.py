#!/usr/bin/env python3
"""
Genera grafici di confronto NCCL vs oneCCL per ogni coppia (collettiva, dtype).

Usage:
    python scripts/python/plot_comparison_nccl_oneccl.py
    python scripts/python/plot_comparison_nccl_oneccl.py --results-dir results/leonardo/4_rank --out-dir plots/leonardo
"""
import os
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size":         70,
    "axes.labelsize":    76,
    "xtick.labelsize":   66,
    "ytick.labelsize":   66,
    "legend.fontsize":   66,
    "xtick.major.pad":   16,
    "ytick.major.pad":   16,
    "xtick.major.size":  16,
    "ytick.major.size":  16,
    "xtick.major.width":  3,
    "ytick.major.width":  3,
    "axes.linewidth":     3,
    "axes.labelpad":     20,
    "legend.framealpha": 0.9,
    "lines.linewidth":    6,
    "lines.markersize":  28,
})

LIBRARY_STYLE = {
    "nccl":   {"color": "#2ca02c", "marker": "s", "label": "NCCL"},
    "oneccl": {"color": "#1f77b4", "marker": "o", "label": "oneCCL"},
}

# Fattore alpha per il bus bandwidth: busbw = alpha * algbw
# algbw = size / time,  busbw = alpha * size / time
BUS_ALPHA = {
    "allreduce":     lambda n: 2 * (n - 1) / n,
    "alltoall":      lambda n: (n - 1) / n,
    "allgather":     lambda n: (n - 1) / n,
    "reducescatter": lambda n: (n - 1) / n,
    "broadcast":     lambda n: (n - 1) / n,
    "reduce":        lambda n: (n - 1) / n,
}


def _fmt_size(b: int) -> str:
    for unit, thresh in [("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)]:
        if b >= thresh and b % thresh == 0:
            return f"{b // thresh}{unit}"
    return f"{b}B"


def load_all_results(results_dir: str) -> pd.DataFrame:
    """Carica tutti i CSV da results_dir/{nccl,oneccl}/{collective}/."""
    frames = []
    for library in ("nccl", "oneccl"):
        lib_dir = os.path.join(results_dir, library)
        if not os.path.isdir(lib_dir):
            continue
        for collective in os.listdir(lib_dir):
            coll_dir = os.path.join(lib_dir, collective)
            if not os.path.isdir(coll_dir):
                continue
            for csv_path in glob.glob(os.path.join(coll_dir, "*_results.csv")):
                try:
                    df = pd.read_csv(csv_path)
                    frames.append(df)
                except Exception as e:
                    print(f"Warning: impossibile leggere {csv_path}: {e}")
    if not frames:
        print(f"Nessun file CSV trovato in {results_dir}")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _mad(series: pd.Series) -> float:
    med = series.median()
    return float(np.median(np.abs(series - med)))


def _make_plot(grouped: pd.DataFrame, metric: str, ylabel: str,
               collective: str, dtype: str, suffix: str, out_dir: str,
               formats: list[str] | None = None, yscale: str = "log"):
    fig, ax = plt.subplots(figsize=(32, 20))

    for library, style in LIBRARY_STYLE.items():
        lib_data = grouped[grouped["library"] == library].sort_values("message_size_bytes")
        if lib_data.empty:
            continue
        ax.errorbar(
            lib_data["message_size_bytes"],
            lib_data["value"],
            yerr=lib_data["mad"],
            color=style["color"],
            marker=style["marker"],
            label=style["label"],
            linewidth=6,
            markersize=24,
            capsize=10,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale(yscale)

    present_sizes = sorted(grouped["message_size_bytes"].unique())
    ax.set_xticks(present_sizes)
    ax.set_xticklabels(
        [_fmt_size(int(s)) for s in present_sizes],
        rotation=45, ha="right", rotation_mode="anchor",
    )

    ax.set_xlabel("Message size")
    ax.set_ylabel(ylabel)
    ax.legend()
    grid_which = "both" if yscale == "log" else "major"
    ax.grid(True, which=grid_which, linestyle="--", alpha=0.4)

    fig.tight_layout()
    for fmt in (formats or ["png"]):
        out_path = os.path.join(out_dir, f"{collective}_{dtype}_{suffix}.{fmt}")
        kwargs = {"dpi": 150} if fmt != "pdf" else {}
        fig.savefig(out_path, **kwargs)
        print(f"Salvato: {out_path}")
    plt.close(fig)


def plot_comparison(df: pd.DataFrame, collective: str, dtype: str, out_dir: str,
                    formats: list[str] | None = None, yscale: str = "log",
                    goodput_mode: str = "algbw"):
    subset = df[(df["collective"] == collective) & (df["data_type"] == dtype)].copy()
    if subset.empty:
        return

    # Per ogni run, prendi il max time_ms tra i rank (= tempo reale della collettiva)
    # num_ranks incluso nel groupby per poter calcolare busbw
    per_run = (
        subset
        .groupby(["library", "message_size_bytes", "run_id", "num_ranks"])["time_ms"]
        .max()
        .reset_index()
    )

    per_run["algbw"] = (per_run["message_size_bytes"] / per_run["num_ranks"]) * 8 / (per_run["time_ms"] * 1e6)

    if goodput_mode in ("busbw", "both"):
        alpha_fn = BUS_ALPHA.get(collective, lambda n: 1.0)
        per_run["busbw"] = per_run["algbw"] * per_run["num_ranks"].apply(alpha_fn)

    grp_time = (
        per_run
        .groupby(["library", "message_size_bytes"])["time_ms"]
        .agg(value="median", mad=_mad)
        .reset_index()
    )
    _make_plot(grp_time, "time_ms", "Time (ms)",
               collective, dtype, "time", out_dir, formats, yscale)

    if goodput_mode in ("algbw", "both"):
        suffix  = "goodput_algbw" if goodput_mode == "both" else "goodput"
        grp = (
            per_run
            .groupby(["library", "message_size_bytes"])["algbw"]
            .agg(value="median", mad=_mad)
            .reset_index()
        )
        _make_plot(grp, "algbw", "Goodput algbw (Gb/s)",
                   collective, dtype, suffix, out_dir, formats, yscale)

    if goodput_mode in ("busbw", "both"):
        suffix = "goodput_busbw" if goodput_mode == "both" else "goodput"
        grp = (
            per_run
            .groupby(["library", "message_size_bytes"])["busbw"]
            .agg(value="median", mad=_mad)
            .reset_index()
        )
        _make_plot(grp, "busbw", "Goodput busbw (Gb/s)",
                   collective, dtype, suffix, out_dir, formats, yscale)


def main():
    parser = argparse.ArgumentParser(
        description="Genera grafici di confronto NCCL vs oneCCL per collettiva e dtype."
    )
    parser.add_argument(
        "--results-dir",
        default="results/leonardo/4_rank",
        help="Directory radice dei risultati (default: results/leonardo/4_rank)",
    )
    parser.add_argument(
        "--out-dir",
        default="plots/leonardo/nccl_vs_oneccl",
        help="Directory di output per i grafici (default: plots/leonardo/nccl_vs_oneccl)",
    )
    parser.add_argument(
        "--format",
        dest="formats",
        nargs="+",
        default=["png"],
        choices=["png", "pdf", "svg"],
        help="Formato/i di output (default: png)",
    )
    parser.add_argument(
        "--yscale",
        default="log",
        choices=["log", "linear"],
        help="Scala dell'asse Y (default: log)",
    )
    parser.add_argument(
        "--goodput-mode",
        dest="goodput_mode",
        default="algbw",
        choices=["algbw", "busbw", "both"],
        help="Metrica goodput: algbw (size/time), busbw (alpha*size/time), both (default: algbw)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_all_results(args.results_dir)
    if df.empty:
        return

    collectives = sorted(df["collective"].unique())
    dtypes = sorted(df["data_type"].unique())

    for collective in collectives:
        for dtype in dtypes:
            plot_comparison(df, collective, dtype, args.out_dir, args.formats, args.yscale, args.goodput_mode)

    n_pairs = len(collectives) * len(dtypes)
    print(f"\nCompletato. {n_pairs} coppie, {n_pairs * 2} grafici in {args.out_dir}/")


if __name__ == "__main__":
    main()
