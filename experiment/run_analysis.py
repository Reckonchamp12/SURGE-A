#!/usr/bin/env python3
"""
Dataset analysis for SURGE-A.

Generates 7 diagnostic figures (time series, distributions, ACF, FFT,
stationarity, correlation heatmaps, SURGE-A readiness) and a summary CSV.

Usage
-----
    python experiments/run_analysis.py [--out results/] [--kaggle-input /kaggle/input]
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from surge.data import load_all
from surge.sampling import effective_rank, adaptive_rate

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold",
})

TARGET = {
    "ETTh1":   "OT",
    "ETTm1":   "OT",
    "Weather": "temperature_2m",
    "Jena":    "T (degC)",
    "Store":   "sales",
}
COLORS = {
    "ETTh1":   "#378ADD",
    "ETTm1":   "#1D9E75",
    "Weather": "#D85A30",
    "Jena":    "#7F77DD",
    "Store":   "#D4537E",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out",          default="results")
    p.add_argument("--kaggle-input", default="/kaggle/input")
    return p.parse_args()


def acf(series, max_lag=72):
    s   = np.asarray(series.dropna(), dtype=float)
    s   = s - s.mean()
    var = np.var(s) + 1e-12
    return np.array([1.0] + [np.mean(s[l:] * s[:-l]) / var for l in range(1, max_lag + 1)])


def top_periods(series, top_n=5):
    s    = np.asarray(series.dropna(), dtype=float)
    n    = len(s)
    pw   = np.abs(np.fft.rfft(s - s.mean())) ** 2
    freq = np.fft.rfftfreq(n)
    pw[0] = 0
    idx  = np.argsort(pw)[-top_n:][::-1]
    return [round(1 / freq[i], 1) if freq[i] > 0 else np.inf for i in idx]


def main():
    args = parse_args()
    OUT  = Path(args.out) / "figures"
    OUT.mkdir(parents=True, exist_ok=True)

    print("LOADING DATASETS")
    datasets_raw = load_all(args.kaggle_input)

    # Build DataFrame-based datasets dict for plotting
    DATASETS = {}
    for name, s in datasets_raw.items():
        col = TARGET.get(name, "value")
        DATASETS[name] = pd.DataFrame({col: s.values})

    print("\nCOMPUTING STATISTICS")
    stats, acf_data, rank_data, period_data = {}, {}, {}, {}

    for name, df in DATASETS.items():
        col = TARGET[name]
        s   = df[col].dropna()

        stats[name] = dict(
            n=len(s), mean=s.mean(), std=s.std(),
            min=s.min(), p25=s.quantile(0.25), median=s.median(),
            p75=s.quantile(0.75), max=s.max(),
            skew=float(s.skew()), kurt=float(s.kurt()),
            nulls=int(df[col].isna().sum()),
            null_pct=round(df[col].isna().mean() * 100, 3),
            cols=df.shape[1],
        )
        acf_data[name]    = acf(s, max_lag=72)
        T                 = len(s)
        er                = effective_rank(s.values)
        p                 = adaptive_rate(T, er)
        rank_data[name]   = dict(T=T, eff_rank=er, p_rate=round(p, 4),
                                 k_sub=max(2, int(T * p)))
        period_data[name] = top_periods(s, top_n=5)
        print(f"  {name}: n={T:,}  er={er}  p={p:.4f}")

    names = list(DATASETS.keys())

    # ── Fig 1: time series ────────────────────────────────────────────
    fig, axes = plt.subplots(len(names), 1, figsize=(14, 3.5 * len(names)))
    fig.suptitle("Full Time Series — Target Variable", fontsize=12, fontweight="bold")
    for ax, name in zip(axes, names):
        col  = TARGET[name]
        s    = DATASETS[name][col].dropna()
        step = max(1, len(s) // 2000)
        ax.plot(s.values[::step], color=COLORS[name], linewidth=0.7, alpha=0.9)
        rm = s.rolling(max(1, len(s) // 40)).mean()
        ax.plot(rm.values[::step], color="black", linewidth=1.2, alpha=0.55,
                linestyle="--", label="rolling mean")
        st = stats[name]
        ax.set_title(f"{name}  n={st['n']:,}  μ={st['mean']:.2f}  "
                     f"σ={st['std']:.2f}  skew={st['skew']:.2f}")
        ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(OUT / "fig1_timeseries.png", dpi=130, bbox_inches="tight")
    plt.close(); print("  fig1_timeseries.png")

    # ── Fig 2: distributions ──────────────────────────────────────────
    nc  = 3
    nr  = (len(names) + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 4 * nr))
    fig.suptitle("Target Variable Distributions", fontsize=12, fontweight="bold")
    axes = axes.ravel()
    for i, name in enumerate(names):
        ax  = axes[i]
        col = TARGET[name]
        s   = DATASETS[name][col].dropna().values
        ax.hist(s, bins=60, color=COLORS[name], alpha=0.75, edgecolor="none", density=True)
        mu, sg = s.mean(), s.std()
        x  = np.linspace(s.min(), s.max(), 300)
        ax.plot(x, np.exp(-0.5 * ((x - mu) / sg) ** 2) / (sg * np.sqrt(2 * np.pi)),
                color="black", linewidth=1.5, linestyle="--", label="Gaussian")
        ax.axvline(mu, color="red", linewidth=1, linestyle=":")
        ax.set_title(f"{name}  skew={pd.Series(s).skew():.2f}  kurt={pd.Series(s).kurt():.2f}")
        ax.set_xlabel(col); ax.legend(fontsize=8)
    for ax in axes[len(names):]: ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "fig2_distributions.png", dpi=130, bbox_inches="tight")
    plt.close(); print("  fig2_distributions.png")

    # ── Fig 3: ACF ────────────────────────────────────────────────────
    fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 4 * nr))
    fig.suptitle("ACF (lags 0–72)", fontsize=12, fontweight="bold")
    axes = axes.ravel()
    CI = 1.96 / np.sqrt(5000)
    for i, name in enumerate(names):
        ax   = axes[i]
        vals = acf_data[name]
        ax.bar(range(len(vals)), vals, color=COLORS[name], alpha=0.75, width=0.85)
        ax.axhline( CI, color="red", linestyle="--", linewidth=0.9, label="95% CI")
        ax.axhline(-CI, color="red", linestyle="--", linewidth=0.9)
        ax.axhline( 0,  color="black", linewidth=0.5)
        ax.set_title(f"{name}  ACF(1)={vals[1]:.3f}")
        ax.set_ylim(-1.05, 1.05); ax.legend(fontsize=8)
    for ax in axes[len(names):]: ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "fig3_acf.png", dpi=130, bbox_inches="tight")
    plt.close(); print("  fig3_acf.png")

    # ── Fig 4: FFT ────────────────────────────────────────────────────
    fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 4 * nr))
    fig.suptitle("FFT Power Spectrum (log scale)", fontsize=12, fontweight="bold")
    axes = axes.ravel()
    for i, name in enumerate(names):
        ax   = axes[i]
        col  = TARGET[name]
        s    = DATASETS[name][col].dropna().values
        pw   = np.abs(np.fft.rfft(s - s.mean())) ** 2
        freq = np.fft.rfftfreq(len(s)); pw[0] = 0
        valid = freq > 0
        periods = 1 / freq[valid]; power = pw[valid]
        idx = np.argsort(periods)
        ax.semilogy(periods[idx], power[idx], color=COLORS[name], linewidth=0.9)
        ax.set_xlim(2, min(1000, periods.max()))
        top_p = periods[np.argmax(power)]
        ax.axvline(top_p, color="red", linestyle="--", linewidth=1.2,
                   label=f"peak={top_p:.0f}")
        ax.set_title(f"{name}"); ax.legend(fontsize=8)
    for ax in axes[len(names):]: ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "fig4_fft.png", dpi=130, bbox_inches="tight")
    plt.close(); print("  fig4_fft.png")

    # ── Fig 5: SURGE-A readiness ──────────────────────────────────────
    T_vals = [rank_data[n]["T"]        for n in names]
    er_v   = [rank_data[n]["eff_rank"] for n in names]
    p_vals = [rank_data[n]["p_rate"]   for n in names]
    k_vals = [rank_data[n]["k_sub"]    for n in names]
    colors = [COLORS[n] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("SURGE-A Benchmark Readiness", fontsize=12, fontweight="bold")
    for ax, ys, title, ylabel in zip(
        axes,
        [T_vals, er_v, p_vals],
        ["Series length (T)", "Effective rank (95%)", "Adaptive rate (p)"],
        ["samples", "rank", "rate"],
    ):
        bars = ax.bar(names, ys, color=colors, alpha=0.82, edgecolor="none")
        ax.set_title(title); ax.set_ylabel(ylabel)
        if ylabel == "samples": ax.set_yscale("log")
        for b, v in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.05 if ylabel == "samples" else v + 0.001,
                    f"{v:,}" if isinstance(v, int) else f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "fig5_surge_readiness.png", dpi=130, bbox_inches="tight")
    plt.close(); print("  fig5_surge_readiness.png")

    # ── Summary CSV ───────────────────────────────────────────────────
    rows = []
    for name in names:
        r = {"dataset": name, **stats[name], **rank_data[name],
             "top_period": period_data[name][0],
             "acf_lag1":   round(float(acf_data[name][1]), 3)}
        rows.append(r)
    summary_df = pd.DataFrame(rows).set_index("dataset")
    summary_df.to_csv(Path(args.out) / "dataset_summary.csv")
    print(f"\n✓ dataset_summary.csv saved")
    print(summary_df[["n", "mean", "std", "skew", "eff_rank", "p_rate", "acf_lag1"]].round(3).to_string())


if __name__ == "__main__":
    main()
