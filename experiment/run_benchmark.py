#!/usr/bin/env python3
"""
SURGE-A Benchmark — main experiment entry point.

Runs all 7 methods × all datasets × 3 seeds, then saves:
    results/surge_a_results.csv
    results/aggressive_rate_results.csv
    results/ablation_rates.csv
    results/figures/*.png

Usage
-----
    python experiments/run_benchmark.py [--out results/] [--seeds 0 1 2]
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from surge.benchmark import (
    LOOKBACK, METHODS, AGG_COLS,
    compute_rank_info, prepare, run_benchmark,
    make_t_idx,
)
from surge.conformal import ALPHA, GAMMA_FIXED, calc_metrics, conf_std, conf_vc
from surge.data import load_all
from surge.datasets import chirp_nonlinear, lorenz_series, mackey_glass
from surge.models import pred_lstm, train_lstm
from surge.sampling import adaptive_rate, effective_rank, u_idx

torch.set_num_threads(4)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out",   default="results", help="Output directory")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--kaggle-input", default="/kaggle/input")
    return p.parse_args()


def main():
    args   = parse_args()
    OUT    = Path(args.out)
    SEEDS  = args.seeds
    (OUT / "figures").mkdir(parents=True, exist_ok=True)

    RUN_START = time.time()
    print("=" * 65)
    print("SURGE-A BENCHMARK v4")
    print("=" * 65)

    # ── 1. Load datasets ──────────────────────────────────────────────
    print("\nLOADING DATASETS")
    print("=" * 65)
    raw = load_all(args.kaggle_input)

    # Add synthetic datasets
    raw["MackeyGlass"] = pd.Series(mackey_glass(n=12_000))
    raw["Chirp"]       = pd.Series(chirp_nonlinear(n=10_000))
    raw["HighRank"]    = pd.Series(lorenz_series(n=10_000))

    # ── 2. Prepare windows ───────────────────────────────────────────
    print("\nPREPARING WINDOWS")
    prepared = {}
    for name, s in tqdm(raw.items(), desc="Windowing", unit="ds", leave=False):
        prepared[name] = prepare(s, LOOKBACK)
        X_tr, _, X_va, _, X_te, *_ = prepared[name]
        tqdm.write(f"  {name:<12} tr={X_tr.shape}  va={X_va.shape}  te={X_te.shape}")

    # ── 3. Rank info ─────────────────────────────────────────────────
    rank_info = compute_rank_info(prepared)

    # ── 4. Main benchmark ────────────────────────────────────────────
    results = run_benchmark(prepared, rank_info, seeds=SEEDS)
    results.to_csv(OUT / "surge_a_results.csv", index=False)
    print(f"\n✓ Saved surge_a_results.csv  ({len(results)} rows)")

    # ── 5. Issue-1 experiment (aggressive rate p=0.05) ───────────────
    _run_aggressive_rate(prepared, rank_info, SEEDS, OUT)

    # ── 6. Ablation (ETTh1, adaptive vs fixed rates) ─────────────────
    _run_ablation(prepared, rank_info, SEEDS, OUT)

    # ── 7. Figures ───────────────────────────────────────────────────
    _make_figures(results, OUT)

    elapsed = time.time() - RUN_START
    m, s = divmod(int(elapsed), 60)
    print(f"\n✓ All done.  Total time: {m}m {s}s")
    print(f"  Outputs: {OUT}/")


# ---------------------------------------------------------------------------
# Issue-1 experiment
# ---------------------------------------------------------------------------

AGG_RATE = 0.05   # forced aggressive rate

def _run_aggressive_rate(prepared, rank_info, seeds, OUT):
    print("\n" + "=" * 72)
    print(f"ISSUE-1 EXPERIMENT: forced p={AGG_RATE} — does correction restore coverage?")
    print("=" * 72)

    agg_rows = []
    for ds_name, packed in prepared.items():
        X_tr, y_tr, X_va, y_va, X_te, y_te, sc, tr_raw, tr_t_idx = packed
        T_total  = len(tr_raw)
        y_std    = float(np.std(y_te))
        n_tr     = len(X_tr)
        k_agg    = max(2, int(n_tr * AGG_RATE))

        for seed in seeds:
            np.random.seed(seed); torch.manual_seed(seed)
            ui  = u_idx(n_tr, k_agg, seed)
            mdl = train_lstm(X_tr[ui], y_tr[ui], X_va, y_va, seed)

            from surge.models import pred_lstm
            vp  = pred_lstm(mdl, X_va)
            tp  = pred_lstm(mdl, X_te)
            res = np.abs(y_va - vp)

            for method, fn in [
                ("Subsample-LSTM-Std", conf_std),
                ("SURGE-A-LSTM",       lambda r, p_: conf_vc(r, p_, AGG_RATE, GAMMA_FIXED)),
            ]:
                t0     = time.time()
                lo, hi = fn(res, tp) if method == "Subsample-LSTM-Std" else fn(res, tp)
                m      = calc_metrics(y_te, lo, hi, y_std)
                agg_rows.append(dict(
                    dataset=ds_name, method=method,
                    p_rate=AGG_RATE, seed=seed,
                    **{k: round(v, 4) for k, v in m.items()},
                    train_time_sec=round(time.time() - t0, 3),
                ))

    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(OUT / "aggressive_rate_results.csv", index=False)
    print(f"\n  {'Dataset':<12} {'Method':<24} {'coverage':>10} {'width':>10}")
    print("  " + "─" * 58)
    for ds in agg_df["dataset"].unique():
        sub = agg_df[agg_df["dataset"] == ds]
        for meth, grp in sub.groupby("method"):
            surge_c = sub[sub["method"] == "SURGE-A-LSTM"]["coverage"].mean()
            std_c   = sub[sub["method"] == "Subsample-LSTM-Std"]["coverage"].mean()
            gap     = f"  ← gap={surge_c - std_c:+.3f}" if meth == "Subsample-LSTM-Std" else ""
            print(f"  {ds:<12} {meth:<24} {grp['coverage'].mean():>10.3f} "
                  f"{grp['width'].mean():>10.3f}{gap}")
        print()


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------

ABLATION_RATES = [0.05, 0.10, 0.20, 0.50, "adaptive"]

def _run_ablation(prepared, rank_info, seeds, OUT):
    print("\n" + "=" * 65)
    print("ABLATION: adaptive vs fixed subsample rates  (ETTh1, LSTM)")
    print("=" * 65)

    packed   = prepared["ETTh1"]
    X_tr, y_tr, X_va, y_va, X_te, y_te, sc, tr_raw, tr_t_idx = packed
    T_total  = len(tr_raw); y_std = float(np.std(y_te)); n_tr = len(X_tr)
    ri_a     = rank_info["ETTh1"]

    from surge.models import pred_lstm
    abl_rows = []
    for rate_spec in ABLATION_RATES:
        p_abl = ri_a["p_rate"] if rate_spec == "adaptive" else float(rate_spec)
        label = f"adaptive({p_abl:.2f})" if rate_spec == "adaptive" else f"fixed({p_abl:.2f})"
        k_abl = max(2, int(n_tr * p_abl))
        for seed in seeds:
            np.random.seed(seed); torch.manual_seed(seed)
            ui  = u_idx(n_tr, k_abl, seed)
            mdl = train_lstm(X_tr[ui], y_tr[ui], X_va, y_va, seed)
            vp  = pred_lstm(mdl, X_va)
            tp  = pred_lstm(mdl, X_te)
            res = np.abs(y_va - vp)
            lo, hi = conf_vc(res, tp, p_abl, GAMMA_FIXED)
            m   = calc_metrics(y_te, lo, hi, y_std)
            abl_rows.append(dict(rate_label=label, p_rate=p_abl, seed=seed,
                                 **{k: round(v, 4) for k, v in m.items()},
                                 gamma=GAMMA_FIXED))

    abl_df = pd.DataFrame(abl_rows)
    abl_df.to_csv(OUT / "ablation_rates.csv", index=False)
    print(f"\n  {'Rate':<22} {'coverage':>10} {'width':>10} {'calib_err':>10}")
    print("  " + "─" * 54)
    for lbl, grp in abl_df.groupby("rate_label", sort=False):
        print(f"  {lbl:<22} {grp['coverage'].mean():>10.3f} "
              f"{grp['width'].mean():>10.3f} {grp['calib_error'].mean():>10.3f}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _make_figures(results: pd.DataFrame, OUT: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COLORS_M = {
        "Full-Linear":         "#378ADD",
        "Full-LSTM":           "#1D9E75",
        "SURGE-A-Linear":      "#D85A30",
        "SURGE-A-LSTM":        "#7F77DD",
        "Subsample-LSTM-Std":  "#D4537E",
        "Naive-Subsample":     "#BA7517",
        "Leverage-Linear":     "#888780",
    }
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "grid.alpha": 0.25, "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold",
    })
    FIGS = OUT / "figures"
    ds_list = results["dataset"].unique().tolist()
    nd = len(ds_list)

    def save(name):
        plt.savefig(FIGS / f"{name}.png", dpi=110, bbox_inches="tight")
        plt.close()
        print(f"  ✓ {name}.png")

    # Fig: Coverage comparison — key LSTM methods
    lstm_m = ["Full-LSTM", "SURGE-A-LSTM", "Subsample-LSTM-Std"]
    fig, axes = plt.subplots(1, nd, figsize=(4 * nd, 4), sharey=True)
    if nd == 1: axes = [axes]
    fig.suptitle("Coverage: Full-LSTM vs SURGE-A-LSTM vs Subsample-LSTM-Std",
                 fontsize=11, fontweight="bold")
    for ax, ds in zip(axes, ds_list):
        sub = results[results["dataset"] == ds]
        for i, m in enumerate(lstm_m):
            g = sub[sub["method"] == m]
            ax.bar(i, g["coverage"].mean(),
                   yerr=g["coverage"].std() if len(g) > 1 else 0,
                   color=COLORS_M[m], alpha=0.82, edgecolor="none",
                   width=0.55, capsize=3)
        ax.axhline(1 - ALPHA, color="red", linestyle="--", linewidth=1.2)
        ax.set_ylim(0, 1.2); ax.set_title(ds, fontsize=9)
        ax.set_xticks(range(len(lstm_m)))
        ax.set_xticklabels([m.replace("-", "\n") for m in lstm_m], fontsize=7)
        if ax is axes[0]: ax.set_ylabel("coverage")
    plt.tight_layout(); save("coverage_lstm_methods")

    # Fig: Coverage-Adjusted Width heatmap
    caw_tbl = (
        results.groupby(["dataset", "method"])["caw"]
        .mean().unstack("method").reindex(columns=METHODS)
    )
    fig, ax = plt.subplots(figsize=(len(METHODS) * 1.8, len(ds_list) * 0.9 + 1))
    im = ax.imshow(caw_tbl.values, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([m.replace("-", "\n") for m in METHODS], fontsize=7)
    ax.set_yticks(range(len(ds_list))); ax.set_yticklabels(ds_list, fontsize=9)
    ax.set_title("Coverage-Adjusted Width  (lower = better)", fontsize=11, fontweight="bold")
    for i in range(caw_tbl.shape[0]):
        for j in range(caw_tbl.shape[1]):
            v = caw_tbl.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.5, color="black")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout(); save("caw_heatmap")

    print(f"\n✓ Figures saved to {FIGS}/")


if __name__ == "__main__":
    main()
