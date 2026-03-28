"""
Core experiment loop for the SURGE-A benchmark.

This module contains:
    prepare          — sliding-window construction + train/val/test split
    make_windows     — vectorised window extraction
    run_benchmark    — main loop over datasets × seeds × methods
    report           — console printer for per-dataset results

Methods evaluated
-----------------
Full-Linear          Ridge + Fourier basis, 100% training data, standard conformal
Full-LSTM            TinyLSTM, 100% training data, standard conformal
SURGE-A-Linear       Ridge, adaptive p%, variance-corrected conformal (γ=1)
SURGE-A-LSTM         TinyLSTM, adaptive p%, variance-corrected conformal (γ=1)
Subsample-LSTM-Std   *same* model as SURGE-A-LSTM, standard conformal (Issue-1 baseline)
Naive-Subsample      Ridge, adaptive p%, standard conformal
Leverage-Linear      Ridge, leverage-score sampling, standard conformal

Critical design note for Issue 1
---------------------------------
SURGE-A-LSTM and Subsample-LSTM-Std share the *identical* trained model.
The model is trained once per (dataset, seed) and cached in ``_sub_cache``.
The only difference between the two methods is whether the conformal
half-width includes the variance-inflation term.
"""

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .conformal import (
    ALPHA, GAMMA_FIXED,
    calc_metrics, conf_std, conf_vc,
)
from .models import (
    pred_linear, pred_lstm,
    train_linear, train_lstm,
)
from .sampling import (
    adaptive_rate, effective_rank,
    l_idx, lev_probs, make_t_idx, u_idx,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOOKBACK  = 24    # look-back window length (samples)
MIN_RATE  = 0.20  # minimum subsampling rate for standard experiments
SEEDS     = [0, 1, 2]

METHODS = [
    "Full-Linear",
    "Full-LSTM",
    "SURGE-A-Linear",
    "SURGE-A-LSTM",
    "Subsample-LSTM-Std",
    "Naive-Subsample",
    "Leverage-Linear",
]

AGG_COLS = ["coverage", "width", "calib_error", "norm_width", "caw", "train_time_sec"]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def make_windows(series: np.ndarray, lb: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sliding windows from a 1-D series.

    Parameters
    ----------
    series : 1-D float array
    lb     : look-back (window) length

    Returns
    -------
    X : np.ndarray, shape (n - lb, lb), dtype float32
    y : np.ndarray, shape (n - lb,),    dtype float32
    """
    n   = len(series) - lb
    idx = np.arange(lb)[None, :] + np.arange(n)[:, None]
    return series[idx].astype(np.float32), series[lb:].astype(np.float32)


def prepare(
    series: pd.Series,
    lb: int = LOOKBACK,
    tr_frac: float = 0.70,
    va_frac: float = 0.10,
) -> tuple:
    """
    Standardise and split a series into train / val / test windows.

    Returns
    -------
    (X_tr, y_tr, X_va, y_va, X_te, y_te, scaler, tr_raw, tr_t_idx)
    """
    v  = series.dropna().values.astype(np.float64)
    n  = len(v)
    t1 = int(n * tr_frac)
    t2 = int(n * (tr_frac + va_frac))

    sc = StandardScaler()
    tr = sc.fit_transform(v[:t1].reshape(-1, 1)).ravel()
    va = sc.transform(v[t1:t2].reshape(-1, 1)).ravel()
    te = sc.transform(v[t2:].reshape(-1, 1)).ravel()

    X_tr, y_tr = make_windows(tr, lb)
    X_va, y_va = make_windows(va, lb)
    X_te, y_te = make_windows(te, lb)

    tr_t_idx = np.arange(len(X_tr), dtype=np.int32)
    return X_tr, y_tr, X_va, y_va, X_te, y_te, sc, v[:t1], tr_t_idx


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(s: pd.Series) -> str:
    return f"{s.mean():.3f}±{s.std():.3f}"


def report(ds_name: str, rows: list[dict], elapsed: float) -> None:
    """Print a per-dataset summary table to stdout."""
    df = pd.DataFrame(rows)
    W  = 26 + 14 * len(AGG_COLS)
    tqdm.write(f"\n{'=' * W}")
    tqdm.write(f"  RESULTS ▸ {ds_name}   ({len(SEEDS)} seeds | {elapsed:.1f}s)")
    tqdm.write(f"{'=' * W}")
    tqdm.write(f"  {'Method':<26}" + "".join(f"{c:>14}" for c in AGG_COLS))
    tqdm.write("  " + "─" * W)
    for m in METHODS:
        g = df[df["method"] == m]
        if g.empty:
            continue
        tqdm.write(f"  {m:<26}" + "".join(f"{_fmt(g[c]):>14}" for c in AGG_COLS))
    tqdm.write("─" * W)


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(
    prepared: dict[str, tuple],
    rank_info: dict[str, dict],
    seeds: list[int] = SEEDS,
) -> pd.DataFrame:
    """
    Run all methods on all datasets for all seeds.

    Parameters
    ----------
    prepared  : {ds_name: prepare(...) output}
    rank_info : {ds_name: dict(T, eff_rank, p_rate, k_sub, n_tr, high_rank)}
    seeds     : list of random seeds

    Returns
    -------
    pd.DataFrame with one row per (dataset, method, seed)
    """
    all_rows: list[dict] = []
    total    = len(prepared) * len(seeds) * len(METHODS)
    overall  = tqdm(total=total, desc="Overall", unit="run", dynamic_ncols=True)

    for ds_name, packed in prepared.items():
        X_tr, y_tr, X_va, y_va, X_te, y_te, sc, tr_raw, tr_t_idx = packed
        ri      = rank_info[ds_name]
        p_rate  = ri["p_rate"]
        k_sub   = ri["k_sub"]
        n_tr    = len(X_tr)
        y_std   = float(np.std(y_te))
        T_total = len(tr_raw)
        ds_t0   = time.time()

        va_t_idx = make_t_idx(T_total,            len(X_va))
        te_t_idx = make_t_idx(T_total + len(X_va), len(X_te))

        tqdm.write(
            f"\n▶ {ds_name}  T={T_total:,}  er={ri['eff_rank']}  "
            f"p={p_rate:.4f}  k={k_sub:,}/{n_tr:,}"
        )

        # Cache deterministic Full-Linear (same across seeds)
        _lp   = lev_probs(X_tr, tr_t_idx, T_total)
        _fl   = train_linear(X_tr, y_tr, tr_t_idx, T_total)
        _flvp = pred_linear(_fl, X_va, va_t_idx, T_total)
        _fltp = pred_linear(_fl, X_te, te_t_idx, T_total)
        _flr  = np.abs(y_va - _flvp)

        ds_rows: list[dict] = []

        for seed in tqdm(seeds, desc=f"  {ds_name}", unit="seed",
                         leave=False, dynamic_ncols=True):
            np.random.seed(seed)
            torch.manual_seed(seed)

            ui = u_idx(n_tr, k_sub, seed)
            li = l_idx(_lp, k_sub, seed)

            # Cache shared subsampled LSTM (Issue-1 fix: same model, two conformal steps)
            _sub_cache: dict[str, Any] = {}

            def _get_sub_lstm():
                if "tp" not in _sub_cache:
                    mdl_ = train_lstm(X_tr[ui], y_tr[ui], X_va, y_va, seed)
                    vp_  = pred_lstm(mdl_, X_va)
                    _sub_cache["tp"]  = pred_lstm(mdl_, X_te)
                    _sub_cache["res"] = np.abs(y_va - vp_)
                return _sub_cache["tp"], _sub_cache["res"]

            for method in tqdm(METHODS, desc=f"    s{seed}", unit="m",
                               leave=False, dynamic_ncols=True):
                t0 = time.time()
                try:
                    lo, hi = _run_method(
                        method=method,
                        X_tr=X_tr, y_tr=y_tr,
                        X_va=X_va, y_va=y_va,
                        X_te=X_te,
                        tr_t_idx=tr_t_idx,
                        va_t_idx=va_t_idx,
                        te_t_idx=te_t_idx,
                        T_total=T_total,
                        ui=ui, li=li,
                        p_rate=p_rate,
                        seed=seed,
                        # Full-Linear pre-computed
                        fl_res=_flr, fl_tp=_fltp,
                        # Shared subsampled LSTM
                        get_sub_lstm=_get_sub_lstm,
                    )

                    m = calc_metrics(y_te, lo, hi, y_std)
                    row = dict(
                        dataset=ds_name,
                        method=method,
                        subsample_rate=round(p_rate, 4) if "Full" not in method else 1.0,
                        seed=seed,
                        **{k: round(v, 4) for k, v in m.items()},
                        train_time_sec=round(time.time() - t0, 3),
                        memory_mb=0,
                    )
                    ds_rows.append(row)
                    all_rows.append(row)

                except Exception as exc:
                    tqdm.write(f"  [{ds_name}] s={seed} {method}  ERR: {exc}")

                overall.update(1)

        report(ds_name, ds_rows, time.time() - ds_t0)

    overall.close()
    return pd.DataFrame(all_rows)


def _run_method(
    *,
    method: str,
    X_tr, y_tr, X_va, y_va, X_te,
    tr_t_idx, va_t_idx, te_t_idx,
    T_total: int,
    ui: np.ndarray,
    li: np.ndarray,
    p_rate: float,
    seed: int,
    fl_res: np.ndarray,
    fl_tp: np.ndarray,
    get_sub_lstm,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to the correct training/conformal pipeline."""

    if method == "Full-Linear":
        return conf_vc(fl_res, fl_tp, 1.0)

    if method == "Full-LSTM":
        mdl = train_lstm(X_tr, y_tr, X_va, y_va, seed)
        vp  = pred_lstm(mdl, X_va)
        tp  = pred_lstm(mdl, X_te)
        return conf_vc(np.abs(y_va - vp), tp, 1.0)

    if method == "SURGE-A-Linear":
        sub_t = tr_t_idx[ui]
        mdl   = train_linear(X_tr[ui], y_tr[ui], sub_t, T_total)
        vp    = pred_linear(mdl, X_va, va_t_idx, T_total)
        tp    = pred_linear(mdl, X_te, te_t_idx, T_total)
        return conf_vc(np.abs(y_va - vp), tp, p_rate, GAMMA_FIXED)

    if method == "SURGE-A-LSTM":
        tp, res = get_sub_lstm()
        return conf_vc(res, tp, p_rate, GAMMA_FIXED)

    if method == "Subsample-LSTM-Std":
        tp, res = get_sub_lstm()  # same model as SURGE-A-LSTM
        return conf_std(res, tp)

    if method == "Naive-Subsample":
        sub_t = tr_t_idx[ui]
        mdl   = train_linear(X_tr[ui], y_tr[ui], sub_t, T_total)
        vp    = pred_linear(mdl, X_va, va_t_idx, T_total)
        tp    = pred_linear(mdl, X_te, te_t_idx, T_total)
        return conf_std(np.abs(y_va - vp), tp)

    if method == "Leverage-Linear":
        sub_t = tr_t_idx[li]
        mdl   = train_linear(X_tr[li], y_tr[li], sub_t, T_total)
        vp    = pred_linear(mdl, X_va, va_t_idx, T_total)
        tp    = pred_linear(mdl, X_te, te_t_idx, T_total)
        return conf_std(np.abs(y_va - vp), tp)

    raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Rank info helper
# ---------------------------------------------------------------------------

def compute_rank_info(
    prepared: dict[str, tuple],
    min_rate: float = MIN_RATE,
) -> dict[str, dict]:
    """
    Compute effective rank and adaptive rate for every dataset.

    Parameters
    ----------
    prepared  : output of {name: prepare(series)}
    min_rate  : minimum subsampling rate floor

    Returns
    -------
    dict {ds_name: {T, eff_rank, p_rate, k_sub, n_tr, high_rank}}
    """
    rank_info = {}
    print(f"\n{'=' * 72}")
    print(f"EFFECTIVE RANK  (min_rate={min_rate})")
    print(f"{'=' * 72}")
    for name, packed in prepared.items():
        X_tr, _, _, _, _, _, _, tr_raw, _ = packed
        T  = len(tr_raw)
        er = effective_rank(tr_raw)
        p  = adaptive_rate(T, er, min_rate=min_rate)
        k  = max(2, int(len(X_tr) * p))
        high = er > 0.3 * T
        rank_info[name] = dict(T=T, eff_rank=er, p_rate=p, k_sub=k,
                               n_tr=len(X_tr), high_rank=high)
        flag = "  ⚠ HIGH RANK" if high else ""
        print(f"  {name:<12} T={T:,}  er={er}  p={p:.4f}  k={k:,}{flag}")
    return rank_info
