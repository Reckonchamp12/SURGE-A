"""
Sampling utilities for SURGE-A.

Provides:
    effective_rank   — spectral effective rank via FFT cumulative power
    adaptive_rate    — translates effective rank into a subsampling rate
    u_idx            — uniform random index selection
    lev_probs        — leverage score probabilities for a feature matrix
    l_idx            — leverage-score-weighted index selection
    make_t_idx       — helper to build contiguous time index arrays
"""

import numpy as np
from sklearn.linear_model import Ridge

# ---------------------------------------------------------------------------
# Effective rank
# ---------------------------------------------------------------------------

_N_FREQ = 20   # number of Fourier frequencies used when building the linear basis

THRESHOLD_DEFAULT = 0.95   # fraction of spectral power captured by effective rank


def effective_rank(y: np.ndarray, threshold: float = THRESHOLD_DEFAULT) -> int:
    """
    Spectral effective rank of a time series.

    Computed as the minimum number of FFT frequency components (sorted by
    descending power) needed to capture ``threshold`` fraction of total
    spectral power.

    Parameters
    ----------
    y         : 1-D array of time-series values (mean will be subtracted)
    threshold : cumulative power fraction (default 0.95)

    Returns
    -------
    int : effective rank ≥ 1
    """
    y = np.asarray(y, dtype=np.float64)
    y = y - y.mean()
    pw = np.abs(np.fft.rfft(y)) ** 2
    pw[0] = 0.0  # remove DC component
    total = pw.sum()
    if total < 1e-12:
        return 1
    # Sort by descending power, build cumulative sum
    cum = np.cumsum(np.sort(pw)[::-1]) / total
    return int(np.searchsorted(cum, threshold)) + 1


# ---------------------------------------------------------------------------
# Adaptive subsampling rate
# ---------------------------------------------------------------------------

def adaptive_rate(
    T: int,
    er: int,
    c: float = 2.0,
    min_rate: float = 0.20,
) -> float:
    """
    Translate effective rank into a subsampling rate.

        p = max(min_rate, min(1.0, c * er / T))

    The idea: a series with low spectral rank can be faithfully represented
    by fewer samples.  The constant ``c`` controls how aggressively to
    subsample; ``min_rate`` prevents the rate from collapsing on very long,
    low-rank series.

    Parameters
    ----------
    T        : total number of training samples
    er       : effective rank (from ``effective_rank``)
    c        : scaling constant (default 2.0)
    min_rate : hard floor on the subsampling rate (default 0.20)

    Returns
    -------
    float : subsampling rate in [min_rate, 1.0]
    """
    return max(min_rate, min(1.0, c * er / T))


# ---------------------------------------------------------------------------
# Index selection
# ---------------------------------------------------------------------------

def u_idx(n: int, k: int, seed: int) -> np.ndarray:
    """
    Draw ``k`` indices uniformly at random from {0, …, n-1} without replacement.
    """
    return np.random.default_rng(seed).choice(n, size=k, replace=False)


def lev_probs(X: np.ndarray, t_idx: np.ndarray, T_total: int) -> np.ndarray:
    """
    Compute leverage-score sampling probabilities for the augmented feature
    matrix [X | Fourier(t_idx)].

    Parameters
    ----------
    X       : window feature matrix, shape (n, lookback)
    t_idx   : integer time indices, shape (n,)
    T_total : total series length (used to normalise the Fourier basis)

    Returns
    -------
    np.ndarray, shape (n,), sums to 1.0
    """
    Xf = _build_Xf(X, t_idx, T_total)
    try:
        U, _, _ = np.linalg.svd(Xf, full_matrices=False)
        lev = np.sum(U ** 2, axis=1)
    except Exception:
        lev = np.ones(len(Xf))
    p = lev / (lev.sum() + 1e-12)
    return p


def l_idx(probs: np.ndarray, k: int, seed: int) -> np.ndarray:
    """
    Draw ``k`` indices according to leverage-score probabilities without
    replacement.
    """
    return np.random.default_rng(seed).choice(
        len(probs), size=k, replace=False, p=probs
    )


def make_t_idx(start: int, n: int) -> np.ndarray:
    """Return contiguous integer time indices [start, start+n)."""
    return np.arange(start, start + n, dtype=np.int32)


# ---------------------------------------------------------------------------
# Fourier feature helpers (shared with models.py via this module)
# ---------------------------------------------------------------------------

_F_CACHE: dict = {}


def fourier_basis_global(t_idx: np.ndarray, T_total: int) -> np.ndarray:
    """
    Build a sin/cos Fourier basis aligned to global time indices.

    The basis uses ``_N_FREQ`` frequencies, each with a sin and cos component,
    giving ``2 * _N_FREQ`` columns.  Time normalisation uses the global series
    length ``T_total`` so that the basis is consistent across train/val/test
    splits.

    Results are cached by (first_t, last_t, n, T_total) to avoid redundant
    computation in the main loop.
    """
    key = (int(t_idx[0]), int(t_idx[-1]), len(t_idx), T_total)
    if key not in _F_CACHE:
        t_norm = t_idx.astype(np.float64) / max(T_total, 1)
        cols = []
        for f in range(1, _N_FREQ + 1):
            cols.append(np.sin(2.0 * np.pi * f * t_norm))
            cols.append(np.cos(2.0 * np.pi * f * t_norm))
        _F_CACHE[key] = np.column_stack(cols).astype(np.float32)
    return _F_CACHE[key]


def _build_Xf(X: np.ndarray, t_idx: np.ndarray, T_total: int) -> np.ndarray:
    """Concatenate window features with global Fourier basis."""
    F = fourier_basis_global(t_idx, T_total)
    return np.concatenate([X, F], axis=1)
