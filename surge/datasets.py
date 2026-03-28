"""
Synthetic dataset generators for SURGE-A.

All series are z-scored before being returned.

Datasets
--------
mackey_glass   Low-rank, nonlinear attractor (τ=17).  Linear models over-smooth;
               LSTMs track the attractor.  Primary dataset for NeurIPS Issues 2 & 3.

lorenz_series  High-rank chaotic signal (Lorenz x-component).  Used as a
               failure-case high-rank diagnostic; kept for Issue 3 ablation.

chirp_nonlinear  Non-stationary instantaneous frequency.  Linear Fourier basis
               cannot represent a quadratically-drifting frequency, so Full-Linear
               produces wide/miscalibrated intervals while an LSTM can track it.
               Reviewer-suggested dataset for Issue 2.
"""

import numpy as np
from scipy.integrate import solve_ivp


def mackey_glass(n: int = 12_000, tau: int = 17, seed: int = 0) -> np.ndarray:
    """
    Mackey-Glass delay-differential equation.

    Parameters
    ----------
    n    : int   Number of output samples (after discarding the warm-up).
    tau  : int   Delay parameter.  tau=17 → chaotic but low-rank spectrum.
    seed : int   RNG seed for initial condition noise.

    Returns
    -------
    np.ndarray, shape (n,), dtype float64, z-scored.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(n + tau)
    x[:tau] = 0.9 + 0.05 * rng.standard_normal(tau)
    for t in range(tau, n + tau - 1):
        x[t + 1] = (
            x[t]
            + 0.2 * x[t - tau] / (1.0 + x[t - tau] ** 10)
            - 0.1 * x[t]
        )
    s = x[tau:]
    return _zscore(s)


def lorenz_series(n: int = 10_000, dt: float = 0.01, seed: int = 0) -> np.ndarray:
    """
    Chaotic Lorenz x-component.

    High effective rank — used as a diagnostic failure case to show that
    SURGE-A coverage degrades when the series is intrinsically high-rank
    and the adaptive rate is forced down.

    Parameters
    ----------
    n    : int    Number of output time steps.
    dt   : float  Integration step size.
    seed : int    RNG seed for initial condition.

    Returns
    -------
    np.ndarray, shape (n,), dtype float64, z-scored.
    """
    rng = np.random.default_rng(seed)

    def _lorenz(t, s):
        x, y, z = s
        return [10.0 * (y - x), x * (28.0 - z) - y, x * y - 2.667 * z]

    sol = solve_ivp(
        _lorenz,
        t_span=[0, n * dt],
        y0=rng.standard_normal(3),
        max_step=dt,
        t_eval=np.arange(0, n * dt, dt),
    )
    return _zscore(sol.y[0])


def chirp_nonlinear(n: int = 10_000, seed: int = 0) -> np.ndarray:
    """
    Chirp signal with quadratically increasing instantaneous frequency.

        s(t) = sin(2π t) + 0.5 sin(4π t + t²/10) + ε,  ε ~ N(0, 0.05²)

    A fixed Fourier basis (used by Full-Linear) cannot represent the drifting
    frequency, resulting in large residuals and wide/miscalibrated intervals.
    An LSTM can track the slowly varying frequency regime.

    Parameters
    ----------
    n    : int  Number of output samples.
    seed : int  RNG seed for noise term.

    Returns
    -------
    np.ndarray, shape (n,), dtype float64, z-scored.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 10, n)
    s = (
        np.sin(2.0 * np.pi * t)
        + 0.5 * np.sin(4.0 * np.pi * t + t ** 2 / 10.0)
        + 0.05 * rng.standard_normal(n)
    )
    return _zscore(s)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _zscore(s: np.ndarray) -> np.ndarray:
    """Return z-scored copy of *s* as float64."""
    s = np.asarray(s, dtype=np.float64)
    mu, sigma = s.mean(), s.std()
    return (s - mu) / (sigma + 1e-8)
