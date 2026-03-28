"""
Conformal prediction utilities for SURGE-A.

The central contribution is ``conf_vc``: a variance-corrected split conformal
interval that accounts for coverage degradation when only a fraction ``p`` of
training data is used.

Design notes
------------
GAMMA_FIXED = 1.0
    Do **not** tune gamma on the calibration residuals.  If you minimise
    ``|coverage - (1-α)|`` over gamma, the optimiser will set gamma → 0
    whenever residual variance is small, which completely disables the
    correction and makes SURGE-A-LSTM identical to Subsample-LSTM-Std.
    Use the theoretically motivated value (1.0) and let the correction
    stand or fall on its own merits.  A separate hold-out ablation helper
    (``tune_gamma_holdout``) is provided for supplementary analysis only.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

ALPHA: float = 0.10          # miscoverage level → target coverage = 1 - ALPHA = 0.90
GAMMA_FIXED: float = 1.0     # fixed inflation coefficient (not tuned on cal set)


# ---------------------------------------------------------------------------
# Interval construction
# ---------------------------------------------------------------------------

def conf_vc(
    res: np.ndarray,
    preds: np.ndarray,
    p: float,
    gamma: float = GAMMA_FIXED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Variance-corrected split conformal interval (SURGE-A).

    Half-width:
        q̂ + γ · sqrt( Var(res) · (1/p − 1) )

    When p = 1.0 (full data) the inflation term is exactly 0, recovering
    standard split conformal.  The correction is uncapped — reviewers
    confirmed this is the correct formulation.

    Parameters
    ----------
    res   : calibration absolute residuals  |y_val − ŷ_val|
    preds : test-set point predictions
    p     : subsampling rate in (0, 1]
    gamma : inflation coefficient (default: GAMMA_FIXED = 1.0)

    Returns
    -------
    (lower, upper) : tuple of np.ndarray, shape == preds.shape
    """
    q = float(np.quantile(res, 1.0 - ALPHA))
    inflation = _inflation(float(np.var(res)), p, gamma)
    hw = q + inflation
    return preds - hw, preds + hw


def conf_std(
    res: np.ndarray,
    preds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Standard split conformal interval (no variance correction).

    Used by *Subsample-LSTM-Std* — the same trained model as SURGE-A-LSTM,
    but without the inflation term.  This is the key Issue-1 comparison:
    model identical, interval construction differs.

    Parameters
    ----------
    res   : calibration absolute residuals
    preds : test-set point predictions

    Returns
    -------
    (lower, upper) : tuple of np.ndarray
    """
    q = float(np.quantile(res, 1.0 - ALPHA))
    return preds - q, preds + q


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calc_metrics(
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    y_std: float,
) -> dict:
    """
    Evaluate a set of prediction intervals against ground truth.

    Parameters
    ----------
    y     : true test values
    lo    : lower interval bounds
    hi    : upper interval bounds
    y_std : std-dev of test targets (for normalised width)

    Returns
    -------
    dict with keys:
        coverage     : empirical coverage in [0, 1]
        width        : mean interval width
        calib_error  : |coverage − (1 − α)|
        norm_width   : width / y_std
        caw          : coverage-adjusted width (penalises under-coverage)
    """
    covered = (y >= lo) & (y <= hi)
    cov = float(np.mean(covered))
    wid = float(np.mean(hi - lo))
    target_cov = 1.0 - ALPHA
    calib_err = abs(cov - target_cov)
    norm_w = wid / (y_std + 1e-8)
    # CAW: penalise under-coverage heavily
    caw = wid if cov >= (target_cov - 0.02) else wid * (1.0 + 10.0 * (target_cov - cov))
    return dict(
        coverage=cov,
        width=wid,
        calib_error=calib_err,
        norm_width=norm_w,
        caw=caw,
    )


# ---------------------------------------------------------------------------
# Supplementary: gamma tuning on hold-out (NOT used in main experiments)
# ---------------------------------------------------------------------------

def tune_gamma_holdout(
    res_hold: np.ndarray,
    p: float,
    gammas: np.ndarray | None = None,
) -> float:
    """
    Tune gamma on a *separate* hold-out split — NOT the calibration set.

    This is provided for supplementary ablation analysis only.  Never call
    this during the main experiment; use GAMMA_FIXED = 1.0 there.

    Parameters
    ----------
    res_hold : absolute residuals from a dedicated hold-out set
    p        : subsampling rate
    gammas   : grid of gamma candidates (default: linspace(0, 5, 101))

    Returns
    -------
    float : gamma value that minimises |coverage − (1 − α)| on res_hold
    """
    if gammas is None:
        gammas = np.linspace(0.0, 5.0, 101)
    q = float(np.quantile(res_hold, 1.0 - ALPHA))
    var = float(np.var(res_hold))
    best_g, best_err = 1.0, float("inf")
    for g in gammas:
        hw = q + _inflation(var, p, float(g))
        cov = float(np.mean(res_hold <= hw))
        err = abs(cov - (1.0 - ALPHA))
        if err < best_err:
            best_err = err
            best_g = float(g)
    return best_g


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _inflation(var: float, p: float, gamma: float) -> float:
    """
    Extra half-width from subsampling variance:
        γ · sqrt( Var_resid · (1/p − 1) )

    No cap — the full correction is applied regardless of magnitude.
    When p = 1 the term is exactly 0.
    """
    return gamma * np.sqrt(var * max(0.0, 1.0 / p - 1.0))
