"""SURGE-A: Adaptive Conformal Prediction for Time Series via Subsampling."""

from .conformal import conf_vc, conf_std, calc_metrics, ALPHA, GAMMA_FIXED
from .sampling import effective_rank, adaptive_rate
from .datasets import mackey_glass, lorenz_series, chirp_nonlinear
from .models import TinyLSTM, train_lstm, pred_lstm, train_linear, pred_linear

__all__ = [
    "conf_vc",
    "conf_std",
    "calc_metrics",
    "ALPHA",
    "GAMMA_FIXED",
    "effective_rank",
    "adaptive_rate",
    "mackey_glass",
    "lorenz_series",
    "chirp_nonlinear",
    "TinyLSTM",
    "train_lstm",
    "pred_lstm",
    "train_linear",
    "pred_linear",
]
