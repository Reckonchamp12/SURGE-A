"""
Prediction models for SURGE-A.

Models
------
TinyLSTM        1-layer LSTM with 32 hidden units + linear head.
train_lstm      Training loop with early stopping on validation MSE.
pred_lstm       Batched inference with no_grad.

train_linear    Ridge regression on [window features | Fourier basis].
pred_linear     Prediction for the linear model.

Architecture rationale
----------------------
TinyLSTM is intentionally small (32 hidden) to keep wall-clock times
manageable on Kaggle P100 hardware.  The benchmark's claim is not that
LSTMs are generally better than linear models, but that:
  (a) on nonlinear datasets (MackeyGlass, Chirp) Full-Linear fails while
      Full-LSTM succeeds, and
  (b) SURGE-A-LSTM matches Full-LSTM coverage while training on only p%
      of the data.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge
from tqdm.auto import tqdm

from .sampling import _build_Xf


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

class TinyLSTM(nn.Module):
    """
    Single-layer LSTM → linear head for univariate sequence regression.

    Input  : (batch, lookback, 1)
    Output : (batch,)
    """

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers=1, batch_first=True)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1) → last hidden state → scalar
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def train_lstm(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    seed: int,
    hidden: int = 32,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 3e-3,
    patience: int = 5,
) -> TinyLSTM:
    """
    Train TinyLSTM with early stopping on validation MSE.

    Parameters
    ----------
    X_tr, y_tr : training windows and targets
    X_va, y_va : validation windows and targets
    seed       : random seed for reproducibility
    hidden     : LSTM hidden size
    epochs     : maximum training epochs
    batch_size : mini-batch size
    lr         : Adam learning rate
    patience   : early-stopping patience (epochs without improvement)

    Returns
    -------
    TinyLSTM with best validation weights loaded.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr = torch.tensor(X_tr, dtype=torch.float32).unsqueeze(-1)
    ytr = torch.tensor(y_tr, dtype=torch.float32)
    Xva = torch.tensor(X_va, dtype=torch.float32).unsqueeze(-1)
    yva = torch.tensor(y_va, dtype=torch.float32)

    tr_dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size,
                       shuffle=True,  num_workers=0)
    va_dl = DataLoader(TensorDataset(Xva, yva), batch_size=batch_size,
                       shuffle=False, num_workers=0)

    model  = TinyLSTM(hidden)
    optim  = torch.optim.Adam(model.parameters(), lr=lr)
    loss_f = nn.MSELoss()

    best_val   = float("inf")
    best_state = None
    wait       = 0

    ep_bar = tqdm(range(epochs), desc="    ep", unit="ep",
                  leave=False, dynamic_ncols=True)

    for epoch in ep_bar:
        model.train()
        for xb, yb in tr_dl:
            optim.zero_grad()
            loss_f(model(xb), yb).backward()
            optim.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_dl:
                val_loss += loss_f(model(xb), yb).item() * len(yb)
        val_loss /= max(1, len(y_va))

        ep_bar.set_postfix(ep=epoch, val=f"{val_loss:.4f}", best=f"{best_val:.4f}")

        if val_loss < best_val - 1e-6:
            best_val   = val_loss
            wait       = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                ep_bar.close()
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def pred_lstm(model: TinyLSTM, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """
    Batched inference for TinyLSTM.

    Parameters
    ----------
    model      : trained TinyLSTM
    X          : window array, shape (n, lookback)
    batch_size : inference batch size

    Returns
    -------
    np.ndarray, shape (n,)
    """
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    chunks = []
    with torch.no_grad():
        for i in range(0, len(Xt), batch_size):
            chunks.append(model(Xt[i : i + batch_size]).numpy())
    return np.concatenate(chunks)


# ---------------------------------------------------------------------------
# Linear (Ridge + Fourier basis)
# ---------------------------------------------------------------------------

def train_linear(
    X: np.ndarray,
    y: np.ndarray,
    t_idx: np.ndarray,
    T_total: int,
    alpha: float = 1e-3,
) -> Ridge:
    """
    Fit Ridge regression on [window features | Fourier basis].

    The Fourier basis encodes global time structure so that the linear
    model can capture long-range periodicities (e.g. daily cycles in
    energy and weather datasets).

    Parameters
    ----------
    X       : window features, shape (n, lookback)
    y       : targets, shape (n,)
    t_idx   : integer time indices, shape (n,)
    T_total : total series length for Fourier normalisation
    alpha   : Ridge regularisation strength

    Returns
    -------
    Fitted sklearn Ridge model.
    """
    Xf = _build_Xf(X, t_idx, T_total)
    m  = Ridge(alpha=alpha, solver="cholesky", fit_intercept=True)
    m.fit(Xf, y)
    return m


def pred_linear(
    model: Ridge,
    X: np.ndarray,
    t_idx: np.ndarray,
    T_total: int,
) -> np.ndarray:
    """
    Generate predictions from a fitted linear model.

    Parameters
    ----------
    model   : fitted Ridge model
    X       : window features, shape (n, lookback)
    t_idx   : integer time indices, shape (n,)
    T_total : total series length

    Returns
    -------
    np.ndarray, shape (n,)
    """
    return model.predict(_build_Xf(X, t_idx, T_total))
