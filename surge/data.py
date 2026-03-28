"""
Real dataset loaders for SURGE-A.

All loaders return a dict with keys matching DATASETS in the benchmark:
    ETTh1, ETTm1, Weather, Jena, Store

Each value is a pd.Series of the univariate target variable.

Jena and Store require Kaggle data under /kaggle/input/ (see README).
ETTh1, ETTm1, and Weather are fetched from public URLs with fallbacks.
"""

import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_all(kaggle_input: str = "/kaggle/input") -> dict[str, pd.Series]:
    """
    Load all datasets used in the SURGE-A benchmark.

    Parameters
    ----------
    kaggle_input : path to Kaggle input directory (only needed for Jena/Store)

    Returns
    -------
    dict mapping dataset name → univariate pd.Series
    """
    raw: dict[str, pd.Series] = {}

    raw["ETTh1"]     = load_etth1()
    raw["ETTm1"]     = load_ettm1()
    raw["Weather"]   = load_weather()
    raw["Jena"]      = load_jena(kaggle_input)
    raw["Store"]     = load_store(kaggle_input)

    # Synthetic datasets are loaded separately in datasets.py
    return raw


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------

_ETT_URLS = {
    "ETTh1": [
        "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
        "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/master/ETT-small/ETTh1.csv",
    ],
    "ETTm1": [
        "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
        "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/master/ETT-small/ETTm1.csv",
    ],
}


def _load_ett(name: str) -> pd.Series:
    for url in _ETT_URLS[name]:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(
                io.StringIO(r.text), parse_dates=["date"], index_col="date"
            )
            print(f"  {name:<8}: {df.shape}  [{url}]")
            return pd.Series(df["OT"].values)
        except Exception as e:
            print(f"  ✗ {url} — {e}")
    raise RuntimeError(f"Could not load {name} from any URL.")


def load_etth1() -> pd.Series:
    """ETT-h1: hourly electricity transformer temperature (OT)."""
    return _load_ett("ETTh1")


def load_ettm1() -> pd.Series:
    """ETT-m1: 15-minute electricity transformer temperature (OT)."""
    return _load_ett("ETTm1")


def load_weather(year: int = 2022) -> pd.Series:
    """
    Frankfurt hourly weather from Open-Meteo (temperature_2m).

    Falls back to a synthetic stand-in if the API is unreachable.
    """
    params = {
        "latitude": 50.11, "longitude": 8.68,
        "start_date": f"{year}-01-01",
        "end_date":   f"{year}-12-31",
        "hourly": (
            "temperature_2m,relative_humidity_2m,surface_pressure,"
            "wind_speed_10m,precipitation,shortwave_radiation"
        ),
        "timezone": "UTC",
    }
    try:
        resp = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params=params, timeout=30
        )
        resp.raise_for_status()
        h = resp.json()["hourly"]
        df = pd.DataFrame(h).set_index(pd.to_datetime(h["time"])).drop(columns="time")
        print(f"  Weather : {df.shape}")
        return pd.Series(df["temperature_2m"].values)
    except Exception as e:
        print(f"  ✗ Open-Meteo failed ({e}). Using synthetic weather stand-in.")
        return _synthetic_weather()


def _synthetic_weather(n: int = 8760) -> pd.Series:
    """Plausible synthetic hourly temperature (365 days)."""
    rng = np.random.default_rng(0)
    t   = np.arange(n)
    s   = (
        10 + 15 * np.sin(2 * np.pi * t / n)
        + 5 * np.sin(2 * np.pi * t / 24)
        + rng.standard_normal(n) * 1.5
    )
    print(f"  Weather : {(n, 1)} (synthetic)")
    return pd.Series(s)


def load_jena(kaggle_input: str = "/kaggle/input") -> pd.Series:
    """
    Jena Climate dataset (10-min → resampled to 1-hourly temperature).

    Expected path: <kaggle_input>/datasets/mnassrib/jena-climate/jena_climate_2009_2016.csv
    """
    path = Path(kaggle_input) / "datasets/mnassrib/jena-climate/jena_climate_2009_2016.csv"
    df = pd.read_csv(str(path), low_memory=False)

    # Robust datetime-column detection
    dt_col = next(
        (c for c in df.columns if "date" in c.lower() or "time" in c.lower()),
        df.columns[0],
    )
    df[dt_col] = pd.to_datetime(df[dt_col], dayfirst=True, infer_datetime_format=True)
    df = df.set_index(dt_col).sort_index()
    df.index = pd.DatetimeIndex(df.index)

    # Fix -9999 sentinel in wind speed
    for col in ["wv (m/s)", "max. wv (m/s)"]:
        if col in df.columns:
            n_bad = (df[col] == -9999.0).sum()
            df[col] = df[col].replace(-9999.0, np.nan).interpolate(method="linear")
            if n_bad:
                print(f"  Fixed {n_bad} sentinel values in '{col}'")

    s = df["T (degC)"].resample("1h").mean()
    print(f"  Jena    : {(len(s), 1)}  (10-min→1h)")
    return pd.Series(s.values)


def load_store(kaggle_input: str = "/kaggle/input") -> pd.Series:
    """
    Kaggle Demand Forecasting store=1, item=1.

    Expected path: <kaggle_input>/competitions/demand-forecasting-kernels-only/train.csv
    """
    path = Path(kaggle_input) / "competitions/demand-forecasting-kernels-only/train.csv"
    df   = pd.read_csv(str(path), parse_dates=["date"])
    s    = (
        df[(df["store"] == 1) & (df["item"] == 1)]
        .sort_values("date")["sales"]
    )
    print(f"  Store   : {(len(s), 1)}")
    return pd.Series(s.values.astype(float))
