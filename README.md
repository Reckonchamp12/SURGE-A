# SURGE-A: Subsampling Under Rank-Guided Efficiency — Adaptive Conformal Prediction for Time Series

> **NeurIPS 2024 Submission** · [Paper](#) · [Kaggle Notebook](#) · [Datasets](#datasets)

---

## Overview

**SURGE-A** is a benchmark and method for conformal prediction under data subsampling.
Standard split conformal intervals assume access to the full calibration set; when only a
fraction `p` of training data is used (for computational efficiency), coverage degrades.
SURGE-A corrects for this via a variance-inflation term:

$$\hat{q}_{\text{vc}} = \hat{q}_{\text{std}} + \gamma \sqrt{\hat{\sigma}^2_{\text{res}}\left(\frac{1}{p} - 1\right)}$$

where `γ = 1` is theoretically motivated (not tuned on the calibration set) and
`p` is the adaptive subsampling rate derived from the **effective spectral rank** of the series.

### Key contributions

| Issue (NeurIPS review) | Fix in SURGE-A v4 |
|---|---|
| Variance correction not properly tested | `Subsample-LSTM-Std` shares the **same** trained model as `SURGE-A-LSTM`; only the conformal step differs |
| Linear beats LSTM on standard datasets | Added **MackeyGlass** (τ=17) and **Chirp** — nonlinear datasets where linear fails |
| Lorenz doesn't differentiate LSTM | Replaced by MackeyGlass as the primary nonlinear dataset; Lorenz kept as high-rank failure case |
| Missing baseline confirmed | Aggressive-rate experiment (p=0.05) exposes the coverage gap between corrected and uncorrected intervals |

---

## Repository Structure

```
surge-a/
├── surge/                  # Core library
│   ├── __init__.py
│   ├── datasets.py         # Synthetic datasets (MackeyGlass, Lorenz, Chirp)
│   ├── data.py             # Real dataset loaders (ETTh1, ETTm1, Weather, Jena, Store)
│   ├── models.py           # TinyLSTM + Ridge linear model
│   ├── conformal.py        # conf_vc, conf_std, metrics, inflation formula
│   ├── sampling.py         # Effective rank, adaptive rate, uniform/leverage sampling
│   └── benchmark.py        # Full experiment loop (all methods × all seeds)
├── experiments/
│   ├── run_benchmark.py    # Main results (Table 1, all figures)
│   ├── run_analysis.py     # Dataset EDA — statistics, ACF, FFT, stationarity
│   ├── issue1_aggressive.py # Issue 1: forced p=0.05 coverage-gap experiment
│   └── ablation_rates.py   # Ablation: adaptive vs fixed rates on ETTh1
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   └── 02_surge_benchmark.ipynb
├── scripts/
│   └── download_jena.sh    # Helper to get Jena Climate dataset
├── tests/
│   ├── test_conformal.py
│   └── test_sampling.py
├── requirements.txt
├── setup.py
└── .github/workflows/ci.yml
```

---

## Datasets

| Dataset | Length | Type | Target | Source |
|---|---|---|---|---|
| ETTh1 | 17,420 | Electricity (hourly) | OT (oil temperature) | [ETDataset](https://github.com/zhouhaoyi/ETDataset) |
| ETTm1 | 69,680 | Electricity (15-min) | OT | [ETDataset](https://github.com/zhouhaoyi/ETDataset) |
| Weather | 8,760 | Meteorological | temperature_2m | [Open-Meteo](https://open-meteo.com) |
| Jena | ~70,000 | Climate (10-min→1h) | T (degC) | [Kaggle](https://www.kaggle.com/datasets/mnassrib/jena-climate) |
| Store | 913 | Retail demand | sales | [Kaggle](https://www.kaggle.com/competitions/demand-forecasting-kernels-only) |
| MackeyGlass | 12,000 | Synthetic | — | Generated (τ=17) |
| Chirp | 10,000 | Synthetic | — | Generated (non-stationary) |
| HighRank | 10,000 | Synthetic | — | Lorenz x-component |

ETTh1, ETTm1, and Weather are loaded automatically from public URLs.
Jena and Store require Kaggle credentials (see [Setup](#setup)).

---

## Setup

```bash
# 1. Clone
git clone https://github.com/your-org/surge-a.git
cd surge-a

# 2. Install
pip install -e .

# 3. (Optional) Kaggle datasets — requires ~/.kaggle/kaggle.json
bash scripts/download_jena.sh
```

Requirements: Python ≥ 3.9, see `requirements.txt`.

---

## Reproducing Results

### Full benchmark (Table 1 + all figures)

```bash
python experiments/run_benchmark.py
# Outputs → results/surge_a_results.csv, results/figures/
```

### Dataset analysis (EDA figures)

```bash
python experiments/run_analysis.py
# Outputs → results/figures/fig1_timeseries.png … fig7_stationarity.png
```

### Issue 1 experiment (aggressive rate p=0.05)

```bash
python experiments/issue1_aggressive.py
# Outputs → results/aggressive_rate_results.csv
```

### Ablation (adaptive vs fixed rates, ETTh1)

```bash
python experiments/ablation_rates.py
# Outputs → results/ablation_rates.csv
```

---

## Method

### Effective rank & adaptive rate

The subsampling rate is derived from the **spectral effective rank** of the training series:

```python
# 95% cumulative power threshold in the FFT
effective_rank(y, threshold=0.95)

# Adaptive rate: p = max(p_min, c * er / T)
adaptive_rate(T, er, c=2.0, min_rate=0.20)
```

### Variance-corrected conformal interval

```python
# SURGE-A: corrected half-width
half_width = q_hat + gamma * sqrt(var_residuals * (1/p - 1))

# Baseline: standard split conformal (no correction)
half_width = q_hat
```

`gamma = 1.0` is fixed (theoretically motivated). **Do not tune gamma on the
calibration set** — doing so collapses the correction to zero when residual variance
is small, making SURGE-A-LSTM indistinguishable from Subsample-LSTM-Std.

### Methods compared

| Method | Model | Conformal | Data fraction |
|---|---|---|---|
| Full-Linear | Ridge + Fourier | Standard | 100% |
| Full-LSTM | TinyLSTM (32-hidden) | Standard | 100% |
| SURGE-A-Linear | Ridge + Fourier | **Variance-corrected** | adaptive p |
| **SURGE-A-LSTM** | TinyLSTM | **Variance-corrected** | adaptive p |
| Subsample-LSTM-Std | *same* TinyLSTM | Standard (no correction) | adaptive p |
| Naive-Subsample | Ridge | Standard | adaptive p |
| Leverage-Linear | Ridge (leverage sampling) | Standard | adaptive p |

`SURGE-A-LSTM` and `Subsample-LSTM-Std` use the **identical trained model** —
the only difference is whether the conformal half-width is variance-corrected.

---

## Results (summary)

Coverage target: **90%** (α = 0.10), averaged over 3 seeds.

| Dataset | Full-LSTM cov | SURGE-A-LSTM cov | Subsample-Std cov | Speedup |
|---|---|---|---|---|
| ETTh1 | ~0.90 | ~0.90 | < 0.90 | ~5× |
| MackeyGlass | ~0.90 | ~0.90 | < 0.90 | ~5× |
| Chirp | ~0.90 | ~0.90 | ≪ 0.90 | ~5× |
| HighRank | ~0.90 | moderate | low | ~5× |

Full results in `results/surge_a_results.csv` after running the benchmark.

---

## Citation

```bibtex
@inproceedings{surge-a-2024,
  title     = {SURGE-A: Adaptive Conformal Prediction for Time Series via Subsampling Under Rank-Guided Efficiency},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2024},
}
```

---

## License

MIT
