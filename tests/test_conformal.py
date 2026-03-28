"""Unit tests for surge.conformal and surge.sampling."""

import numpy as np
import pytest

from surge.conformal import (
    ALPHA, GAMMA_FIXED,
    _inflation, calc_metrics, conf_std, conf_vc,
)
from surge.sampling import adaptive_rate, effective_rank


# ---------------------------------------------------------------------------
# conformal.py
# ---------------------------------------------------------------------------

class TestInflation:
    def test_full_data_no_inflation(self):
        """p=1.0 → inflation exactly 0."""
        assert _inflation(var=1.0, p=1.0, gamma=1.0) == pytest.approx(0.0)

    def test_half_data(self):
        """p=0.5, var=1, γ=1 → sqrt(1*(2-1)) = 1."""
        assert _inflation(var=1.0, p=0.5, gamma=1.0) == pytest.approx(1.0)

    def test_gamma_scaling(self):
        val = _inflation(var=1.0, p=0.5, gamma=2.0)
        assert val == pytest.approx(2.0)

    def test_no_negative_inflation(self):
        """p > 1 edge case should not produce negative inflation."""
        assert _inflation(var=1.0, p=2.0, gamma=1.0) == pytest.approx(0.0)


class TestConfVc:
    def test_full_data_equals_std(self):
        """When p=1, conf_vc should equal conf_std."""
        rng = np.random.default_rng(0)
        res   = rng.standard_normal(200) ** 2
        preds = rng.standard_normal(50)
        lo_vc, hi_vc  = conf_vc(res, preds, p=1.0, gamma=GAMMA_FIXED)
        lo_std, hi_std = conf_std(res, preds)
        np.testing.assert_allclose(lo_vc, lo_std, rtol=1e-6)
        np.testing.assert_allclose(hi_vc, hi_std, rtol=1e-6)

    def test_subsample_wider_than_std(self):
        """For p < 1, variance-corrected intervals should be wider than standard."""
        rng   = np.random.default_rng(1)
        res   = np.abs(rng.standard_normal(200))
        preds = rng.standard_normal(100)
        lo_vc, hi_vc   = conf_vc(res, preds, p=0.2, gamma=GAMMA_FIXED)
        lo_std, hi_std = conf_std(res, preds)
        assert np.all(hi_vc >= hi_std)
        assert np.all(lo_vc <= lo_std)

    def test_coverage_target(self):
        """Sanity-check that conf_vc achieves ≥90% on calibration residuals."""
        rng = np.random.default_rng(42)
        n   = 2000
        y   = rng.standard_normal(n)
        # Perfect model (pred = y) → residuals near 0, should always cover
        res   = np.zeros(500)
        preds = y
        lo, hi = conf_vc(res, preds, p=1.0)
        cov = np.mean((y >= lo) & (y <= hi))
        assert cov >= 1 - ALPHA - 0.01


class TestCalcMetrics:
    def test_perfect_coverage(self):
        y    = np.array([0.0, 1.0, 2.0])
        lo   = np.array([-10.0, -10.0, -10.0])
        hi   = np.array([10.0, 10.0, 10.0])
        m    = calc_metrics(y, lo, hi, y_std=1.0)
        assert m["coverage"] == pytest.approx(1.0)

    def test_zero_coverage(self):
        y    = np.array([0.0, 1.0, 2.0])
        lo   = np.array([5.0, 5.0, 5.0])
        hi   = np.array([6.0, 6.0, 6.0])
        m    = calc_metrics(y, lo, hi, y_std=1.0)
        assert m["coverage"] == pytest.approx(0.0)
        assert m["calib_error"] == pytest.approx(1.0 - ALPHA)

    def test_norm_width(self):
        y    = np.zeros(10)
        lo   = np.full(10, -1.0)
        hi   = np.full(10,  1.0)
        m    = calc_metrics(y, lo, hi, y_std=2.0)
        assert m["width"]      == pytest.approx(2.0)
        assert m["norm_width"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# sampling.py
# ---------------------------------------------------------------------------

class TestEffectiveRank:
    def test_single_frequency(self):
        """Pure sinusoid → effective rank should be very low (1 or 2)."""
        t  = np.linspace(0, 10, 1000)
        s  = np.sin(2 * np.pi * 5 * t)
        er = effective_rank(s, threshold=0.95)
        assert er <= 3  # one frequency, maybe a little leakage

    def test_white_noise_high_rank(self):
        """White noise → effective rank should be high."""
        rng = np.random.default_rng(0)
        s   = rng.standard_normal(1000)
        er  = effective_rank(s, threshold=0.95)
        assert er > 50

    def test_returns_at_least_one(self):
        s = np.zeros(100)
        assert effective_rank(s) >= 1


class TestAdaptiveRate:
    def test_min_rate_floor(self):
        """Low-rank series should hit the min_rate floor."""
        rate = adaptive_rate(T=10_000, er=1, min_rate=0.20)
        assert rate == pytest.approx(0.20)

    def test_full_rate_cap(self):
        """High-rank series should be capped at 1.0."""
        rate = adaptive_rate(T=100, er=100, c=2.0)
        assert rate == pytest.approx(1.0)

    def test_monotone_in_er(self):
        """Higher effective rank → higher (or equal) rate."""
        T = 5000
        rates = [adaptive_rate(T, er, min_rate=0.01) for er in [1, 10, 50, 200, 500]]
        assert rates == sorted(rates) or all(r == rates[0] for r in rates)
