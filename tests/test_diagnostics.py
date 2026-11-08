"""Tests for Stage 2 diagnostics, including the FX timing method on simulated asynchronous data."""

import numpy as np
import pandas as pd
import pytest

from gasrisk.diagnostics import (
    SeriesSpec,
    capture_ratio,
    cross_correlations,
    distribution_row,
    fx_pair_leadlag,
    fx_timing,
    robust_portmanteau,
    serial_dependence_table,
    stationarity_table,
    stationarity_verdict,
)


def _intraday_paths(n_days: int = 3000, steps: int = 24, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Common FX factor and a gas path that loads on it, both on an intraday grid."""
    rng = np.random.default_rng(seed)
    fx_inc = rng.normal(0, 1, n_days * steps)
    gas_inc = 0.6 * fx_inc + 0.8 * rng.normal(0, 1, n_days * steps)
    return np.cumsum(fx_inc), np.cumsum(gas_inc)


def _sample(path: np.ndarray, step: int, steps: int = 24) -> pd.Series:
    idx = pd.bdate_range("2000-01-03", periods=len(path) // steps)
    level = path[step::steps][: len(idx)]
    return pd.Series(np.diff(level, prepend=np.nan), index=idx)


def test_fx_timing_ranks_the_source_closest_to_the_anchor_first() -> None:
    fx_path, gas_path = _intraday_paths()
    anchor = _sample(gas_path, 17)  # "TTF settles at hour 17"
    sources = {"early": _sample(fx_path, 11), "close": _sample(fx_path, 14), "late_next_day": _sample(fx_path, 23)}
    table, pairs = fx_timing(anchor, sources, max_lag=2, reps=200, block=20, seed=1)
    t = table.set_index("source")
    assert t["capture_ratio"].idxmax() == "close"
    assert t.loc["close", "capture_ratio"] > t.loc["early", "capture_ratio"]
    # An earlier source leaks into FX_t+1, a later one into FX_t-1.
    assert t.loc["early", "corr_lag_+1"] > t.loc["early", "corr_lag_-1"]
    assert t.loc["late_next_day", "corr_lag_-1"] > t.loc["late_next_day", "corr_lag_+1"]
    assert set(pairs.columns) >= {"comparison", "ci_low", "ci_high", "share_positive"}


def test_fx_pair_leadlag_identifies_sampling_order_and_offset() -> None:
    fx_path, _ = _intraday_paths()
    a, b = _sample(fx_path, 8), _sample(fx_path, 14)
    row = fx_pair_leadlag({"a": a, "b": b}).iloc[0]
    assert row["sampled_earlier"] == "a" and row["sampled_later"] == "b"
    assert not row["date_offset_suspected"]
    # A quote taken at hour 1 (previous evening in European terms) vs hour 14: lagged > same-day.
    c = _sample(fx_path, 1)
    row2 = fx_pair_leadlag({"b": b, "c": c}).iloc[0]
    assert row2["sampled_earlier"] == "c" and bool(row2["date_offset_suspected"])


def test_capture_ratio_and_cross_correlations() -> None:
    cc = pd.Series({-1: 0.1, 0: 0.6, 1: 0.3})
    assert capture_ratio(cc) == pytest.approx(0.6)
    x = pd.Series(np.random.default_rng(0).normal(size=500))
    assert cross_correlations(x, x, 1).loc[0] == pytest.approx(1.0)


@pytest.mark.parametrize("adf_p, kpss_p, verdict", [
    (0.01, 0.10, "stationary"), (0.40, 0.01, "unit root"),
    (0.01, 0.01, "conflicting (breaks or long memory)"), (0.40, 0.10, "inconclusive"),
])
def test_stationarity_verdict(adf_p: float, kpss_p: float, verdict: str) -> None:
    assert stationarity_verdict(adf_p, kpss_p, 0.05) == verdict


def test_random_walk_level_is_unit_root_and_returns_stationary() -> None:
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2015-01-01", periods=1500)
    r = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    spec = SeriesSpec("rw", "random walk", r.cumsum(), r, "level", "return")
    st = stationarity_table([spec], 0.05).set_index("transform")
    assert st.loc["level", "verdict"] == "unit root"
    assert st.loc["return", "verdict"] == "stationary"


def _garch(n: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    e = np.zeros(n)
    h = np.full(n, 1e-4)
    for t in range(1, n):
        h[t] = 2e-6 + 0.10 * e[t - 1] ** 2 + 0.88 * h[t - 1]
        e[t] = np.sqrt(h[t]) * rng.standard_normal()
    return pd.Series(e, index=pd.bdate_range("2010-01-01", periods=n))


def test_arch_detected_and_robust_portmanteau_keeps_size_under_garch() -> None:
    r = _garch(3000, seed=11)
    spec = SeriesSpec("g", "garch", r.cumsum(), r, "level", "return")
    sd = serial_dependence_table([spec], [5, 10], 5).iloc[0]
    assert sd["arch_lm_p_5"] < 0.01
    assert sd["robust_q_ret_p_10"] > 0.05  # no autocorrelation in the mean: robust test does not reject


def test_robust_portmanteau_detects_autocorrelation() -> None:
    rng = np.random.default_rng(5)
    e = rng.normal(size=2000)
    x = np.zeros(2000)
    for t in range(1, 2000):
        x[t] = 0.3 * x[t - 1] + e[t]
    q, p = robust_portmanteau(pd.Series(x), [5])[5]
    assert p < 0.001


def test_distribution_row_normal_sample() -> None:
    r = pd.Series(np.random.default_rng(9).normal(0, 0.02, 20000))
    d = distribution_row(r)
    assert abs(d["excess_kurtosis"]) < 0.15
    assert d["share_abs_z_gt_3"] == pytest.approx(d["normal_share_abs_z_gt_3"], abs=0.0015)
    assert d["vol_annualised"] == pytest.approx(0.02 * np.sqrt(252), rel=0.03)
