"""Tests for Stage 4 volatility models and evaluation."""

import numpy as np
import pandas as pd
import pytest

from gasrisk.volatility import (
    MeanSpec,
    aggregate_variance,
    evaluate_vol,
    ewma_variance,
    fit_summary,
    garch_variance_path,
    mincer_zarnowitz,
    psi_weights,
    qlike,
    select_dist,
    select_mean,
    standardized_cdf,
    standardized_ppf,
    walk_forward_vol,
)

CONST = MeanSpec("constant", 0, False)

CFG = {"distributions": ["normal", "t"], "ewma_lambda": 0.94, "rolling_window": 63, "train_window": 400,
       "refit_every": 50, "horizons": [1, 5]}


def _garch_sim(n: int, seed: int, nu: float | None = None) -> pd.Series:
    rng = np.random.default_rng(seed)
    omega, a, b = 0.05, 0.08, 0.90
    h = np.full(n, omega / (1 - a - b))
    r = np.zeros(n)
    for t in range(n):
        if t:
            h[t] = omega + a * r[t - 1] ** 2 + b * h[t - 1]
        z = rng.standard_t(nu) / np.sqrt(nu / (nu - 2)) if nu else rng.standard_normal()
        r[t] = np.sqrt(h[t]) * z
    return pd.Series(r, index=pd.bdate_range("2010-01-01", periods=n))


def test_ewma_recursion() -> None:
    r = np.array([1.0, -2.0, 0.5, 3.0, -1.0])
    s2 = ewma_variance(r, 0.9, init_window=2)
    assert s2[2] == pytest.approx((1 + 4) / 2)
    assert s2[3] == pytest.approx(0.9 * 2.5 + 0.1 * 0.25)
    assert s2[4] == pytest.approx(0.9 * s2[3] + 0.1 * 9.0)
    assert np.isnan(s2[1])


def test_garch_t_recovers_parameters_and_is_preferred_for_t_data() -> None:
    r = _garch_sim(4000, seed=1, nu=5.0)
    row_t, vol, _ = fit_summary(r, "t", CONST, [5, 10])
    row_n, _, _ = fit_summary(r, "normal", CONST, [5, 10])
    assert row_t["aic"] < row_n["aic"]
    assert 3.5 < row_t["nu"] < 8.0
    assert row_t["persistence"] == pytest.approx(0.98, abs=0.03)
    assert abs(row_t["share_below_q01"] - 0.01) < 0.006
    assert len(vol) == len(r)


def test_standardized_t_has_unit_variance_quantiles() -> None:
    nu = 6.0
    # The standardised-t 1% quantile is less extreme than the raw t quantile, more than the normal one.
    q = standardized_ppf(0.01, "t", [nu])
    assert standardized_ppf(0.01, "normal", []) > q
    assert standardized_cdf(np.array([q]), "t", [nu])[0] == pytest.approx(0.01)
    qs = standardized_ppf(0.01, "skewt", [nu, -0.2])
    assert standardized_cdf(np.array([qs]), "skewt", [nu, -0.2])[0] == pytest.approx(0.01, abs=1e-6)


def test_qlike_minimised_at_true_variance() -> None:
    rv = 2.0
    grid = np.linspace(0.5, 5, 451)
    assert grid[np.argmin(qlike(np.full_like(grid, rv), grid))] == pytest.approx(rv, abs=0.01)


def test_mincer_zarnowitz_unbiased_forecast() -> None:
    rng = np.random.default_rng(3)
    f = pd.Series(rng.uniform(1, 3, 3000))
    rv = f * rng.chisquare(1, 3000)  # E[rv | f] = f
    mz = mincer_zarnowitz(rv, f)
    assert mz["mz_b"] == pytest.approx(1.0, abs=0.15)


def test_walk_forward_vol_alignment_and_no_look_ahead() -> None:
    r = _garch_sim(560, seed=4)
    fc = walk_forward_vol(r, CFG)
    assert set(fc["model"]) == {"garch_normal", "garch_t", "ewma", "rolling"}
    # Realised variance for h=1 at origin t is r[t+1]^2.
    one = fc[(fc["h"] == 1) & (fc["model"] == "ewma")].iloc[0]
    pos = r.index.get_loc(one["origin"])
    assert one["realised_var"] == pytest.approx(r.iloc[pos + 1] ** 2)
    # Forecasts at origins up to t do not change when returns after t change.
    T = 480
    r2 = r.copy()
    r2.iloc[T + 1:] *= 3.0
    fc2 = walk_forward_vol(r2, CFG)
    cut = r.index[T]
    a = fc[fc["origin"] <= cut].reset_index(drop=True)
    b = fc2[fc2["origin"] <= cut].reset_index(drop=True)
    np.testing.assert_allclose(a["forecast_var"], b["forecast_var"])


def test_evaluate_vol_prefers_true_model() -> None:
    r = _garch_sim(900, seed=5, nu=6.0)
    fc = walk_forward_vol(r, CFG).assign(series="x")
    ev = evaluate_vol(fc, "rolling", 0.05)
    one = ev[ev["h"] == 1].set_index("model")
    assert one.loc["garch_t", "qlike"] < one.loc["rolling", "qlike"]
    assert one.loc["rolling", "verdict_qlike"] == "benchmark"


def test_psi_weights_and_aggregate_variance_ar1() -> None:
    phi = np.array([0.5])
    np.testing.assert_allclose(psi_weights(phi, 4), [1, 0.5, 0.25, 0.125])
    # h=2, constant sigma2=1: forecast error = eps1*(1+0.5) + eps2 -> variance 1.5^2 + 1
    assert aggregate_variance(np.ones(2), phi, 2) == pytest.approx(1.5**2 + 1)
    assert aggregate_variance(np.array([2.0, 3.0]), np.zeros(0), 2) == pytest.approx(5.0)  # no AR: sum of variances


def test_garch_variance_path() -> None:
    params = pd.Series({"omega": 0.1, "alpha[1]": 0.1, "beta[1]": 0.8})
    path = garch_variance_path(params, resid_last=2.0, sigma2_last=1.0, H=3)
    assert path[0] == pytest.approx(0.1 + 0.1 * 4 + 0.8)
    assert path[1] == pytest.approx(0.1 + 0.9 * path[0])


def test_select_mean_removes_ar_structure() -> None:
    rng = np.random.default_rng(8)
    g = _garch_sim(2500, seed=8, nu=6.0).to_numpy()
    y = np.zeros_like(g)
    for t in range(1, len(g)):
        y[t] = -0.35 * y[t - 1] + g[t]
    r = pd.Series(y, index=pd.bdate_range("2010-01-01", periods=len(y)))
    cands = [CONST, MeanSpec("ar1", (1,), False), MeanSpec("ar5", 5, False)]
    spec, tbl = select_mean(r, cands, [5, 10], 0.05)
    assert spec.name == "ar1"  # constant fails Ljung-Box, AR(1) passes with lower BIC than AR(5)
    assert not tbl.set_index("mean").loc["constant", "lb_passes"]


def test_select_dist_rules() -> None:
    agree = pd.DataFrame({"dist": ["normal", "t"], "bic": [10.0, 5.0], "calib_error_1pct": [0.02, 0.001]})
    assert select_dist(agree)[0] == "t"
    disagree = pd.DataFrame({"dist": ["t", "skewt"], "bic": [5.0, 6.0], "calib_error_1pct": [0.01, 0.002]})
    d, reason = select_dist(disagree)
    assert d == "skewt" and "calibration chosen" in reason


def test_walk_forward_with_ar_mean_runs_and_ignores_future() -> None:
    r = _garch_sim(560, seed=9)
    spec = MeanSpec("ar1", (1,), False)
    fc = walk_forward_vol(r, CFG, spec, ["t"])
    r2 = r.copy()
    r2.iloc[481:] *= 3.0
    fc2 = walk_forward_vol(r2, CFG, spec, ["t"])
    cut = r.index[480]
    np.testing.assert_allclose(fc[fc["origin"] <= cut]["forecast_var"], fc2[fc2["origin"] <= cut]["forecast_var"])
