"""Tests for Stage 5 VaR/ES of the synthetic book."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from gasrisk.var import (
    FACTORS,
    copula_shocks,
    fit_copula_nu,
    mean_forecast,
    pnl_legs,
    positions,
    run_var,
    var_es,
)
from gasrisk.volatility import MeanSpec

BOOK = {"q_open_mwh": 500000, "ttf_hedge_ratio": 0.7, "fx_hedge_ratio": 0.7}
LV = {"tge_eur": 40.0, "ttf": 35.0, "fx": 4.3}


def _shock(tge=0.0, ttf=0.0, fx=0.0) -> np.ndarray:
    return np.array([[tge, ttf, fx]])


def test_sign_conventions_and_legs_sum_to_total() -> None:
    pos = positions(BOOK, LV["tge_eur"])
    up = 0.05
    # TGE rise loses on the short floating position (and nothing else moves)
    legs = pnl_legs(LV, _shock(tge=up), pos)
    assert legs["tge"][0] < 0 and legs["ttf"][0] == 0 and legs["fx"][0] == 0
    # TTF rise gains on the long futures
    legs = pnl_legs(LV, _shock(ttf=up), pos)
    assert legs["ttf"][0] > 0 and legs["tge"][0] == 0
    # EUR/PLN rise gains on the forward and raises the PLN cost of the TGE position
    legs = pnl_legs(LV, _shock(fx=up), pos)
    assert legs["fx"][0] > 0 and legs["tge"][0] < 0
    # the three legs sum to the book total for any scenario
    rng = np.random.default_rng(0)
    sims = rng.normal(0, 0.05, (1000, 3))
    legs = pnl_legs(LV, sims, pos)
    np.testing.assert_allclose(legs["tge"] + legs["ttf"] + legs["fx"], legs["total"])


def test_tge_leg_value_is_tge_eur_times_fx() -> None:
    pos = positions(BOOK, LV["tge_eur"])
    legs = pnl_legs(LV, _shock(tge=np.log(1.1), fx=np.log(1.02)), pos)
    expected = -BOOK["q_open_mwh"] * (40.0 * 1.1 * 4.3 * 1.02 - 40.0 * 4.3)
    assert legs["tge"][0] == pytest.approx(expected)


def test_var_es_definitions() -> None:
    pnl = -np.arange(1, 1001, dtype=float)  # losses 1..1000
    r = var_es(pnl, [0.99], [0.975, 0.99])
    assert r["var_99"] == pytest.approx(np.quantile(np.arange(1, 1001), 0.99))
    assert r["es_99"] >= r["var_99"] and r["es_97.5"] <= r["es_99"]


def test_t_copula_has_more_joint_tail_events_than_gaussian() -> None:
    rng = np.random.default_rng(1)
    corr = np.array([[1, 0.5], [0.5, 1]])
    ug = copula_shocks(rng, corr, 200000, "gauss", None)
    ut = copula_shocks(rng, corr, 200000, "t", 4.0)
    joint = lambda u: np.mean((u[:, 0] < 0.01) & (u[:, 1] < 0.01))  # noqa: E731
    assert joint(ut) > 1.5 * joint(ug)


def test_copula_df_is_recovered() -> None:
    rng = np.random.default_rng(2)
    corr = np.array([[1, 0.4, 0.2], [0.4, 1, 0.1], [0.2, 0.1, 1]])
    u = copula_shocks(rng, corr, 4000, "t", 5.0)
    nu = fit_copula_nu(stats.norm.ppf(u), corr, [3, 4, 5, 6, 8, 10, 20, 50])
    assert 3 <= nu <= 8
    u2 = copula_shocks(rng, corr, 4000, "gauss", None)
    assert fit_copula_nu(stats.norm.ppf(u2), corr, [3, 4, 5, 6, 8, 10, 20, 50]) >= 10


def test_mean_forecast_ar_and_dow() -> None:
    params = pd.Series({"Const": 0.1, "y[1]": -0.3, "y[2]": 0.05, "dow_2": 0.4, "omega": 0.1,
                        "alpha[1]": 0.1, "beta[1]": 0.8, "nu": 5.0})
    spec = MeanSpec("ar2_dow", (1, 2), True)
    hist = np.array([0.0, 1.0, 2.0])
    assert mean_forecast(params, spec, hist, next_dow=2) == pytest.approx(0.1 - 0.3 * 2.0 + 0.05 * 1.0 + 0.4)
    assert mean_forecast(params, spec, hist, next_dow=0) == pytest.approx(0.1 - 0.6 + 0.05)


def test_run_var_no_look_ahead_and_shapes() -> None:
    rng = np.random.default_rng(3)
    n = 190
    idx = pd.bdate_range("2020-01-01", periods=n)
    rets = pd.DataFrame(rng.standard_t(6, (n, 3)) * [0.04, 0.03, 0.003], index=idx, columns=FACTORS)
    levels = pd.DataFrame(np.exp(np.cumsum(rets.to_numpy(), 0)) * [40, 35, 4.3], index=idx, columns=FACTORS)
    const = (MeanSpec("constant", 0, False), "t")
    specs = {f: const for f in FACTORS}
    cfg = {"hs_window": 100, "normal_window": 60, "garch_window": 120, "refit_every": 20, "ewma_lambda": 0.94,
           "mc_sims": 2000, "var_levels": [0.95, 0.99], "es_levels": [0.95, 0.975, 0.99], "start_obs": 150,
           "copula_nu_grid": [4, 8, 30]}
    a = run_var(levels, rets, specs, BOOK, cfg, seed=0)
    assert set(a["method"]) == {"hs", "normal", "garch_t_gauss", "garch_t_tcop", "fhs", "ewma"}
    assert (a["es_99"] >= a["var_99"] - 1e-6).all()
    T = 170
    rets2, levels2 = rets.copy(), levels.copy()
    rets2.iloc[T + 1:] *= 4.0
    levels2.iloc[T + 1:] *= 2.0
    b = run_var(levels2, rets2, specs, BOOK, cfg, seed=0)
    cut = idx[T]
    ka = a[a["origin"] <= cut].reset_index(drop=True)
    kb = b[b["origin"] <= cut].reset_index(drop=True)
    np.testing.assert_allclose(ka["var_99"], kb["var_99"])
    np.testing.assert_allclose(ka["es_97.5"], kb["es_97.5"])
