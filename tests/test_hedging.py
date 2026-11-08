"""Tests for Stage 11 hedging."""

import numpy as np
import pandas as pd
import pytest

from gasrisk.hedging import blocks, effectiveness, hedge_series, mv_ratio, strategy_pnl

HCFG = {"dollar_offset_band": [0.8, 1.25], "regression_r2_min": 0.8, "regression_slope_band": [0.8, 1.25],
        "cumulative_periods": 3, "collar_call_moneyness": 1.1, "collar_put_moneyness": 0.9, "fx_ratio": 0.7}


def _panel(n=300, seed=1, basis_sd=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    ttf = 30 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    fx = 4.3 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    g = ttf + 2 + np.cumsum(rng.normal(0, basis_sd, n))
    r = np.r_[np.nan, np.diff(np.log(ttf))]
    return pd.DataFrame({"ttf_c2": ttf, "ttf_risk_ret": r, "eurpln": fx, "tge_eur": g, "tge_pln": g * fx,
                         "eurpln_ret": np.r_[np.nan, np.diff(np.log(fx))]}, index=idx)


def test_mv_ratio_recovers_known_slope() -> None:
    rng = np.random.default_rng(0)
    f = rng.normal(size=20000)
    s = 0.8 * f + 0.3 * rng.normal(size=20000)
    assert mv_ratio(s, f) == pytest.approx(0.8, abs=0.01)


def test_ineffectiveness_identity_is_exact() -> None:
    d = hedge_series(_panel(basis_sd=0.5))
    np.testing.assert_allclose(d["dS"] - d["dF"], d["basis_term"] + d["fx_term"], atol=1e-9)


def test_blocks_are_non_overlapping_and_complete() -> None:
    d = hedge_series(_panel())
    m = blocks(d, 21)
    assert (m["n"] == 21).all() and m.index[-1] == d.index[-1]
    assert m["dS"].sum() == pytest.approx(d["dS"].iloc[-21 * len(m):].sum())


def test_zero_basis_gives_ineffectiveness_from_fx_only() -> None:
    m = blocks(hedge_series(_panel(basis_sd=0.0)), 21)
    eff, _, dec = effectiveness(m, HCFG)
    assert np.allclose(m["basis_term"], 0, atol=1e-9)
    with_fx = eff.set_index("hedge").loc["TTF futures and EUR forward"]
    assert with_fx["dollar_offset_pass_rate"] == pytest.approx(1.0)  # EUR forward removes the FX term exactly


def test_strategy_pnl_signs() -> None:
    lv = {"tge_eur": 40.0, "ttf": 35.0, "fx": 4.3}
    mi = {"T": 0.2, "sigma": 0.5}
    up = np.array([[0.1, 0.1, 0.0]])
    unhedged = strategy_pnl(lv, up, 1000.0, 0.0, 0.0, False, mi, HCFG, 21)[0]
    hedged = strategy_pnl(lv, up, 1000.0, 1000.0, 0.0, False, mi, HCFG, 21)[0]
    collar = strategy_pnl(lv, up, 1000.0, 0.0, 1000.0, False, mi, HCFG, 21)[0]
    assert unhedged < 0 and unhedged < collar < hedged
