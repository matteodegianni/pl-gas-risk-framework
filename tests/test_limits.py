"""Tests for Stage 7 limits and breach analysis."""

import numpy as np
import pandas as pd
import pytest

from gasrisk.limits import breach_analysis, choose_official, episodes, euler_components, sensitivities
from gasrisk.var import pnl_legs, positions

BOOK = {"q_open_mwh": 500000, "ttf_hedge_ratio": 0.7, "fx_hedge_ratio": 0.7}


def test_sensitivities_match_full_revaluation_derivative() -> None:
    row = pd.Series({"level_tge_eur": 40.0, "level_ttf": 35.0, "level_fx": 4.3})
    s = sensitivities(row, BOOK)
    lv = {"tge_eur": 40.0, "ttf": 35.0, "fx": 4.3}
    pos = positions(BOOK, 40.0)
    eps = 1e-6
    for i in range(3):
        shock = np.zeros((1, 3))
        shock[0, i] = eps
        num = pnl_legs(lv, shock, pos)["total"][0] / eps
        assert s[i] == pytest.approx(num, rel=1e-4)


def test_euler_components_sum_to_total() -> None:
    s = np.array([-1e6, 6e5, -2e5])
    sig = np.array([0.05, 0.04, 0.004])
    corr = np.array([[1, 0.5, 0.1], [0.5, 1, 0.05], [0.1, 0.05, 1]])
    c = euler_components(s, sig, corr, total_var=123.0)
    assert c.sum() == pytest.approx(123.0)


def test_episodes() -> None:
    idx = pd.bdate_range("2024-01-01", periods=8)
    flag = pd.Series([False, True, True, False, True, False, True, True], index=idx)
    assert episodes(flag) == [(idx[1], idx[2], 2), (idx[4], idx[4], 1), (idx[6], idx[7], 2)]


def test_choose_official_rule() -> None:
    bt = pd.DataFrame({"method": ["a", "a", "b", "b", "c", "c"], "pass_cc": [True, True, True, True, True, False]})
    tl = pd.DataFrame({"method": ["a", "b", "c"], "share_windows_green": [0.5, 0.9, 1.0]})
    m, reason = choose_official(bt, tl)
    assert m == "b" and "a, b" in reason


def test_exposure_driver_split_is_exact() -> None:
    idx = pd.bdate_range("2024-01-01", periods=6)
    u = pd.DataFrame({
        "util_var_99": [0.5] * 6, "util_open_gas": [0.7, 0.9, 1.2, 1.3, 0.5, 0.5], "util_fx_net_open": [0.5] * 6,
        "level_tge_eur": [40, 45, 50, 52, 30, 30], "level_fx": [4.3, 4.3, 4.4, 4.4, 4.3, 4.3],
    }, index=idx, dtype=float)
    br = breach_analysis(u, lookback=3, amber=0.8)
    row = br.iloc[0]
    assert row["limit"] == "open_gas" and row["days"] == 2
    assert row["share_tge_eur_at_start"] + row["share_fx_at_start"] == pytest.approx(1.0)
    assert row["share_tge_eur_at_start"] == pytest.approx(np.log(50 / 40) / (np.log(50 / 40) + np.log(4.4 / 4.3)))
