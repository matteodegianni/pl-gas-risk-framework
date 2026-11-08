"""Tests for Stage 6 VaR backtests."""

import numpy as np
import pytest
from scipy import stats

from gasrisk.backtest import christoffersen, kupiec, traffic_light


def test_kupiec_matches_formula() -> None:
    n, x, p = 250, 8, 0.01
    e = np.zeros(n)
    e[:x] = 1
    lr, pv = kupiec(e, p)
    phat = x / n
    expected = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - phat) - x * np.log(phat))
    assert lr == pytest.approx(expected)
    assert pv == pytest.approx(stats.chi2.sf(expected, 1))


def test_kupiec_zero_exceptions_and_exact_rate() -> None:
    lr0, p0 = kupiec(np.zeros(1000), 0.01)
    assert lr0 == pytest.approx(-2 * 1000 * np.log(0.99))
    e = np.zeros(1000)
    e[::100] = 1  # exactly 1%
    assert kupiec(e, 0.01)[0] == pytest.approx(0.0, abs=1e-9)


def test_christoffersen_detects_clustering() -> None:
    clustered = np.zeros(1000)
    for s in (100, 400, 700):
        clustered[s:s + 4] = 1
    spread = np.zeros(1000)
    spread[np.linspace(50, 950, 12).astype(int)] = 1
    assert christoffersen(clustered, 0.01)["p_ind"] < 0.01
    assert christoffersen(spread, 0.01)["p_ind"] > 0.05
    r = christoffersen(spread, 0.01)
    assert r["lr_cc"] == pytest.approx(kupiec(spread, 0.01)[0] + r["lr_ind"])


def test_traffic_light_zones() -> None:
    assert [traffic_light(k) for k in (0, 4, 5, 9, 10)] == ["green", "green", "yellow", "yellow", "red"]
