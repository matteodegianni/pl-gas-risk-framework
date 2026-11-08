"""Tests for Stage 8 stress testing and sensitivities."""

import numpy as np
import pandas as pd
import pytest

from gasrisk.stress import level_shock_returns, reverse_stress, sensitivities, worst_windows
from gasrisk.var import pnl_legs, positions

BOOK = {"q_open_mwh": 500000, "ttf_hedge_ratio": 0.7, "fx_hedge_ratio": 0.7}
LV = {"tge_eur": 40.0, "ttf": 35.0, "fx": 4.3}


def test_sensitivities_are_exact_linear_values() -> None:
    pos = positions(BOOK, LV["tge_eur"])
    s = sensitivities(LV, pos).set_index("scenario")["pnl_total"]
    assert s["+1 PLN/MWh TGE"] == pytest.approx(-500000.0)
    assert s["+1 EUR/MWh TTF"] == pytest.approx(0.7 * 500000 * 4.3)
    assert s["+0.01 EUR/PLN"] == pytest.approx(-500000 * 40.0 * 0.01 + pos["fx_eur"] * 0.01)


def test_level_shock_returns() -> None:
    r = level_shock_returns(LV, d_tge_pln=43.0, d_ttf_eur=3.5, d_fx=0.43)
    np.testing.assert_allclose(np.exp(r), [1 + 10.0 / 40.0, 1.1, 1.1])


def test_worst_window_is_the_minimum_pnl_window() -> None:
    idx = pd.bdate_range("2022-01-03", periods=30)
    rets = pd.DataFrame(0.0, index=idx, columns=["tge_eur", "ttf", "fx"])
    rets.iloc[10:13, 0] = 0.2  # TGE up 3 days in a row: worst for a short TGE book
    pos = positions(BOOK, LV["tge_eur"])
    w = worst_windows(rets, LV, pos, ["2022-01-03", "2022-12-31"], [3]).iloc[0]
    assert str(w["from"]) == str(idx[10].date()) and str(w["to"]) == str(idx[12].date())
    assert w["pnl_total"] == pytest.approx(pnl_legs(LV, np.array([[0.6, 0, 0]]), pos)["total"][0])


def test_reverse_stress_loses_exactly_the_target() -> None:
    pos = positions(BOOK, LV["tge_eur"])
    rs = reverse_stress(LV, pos, 25e6).set_index("scenario")
    x = rs.iloc[0]["required_move"]
    assert pnl_legs(LV, np.array([[np.log1p(x), 0, 0]]), pos)["total"][0] == pytest.approx(-25e6, rel=1e-6)
