"""Tests for Stage 13 counterparty credit exposure."""

import numpy as np
import pandas as pd
import pytest

from gasrisk.credit import exposure_profiles, score_ratio, scorecard, trade_table, trade_values

CURVE = pd.Series([0.0, 0.0], index=[0.25, 2.0])
GRADES = [{"grade": 1, "label": "a", "min_score": 4.0, "pfe_limit_eur": 10}, {"grade": 2, "label": "b", "min_score": 0.0, "pfe_limit_eur": 1}]
SC = {"interest_coverage": [0.5, True, [2, 4, 7, 12]], "net_debt_to_ebitda": [0.5, False, [4, 3, 2, 1]]}


def test_score_ratio_directions() -> None:
    assert score_ratio(13, True, [2, 4, 7, 12]) == 5 and score_ratio(1, True, [2, 4, 7, 12]) == 1
    assert score_ratio(0.5, False, [4, 3, 2, 1]) == 5 and score_ratio(5, False, [4, 3, 2, 1]) == 1


def test_scorecard_maps_to_grade() -> None:
    cps = [{"id": "X", "type": "t", "interest_coverage": 13, "net_debt_to_ebitda": 0.5},
           {"id": "Y", "type": "t", "interest_coverage": 1, "net_debt_to_ebitda": 5}]
    s = scorecard(cps, SC, GRADES).set_index("counterparty")
    assert s.loc["X", "grade"] == 1 and s.loc["Y", "grade"] == 2


def test_trade_value_sign_and_expiry() -> None:
    tr = trade_table([{"id": "X", "trades": [{"side": "buy", "start": 1, "end": 2, "mwh_per_month": 10, "strike_x_f": 1.0}]}])
    v = trade_values(tr, np.array([110.0]), 100.0, 0, CURVE)
    np.testing.assert_allclose(v, [[100.0, 100.0]])
    v1 = trade_values(tr, np.array([110.0]), 100.0, 1, CURVE)
    np.testing.assert_allclose(v1, [[0.0, 100.0]])  # month 1 has started delivery


def test_netting_never_increases_exposure_and_ce_is_today_value() -> None:
    cps = [{"id": "X", "trades": [{"side": "buy", "start": 1, "end": 3, "mwh_per_month": 10, "strike_x_f": 0.9},
                                  {"side": "sell", "start": 1, "end": 3, "mwh_per_month": 8, "strike_x_f": 1.0}]}]
    tr = trade_table(cps)
    rng = np.random.default_rng(0)
    paths = 100 * np.exp(np.column_stack([np.zeros(2000), np.cumsum(rng.normal(0, 0.1, (2000, 3)), axis=1)]))
    prof, _ = exposure_profiles(tr, paths, 100.0, CURVE, [0.95])
    assert (prof["ee_eur"] <= prof["ee_no_netting_eur"] + 1e-9).all()
    assert prof.loc[prof["month"] == 0, "ee_eur"].iloc[0] == pytest.approx(3 * (10 * 10 - 8 * 0))
