"""Tests for Stage 9 P&L attribution: components must reconcile exactly to total P&L."""

import numpy as np
import pandas as pd
import pytest

from gasrisk.attribution import COMPONENTS, attribute, demand_profile


def _random_inputs(seed: int, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-01", periods=n)
    dem = demand_profile(idx, 500000, 0.35, 0.7, 0.08, seed)
    fx = 4.3 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    return dem.assign(tge_eur=40 * np.exp(np.cumsum(rng.normal(0, 0.05, n))), fx=fx,
                      d_ttf=rng.normal(0, 1.5, n), d_fx=np.diff(np.r_[4.3, fx]),
                      n_ttf=rng.uniform(0, 3e5, n), n_fx=rng.uniform(0, 1e7, n))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_components_reconcile_exactly(seed: int) -> None:
    df = _random_inputs(seed)
    att = attribute(df, S=200.0, P0=40.0, F0=4.3)
    np.testing.assert_allclose(att[COMPONENTS].sum(axis=1), att["total_pnl"], rtol=0, atol=1e-6)
    assert att["residual"].abs().max() < 1e-6


def test_effects_vanish_at_plan() -> None:
    df = _random_inputs(3)
    df = df.assign(actual_mwh=df["contracted_mwh"], tge_eur=40.0, fx=4.3, d_ttf=0.0, d_fx=0.0)
    att = attribute(df, S=200.0, P0=40.0, F0=4.3)
    for c in ("volume_effect", "price_effect", "fx_effect"):
        assert att[c].abs().max() < 1e-9
    np.testing.assert_allclose(att["total_pnl"], df["contracted_mwh"] * (200.0 - 40.0 * 4.3))


def test_signs_of_effects() -> None:
    df = _random_inputs(4).iloc[:5].assign(actual_mwh=lambda d: d["contracted_mwh"], d_ttf=0.0, d_fx=0.0, n_fx=0.0,
                                          fx=4.3, tge_eur=44.0)
    att = attribute(df, S=200.0, P0=40.0, F0=4.3)
    assert (att["price_effect"] < 0).all()  # TGE above plan costs money
    df2 = df.assign(fx=4.5, tge_eur=40.0)
    assert (attribute(df2, 200.0, 40.0, 4.3)["fx_effect"] < 0).all()  # weaker PLN raises purchase cost


def test_demand_profile_sums_to_total_and_peaks_in_winter() -> None:
    idx = pd.date_range("2025-01-01", "2025-12-31")
    d = demand_profile(idx, 500000, 0.35, 0.7, 0.08, 0)
    assert d["contracted_mwh"].sum() == pytest.approx(500000)
    assert d.loc["2025-01", "contracted_mwh"].mean() > d.loc["2025-07", "contracted_mwh"].mean()
