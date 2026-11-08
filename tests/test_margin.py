"""Tests for Stage 12 margin and the shared TTF simulator."""

import numpy as np
import pandas as pd
import pytest

from gasrisk.margin import historical_margin, im_from_vol, simulate_margin
from gasrisk.ttf_sim import TTFGarch, simulate_returns


def _model(z=None):
    rng = np.random.default_rng(3)
    z = rng.standard_normal(5000) if z is None else z
    z = (z - z.mean()) / z.std()
    return TTFGarch(omega=2e-5, alpha=0.08, beta=0.9, nu=5.0, sigma2_next=0.03**2, std_resid=z, mean_name="constant",
                    params=pd.Series(dtype=float), spec=None, resid_last=0.0, sigma2_last=9.0)


@pytest.mark.parametrize("innov", ["fhs", "t"])
def test_simulated_futures_price_is_driftless(innov) -> None:
    r, s2 = simulate_returns(_model(), 200000, 10, np.random.default_rng(0), innov)
    assert np.exp(r.sum(axis=1)).mean() == pytest.approx(1.0, abs=0.005)
    assert s2[:, 0] == pytest.approx(0.03**2)


def test_im_formula_scales_with_position_price_and_vol() -> None:
    a = im_from_vol(100.0, 50.0, 0.02, -2.33, 2)
    assert a == pytest.approx(100 * 50 * (1 - np.exp(np.sqrt(2) * 0.02 * -2.33)))
    assert im_from_vol(200.0, 50.0, 0.02, -2.33, 2) == pytest.approx(2 * a)
    assert im_from_vol(100.0, 50.0, 0.04, -2.33, 2) > a


def test_margin_calls_add_up() -> None:
    cfg = {"buffer_horizon_days": 5, "n_paths": 1000, "im_mpor_days": 2}
    sim = simulate_margin(_model(), 1000.0, 40.0, cfg, -2.4, np.random.default_rng(1))
    np.testing.assert_allclose(sim["cum"][:, -1], (-sim["vm"]).sum(1) + sim["im"][:, -1] - sim["im0"])


def test_ttf_fall_creates_a_margin_call_for_a_long_position() -> None:
    idx = pd.bdate_range("2022-01-03", periods=4)
    panel = pd.DataFrame({"ttf_c2": [100.0, 90.0, 90.0, 90.0], "ttf_risk_ret": [np.nan, np.log(0.9), 0.0, 0.0]}, index=idx)
    cvol = pd.Series(2.0, index=idx)
    h = historical_margin(panel, cvol, np.array([-2.0, 0.0, 2.0]), 10.0, {"im_confidence": 0.99, "im_mpor_days": 2})
    assert h.loc[idx[1], "vm_eur"] == pytest.approx(-100.0)
    assert h.loc[idx[1], "margin_call_eur"] > 0
