"""Leakage tests for Stage 3.

Timing convention (forecast made after the close of day t):

* TGEgasDA trade date t: volume weighted price of trades on day t (known end of day t)
* TTF settlement of day t: around 17:00 CET (known end of day t)
* ECB EUR/PLN of day t: early afternoon CET (known end of day t)
* target: value at t+h with h >= 1 (for TGE at h=1: trade date t+1)

So every feature at row t may use day-t values of TGE, TTF and ECB, but nothing
dated t+1 or later. These tests check that mechanically.
"""

import numpy as np
import pandas as pd
import pytest

from gasrisk.forecasting import (
    ArimaModel,
    DirectRegression,
    EtsModel,
    RandomWalk,
    build_series,
    walk_forward,
)

N, T = 160, 120  # panel length and the "forecast day" position used in perturbation tests


def _panel(seed: int = 0, n: int = N) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n)
    ttf_ret = rng.normal(0, 0.03, n)
    ttf = 30 * np.exp(np.cumsum(ttf_ret))
    fx = 4.4 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    tge = 140 * np.exp(np.cumsum(rng.normal(0, 0.04, n)))
    p = pd.DataFrame({"ttf_risk_ret": ttf_ret, "ttf_c2": ttf, "eurpln": fx, "tge_pln": tge}, index=idx)
    p["tge_eur"] = p["tge_pln"] / p["eurpln"]
    p["basis_tge_ttfc2_eur"] = p["tge_eur"] - p["ttf_c2"]
    return p


def _perturb_after(panel: pd.DataFrame, pos: int, cols: list[str] | None = None, seed: int = 99) -> pd.DataFrame:
    """Replace every value strictly after position ``pos`` (in ``cols``) with different data."""
    q = panel.copy()
    rng = np.random.default_rng(seed)
    for c in cols or ["ttf_risk_ret", "ttf_c2", "eurpln", "tge_pln"]:
        q.iloc[pos + 1:, q.columns.get_loc(c)] *= np.exp(rng.normal(0, 0.2, len(q) - pos - 1))
    q["tge_eur"] = q["tge_pln"] / q["eurpln"]
    q["basis_tge_ttfc2_eur"] = q["tge_eur"] - q["ttf_c2"]
    return q


def _frames(sd) -> dict[str, pd.DataFrame]:
    out = {"y": sd.y.to_frame(), "features": sd.features, "dow": sd.dow}
    if sd.ecm_features is not None:
        out["linear"] = sd.ecm_features
    return out


@pytest.mark.parametrize("cross_lag", [0, 1])
def test_no_feature_at_t_depends_on_data_after_t(cross_lag: int) -> None:
    base = {s.name: s for s in build_series(_panel(), cross_lag)}
    pert = {s.name: s for s in build_series(_perturb_after(_panel(), T), cross_lag)}
    for name in base:
        for key, frame in _frames(base[name]).items():
            other = _frames(pert[name])[key]
            pd.testing.assert_frame_equal(frame.iloc[: T + 1], other.iloc[: T + 1], obj=f"{name}.{key} up to t")


def test_tge_features_use_same_day_ttf_and_fx_and_placebo_removes_them() -> None:
    """Changing TTF and FX on day t changes TGE inputs at t (allowed: settled before the forecast);
    with the placebo lag it no longer does."""
    p = _panel()
    q = p.copy()
    q.iloc[T, q.columns.get_loc("ttf_c2")] *= 1.3
    q.iloc[T, q.columns.get_loc("ttf_risk_ret")] += np.log(1.3)
    q.iloc[T, q.columns.get_loc("eurpln")] *= 1.02
    q["tge_eur"] = q["tge_pln"] / q["eurpln"]
    q["basis_tge_ttfc2_eur"] = q["tge_eur"] - q["ttf_c2"]
    cross = ["ttf_chg", "eurpln_chg", "z_tge_ttf"]
    a = {s.name: s for s in build_series(p, 0)}["tge"].ecm_features
    b = {s.name: s for s in build_series(q, 0)}["tge"].ecm_features
    assert not np.allclose(a.iloc[T][cross], b.iloc[T][cross])
    a1 = {s.name: s for s in build_series(p, 1)}["tge"].ecm_features
    b1 = {s.name: s for s in build_series(q, 1)}["tge"].ecm_features
    np.testing.assert_allclose(a1.iloc[T][cross], b1.iloc[T][cross])
    # TGE's own inputs and the target are untouched by the placebo
    np.testing.assert_allclose(a1.iloc[T][["tge_chg", "tge_chg_lag1"]], a.iloc[T][["tge_chg", "tge_chg_lag1"]])


def test_target_is_strictly_after_origin() -> None:
    sd = {s.name: s for s in build_series(_panel())}["tge"]
    cfg = {"horizons": [1, 5], "train_window": 100, "refit_every": 20, "arima_orders": [[0, 0]],
           "gbm": {"max_iter": 20, "max_depth": 2, "min_samples_leaf": 10}}
    fc = walk_forward(sd, cfg, 0, [RandomWalk()])
    assert (fc["target_date"] > fc["origin"]).all()
    pos = {d: i for i, d in enumerate(sd.y.index)}
    assert all(pos[t] - pos[o] == h for o, t, h in zip(fc["origin"], fc["target_date"], fc["h"]))


def test_walk_forward_predictions_ignore_data_after_origin() -> None:
    """Every model's forecast at origin t is identical whether data after t is real or perturbed."""
    cfg = {"horizons": [1, 5], "train_window": 80, "refit_every": 15, "arima_orders": [[0, 0], [1, 0]],
           "gbm": {"max_iter": 30, "learning_rate": 0.1, "max_depth": 2, "min_samples_leaf": 10, "l2_regularization": 0.0}}
    p, q = _panel(n=150), _perturb_after(_panel(n=150), T)
    for name in ("tge", "basis"):
        sa = {s.name: s for s in build_series(p)}[name]
        sb = {s.name: s for s in build_series(q)}[name]
        for make in (RandomWalk, ArimaModel, EtsModel, lambda: DirectRegression("ecm"),
                     lambda: DirectRegression("gbm", cfg["gbm"], 0)):
            fa = walk_forward(sa, cfg, 0, [make()])
            fb = walk_forward(sb, cfg, 0, [make()])
            ka = fa[fa["origin"] <= sa.y.index[T]].reset_index(drop=True)
            kb = fb[fb["origin"] <= sb.y.index[T]].reset_index(drop=True)
            assert len(ka) > 0
            np.testing.assert_allclose(ka["y_pred"], kb["y_pred"], err_msg=f"{name} {ka['model'].iloc[0]}")
