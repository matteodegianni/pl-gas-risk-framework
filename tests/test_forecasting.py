"""Tests for the Stage 3 forecasting engine and evaluation."""

import numpy as np
import pandas as pd
import pytest

from gasrisk.forecasting import (
    ArimaModel,
    DirectRegression,
    RandomWalk,
    SeriesData,
    _dow_dummies,
    diebold_mariano,
    directional_accuracy,
    evaluate,
    training_rows,
    walk_forward,
)

CFG = {"horizons": [1, 5], "train_window": 120, "refit_every": 20, "alpha": 0.05,
       "arima_orders": [[0, 0], [1, 0]],
       "gbm": {"max_iter": 50, "learning_rate": 0.1, "max_depth": 2, "min_samples_leaf": 10, "l2_regularization": 0.0}}


def _series(y: np.ndarray, is_log: bool = True, on_level: bool = False) -> SeriesData:
    idx = pd.bdate_range("2020-01-01", periods=len(y))
    s = pd.Series(y, index=idx)
    feats = pd.DataFrame({"chg": s.diff(), "chg_lag1": s.diff().shift(1), "level": s}, index=idx)
    return SeriesData("x", s, is_log, on_level, False, feats, _dow_dummies(idx))


def test_training_rows_never_use_targets_after_fit_date() -> None:
    rows = training_rows(fit_pos=500, h=21, window=750)
    assert rows.start == 0 and max(rows) + 21 == 500
    rows = training_rows(fit_pos=1000, h=5, window=250)
    assert min(rows) == 751 and max(rows) + 5 == 1000


def test_direct_model_is_not_affected_by_future_data() -> None:
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(0, 0.01, 300))
    sd1 = _series(y)
    y2 = y.copy()
    y2[201:] += 5.0  # change everything after the fit date
    sd2 = _series(y2)
    m1, m2 = DirectRegression("ecm"), DirectRegression("ecm")
    sd1.ecm_features, sd2.ecm_features = sd1.features, sd2.features
    m1.fit(sd1, 200, CFG, [5])
    m2.fit(sd2, 200, CFG, [5])
    x = sd1.features.iloc[[200]].to_numpy()
    assert m1.models[5].predict(x)[0] == pytest.approx(m2.models[5].predict(x)[0])


def test_random_walk_predicts_last_value() -> None:
    sd = _series(np.arange(50, dtype=float))
    assert RandomWalk().predict(sd, 30, [1, 5]) == {1: 30.0, 5: 30.0}


def test_arima_on_returns_integrates_to_log_level() -> None:
    rng = np.random.default_rng(1)
    y = np.cumsum(0.002 + rng.normal(0, 0.01, 300))  # drift 0.002 per day
    sd = _series(y)
    m = ArimaModel()
    m.fit(sd, 250, CFG)
    p = m.predict(sd, 250, [1, 5])
    # The level forecast is the current log level plus the summed return forecasts.
    ret_fc = np.asarray(m.res.forecast(5))
    assert p[5] == pytest.approx(y[250] + ret_fc.sum())
    assert p[1] == pytest.approx(y[250] + ret_fc[0])
    # Daily state update keeps working past the fit date.
    p2 = m.predict(sd, 255, [1])
    assert np.isfinite(p2[1])


def test_walk_forward_output_alignment() -> None:
    rng = np.random.default_rng(2)
    sd = _series(np.cumsum(rng.normal(0, 0.01, 200)))
    sd.ecm_features = None
    fc = walk_forward(sd, CFG, seed=0)
    rw = fc[(fc["model"] == "random_walk") & (fc["h"] == 5)]
    assert (rw["y_pred"] == rw["y_origin"]).all()
    assert ((rw["target_date"] - rw["origin"]).dt.days >= 5).all()
    assert rw["origin"].min() == sd.y.index[CFG["train_window"]]
    assert set(fc["model"]) == {"random_walk", "arima", "ets", "gbm"}


def test_diebold_mariano_identical_errors_gives_zero() -> None:
    e = np.random.default_rng(3).normal(size=500)
    stat, p_two, p_better = diebold_mariano(e, e.copy() + 0.0, h=1)
    assert stat == 0 or np.isnan(stat) or abs(stat) < 1e-12


def test_diebold_mariano_detects_better_model() -> None:
    rng = np.random.default_rng(4)
    truth_noise = rng.normal(size=1000)
    e_good = truth_noise
    e_bad = truth_noise + rng.normal(0, 1.0, 1000)
    stat, p_two, p_better = diebold_mariano(e_good, e_bad, h=5)
    assert stat < 0 and p_better < 0.001 and p_two < 0.002


def test_directional_accuracy() -> None:
    o = np.zeros(6)
    true = np.array([1, -1, 1, -1, 0, 1.0])
    pred = np.array([1, -1, -1, -1, 1, 0.0])
    acc, n, p = directional_accuracy(o, true, pred)
    assert n == 4 and acc == pytest.approx(0.75)


def test_evaluate_marks_random_walk_as_benchmark() -> None:
    rng = np.random.default_rng(5)
    sd = _series(np.cumsum(rng.normal(0, 0.01, 200)))
    sd.ecm_features = None
    fc = walk_forward(sd, CFG, seed=0)
    m = evaluate(fc, {"x": sd}, 0.05)
    rw = m[m["model"] == "random_walk"]
    assert (rw["verdict"] == "benchmark").all() and np.allclose(rw["rmse_ratio_vs_rw"], 1.0)


def test_ablation_groups_match_linear_feature_columns() -> None:
    from gasrisk.forecasting import ABLATION_GROUPS, build_series

    idx = pd.bdate_range("2020-01-01", periods=120)
    rng = np.random.default_rng(6)
    panel = pd.DataFrame({
        "ttf_risk_ret": rng.normal(0, 0.03, 120), "ttf_c2": 30 + rng.normal(0, 1, 120).cumsum().clip(-20, None),
        "eurpln": 4.3 + rng.normal(0, 0.01, 120), "tge_pln": 150 + rng.normal(0, 3, 120).cumsum().clip(-100, None),
    }, index=idx)
    panel["tge_eur"] = panel["tge_pln"] / panel["eurpln"]
    panel["basis_tge_ttfc2_eur"] = panel["tge_eur"] - panel["ttf_c2"]
    by = {s.name: s for s in build_series(panel)}
    for name, groups in ABLATION_GROUPS.items():
        cols = [c for _, g in groups for c in g]
        assert cols == list(by[name].ecm_features.columns)  # ablation covers exactly the full linear model


@pytest.mark.parametrize("h", [1, 5, 21])
def test_dm_matches_hand_computation_with_h_minus_1_lags_and_hln(h: int) -> None:
    from scipy import stats

    from gasrisk.forecasting import diebold_mariano_full, hln_factor

    rng = np.random.default_rng(h)
    e1 = rng.normal(size=400)
    e2 = e1 + rng.normal(0, 0.7, 400)
    d = e1**2 - e2**2
    n, dc = len(d), d - d.mean()
    acov = [np.dot(dc[k:], dc[: n - k]) / n for k in range(h)]  # lags 0..h-1
    lrv = acov[0] + 2 * sum(acov[1:])
    expected = d.mean() / np.sqrt(lrv / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    r = diebold_mariano_full(e1, e2, h)
    assert r["stat"] == pytest.approx(expected)
    assert r["hac_lags"] == h - 1
    assert r["p_better"] == pytest.approx(stats.t.cdf(expected, df=n - 1))
    assert hln_factor(n, 1) == pytest.approx(np.sqrt((n - 1) / n))


def test_dm_bartlett_kernel_weights() -> None:
    from gasrisk.forecasting import diebold_mariano_full

    rng = np.random.default_rng(1)
    e1, e2 = rng.normal(size=300), rng.normal(size=300)
    r1 = diebold_mariano_full(e1, e2, 1, kernel="bartlett")
    r2 = diebold_mariano_full(e1, e2, 1, kernel="rectangular")
    assert r1["stat"] == pytest.approx(r2["stat"])  # identical when h=1 (no lags)
    with pytest.raises(ValueError):
        diebold_mariano_full(e1, e2, 5, kernel="parzen")


def test_holm_matches_statsmodels() -> None:
    from statsmodels.stats.multitest import multipletests

    from gasrisk.forecasting import holm

    p = np.array([0.0022, 0.0024, 0.03, 0.2, 0.5, 0.01])
    np.testing.assert_allclose(holm(p), multipletests(p, method="holm")[1])
