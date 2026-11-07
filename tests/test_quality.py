"""Tests for data quality checks and the issue log."""

import numpy as np
import pandas as pd
import pytest

from gasrisk.data.build import _asof
from gasrisk.data.quality import (
    IssueLog,
    check_missing_days,
    check_ohlc,
    classify_outliers,
    contiguous_runs,
    handle_outliers,
    rolling_robust_scale,
)

QCFG = {"outlier_window": 60, "outlier_min_periods": 20, "outlier_z": 6.0,
        "confirm_ratio": 0.5, "contradict_ratio": 0.2, "reversal_ratio": 0.5}


def test_contiguous_runs_group_adjacent_positions() -> None:
    ref = pd.bdate_range("2024-01-01", periods=10)
    runs = contiguous_runs(ref[[1, 2, 3, 6, 8, 9]], ref)
    assert [(s, e, n) for s, e, n in runs] == [(ref[1], ref[3], 3), (ref[6], ref[6], 1), (ref[8], ref[9], 2)]


def test_issue_log_rejects_unknown_severity() -> None:
    with pytest.raises(ValueError):
        IssueLog().add("s", "x", "c", "fatal", "a", "r")


def test_missing_days_logged_as_runs() -> None:
    expected = pd.bdate_range("2024-01-01", periods=10)
    observed = expected.delete([3, 4, 5, 8])
    log = IssueLog()
    missing = check_missing_days(observed, expected, "src", "x", log, "test", long_run=3)
    assert len(missing) == 4
    df = log.to_frame()
    assert df["n_obs"].tolist() == [3, 1]
    assert df["severity"].tolist() == ["warning", "info"]


def test_ohlc_flags_settlement_only_and_outside_range() -> None:
    idx = pd.bdate_range("2024-01-01", periods=3)
    df = pd.DataFrame({"open": [10, 10, 10], "high": [10, 11, 11], "low": [10, 9, 9], "close": [10, 10.5, 11.5]}, index=idx, dtype=float)
    log = IssueLog()
    check_ohlc(df, "src", "x", log)
    checks = log.to_frame()["check"].tolist()
    assert checks == ["settlement_only_row", "close_outside_high_low"]


def test_robust_scale_has_no_look_ahead() -> None:
    x = pd.Series(np.r_[np.random.default_rng(1).normal(0, 0.01, 100), 1.0])
    scale = rolling_robust_scale(x, window=60, min_periods=20)
    assert scale.iloc[-1] < 0.02  # the spike itself does not inflate the scale used to judge it


def _series_with_spike(reverse: bool) -> pd.Series:
    rng = np.random.default_rng(2)
    r = rng.normal(0, 0.01, 120)
    r[100] = 0.30
    if reverse:
        r[101] = -0.29
    return pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2023-01-02", periods=120))


def test_outlier_classification_confirmed_contradicted_unconfirmed() -> None:
    p = _series_with_spike(reverse=True)
    r = np.log(p / p.shift(1))
    z = (r - r.rolling(60, min_periods=20).median().shift(1)) / rolling_robust_scale(r, 60, 20)
    d = p.index[100]
    confirm = pd.Series({d: 0.25})
    contradict = pd.Series({d: 0.001})
    kw = dict(z_threshold=6.0, confirm_ratio=0.5, contradict_ratio=0.2, reversal_ratio=0.5)
    status = lambda others: classify_outliers(r, z, others, **kw).set_index("date").at[d, "status"]  # noqa: E731
    assert status({"other": confirm}) == "confirmed"
    assert status({"other": contradict}) == "contradicted"
    assert status({}) == "unconfirmed"


def test_only_contradicted_bad_prints_are_removed() -> None:
    p = _series_with_spike(reverse=True)
    d = p.index[100]
    log = IssueLog()
    cleaned, _ = handle_outliers(p, {"other": pd.Series({d: 0.0})}, QCFG, "src", "x", log)
    assert np.isnan(cleaned[d])
    p2 = _series_with_spike(reverse=False)  # same spike, no reversal: presumed real
    cleaned2, _ = handle_outliers(p2, {"other": pd.Series({d: 0.0})}, QCFG, "src", "x", IssueLog())
    assert cleaned2[d] == p2[d]


def test_asof_respects_carry_limit() -> None:
    s = pd.Series([4.0, 4.1], index=pd.to_datetime(["2024-01-01", "2024-01-10"]))
    idx = pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-05", "2024-01-10"])
    vals, age = _asof(s, pd.DatetimeIndex(idx), max_days=3)
    assert vals.tolist()[:2] == [4.0, 4.0]
    assert np.isnan(vals.iloc[2])  # 4 days old exceeds the limit
    assert age.tolist()[1] == 2 and vals.iloc[3] == 4.1
