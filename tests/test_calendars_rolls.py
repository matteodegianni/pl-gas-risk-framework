"""Tests for TTF expiry dates and roll-adjusted returns."""

import numpy as np
import pandas as pd
import pytest

from gasrisk.data.calendars import roll_schedule, ttf_expiry, uk_business_days
from gasrisk.data.quality import IssueLog
from gasrisk.data.rolls import roll_adjusted_returns


@pytest.mark.parametrize(
    "month, expected",
    [
        ("2026-11", "2026-10-29"),  # 1 Nov is a Sunday: count Fri 30, Thu 29
        ("2026-12", "2026-11-27"),  # 1 Dec is a Tuesday: Mon 30 Nov, Fri 27 Nov
        ("2027-01", "2026-12-30"),  # 1 Jan is a Friday: Thu 31, Wed 30 Dec
        ("2026-09", "2026-08-27"),  # Mon 31 Aug is a UK bank holiday and is skipped
        ("2026-06", "2026-05-28"),  # 1 Jun is a Monday: Fri 29, Thu 28 May
    ],
)
def test_ttf_expiry_two_uk_business_days_before_delivery(month: str, expected: str) -> None:
    assert ttf_expiry(pd.Period(month, "M")) == pd.Timestamp(expected)


def test_uk_business_days_exclude_bank_holidays() -> None:
    days = uk_business_days("2026-12-21", "2027-01-04")
    for hol in ["2026-12-25", "2026-12-28", "2027-01-01"]:
        assert pd.Timestamp(hol) not in days
    assert pd.Timestamp("2026-12-24") in days


def test_roll_date_is_first_observed_trading_day_after_expiry() -> None:
    # The venue trades on the UK holiday 31 Aug 2026; the roll is still the first day after expiry (27 Aug).
    trading = pd.bdate_range("2026-08-03", "2026-09-30")
    sched = roll_schedule(trading)
    row = sched[sched["expiring_month"] == "2026-09"].iloc[0]
    assert row["expiry"] == pd.Timestamp("2026-08-27")
    assert row["roll_date"] == pd.Timestamp("2026-08-28")


def _toy_pair() -> tuple[pd.Series, pd.Series, pd.DatetimeIndex]:
    idx = pd.bdate_range("2024-01-01", periods=5)
    # Day 3 is the roll: the new front month (c1) is yesterday's second month.
    c1 = pd.Series([10.0, 10.1, 11.0, 11.1, 11.2], index=idx)
    c2 = pd.Series([11.0, 11.1, 12.0, 12.1, 12.2], index=idx)
    return c1, c2, pd.DatetimeIndex([idx[2]])


def test_roll_day_return_uses_same_contract() -> None:
    c1, c2, rolls = _toy_pair()
    out = roll_adjusted_returns(c1, c2, rolls, roll_z=4.0, scale_window=60)
    d = rolls[0]
    expected = np.log(11.0 / 11.1)
    assert out.at[d, "r_roll"] == pytest.approx(expected)
    assert out.at[d, "r_c1_adj"] == pytest.approx(expected)
    assert out.at[d, "r_c2_adj"] == pytest.approx(expected)
    # Raw returns that day contain the calendar spread and differ from the adjusted one.
    assert out.at[d, "r_c1_raw"] == pytest.approx(np.log(11.0 / 10.1))
    # Non-roll days are untouched.
    nd = out.index[3]
    assert out.at[nd, "r_c2_adj"] == pytest.approx(np.log(12.1 / 12.0))


def test_misaligned_roll_is_excluded_and_logged() -> None:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-01", periods=80)
    c2 = pd.Series(50 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    c1 = c2 * 0.98
    roll = idx[70]
    c1.iloc[70] = c2.iloc[69] * 1.5  # implausible 40% same-contract jump: venues rolled on different days
    log = IssueLog()
    out = roll_adjusted_returns(c1, c2, pd.DatetimeIndex([roll]), roll_z=4.0, scale_window=60, log=log)
    assert out.at[roll, "roll_status"] == "excluded_misaligned"
    assert np.isnan(out.at[roll, "r_c2_adj"]) and np.isnan(out.at[roll, "r_c1_adj"])
    assert "roll_misaligned" in log.to_frame()["check"].tolist()


def test_roll_without_c1_is_excluded() -> None:
    c1, c2, rolls = _toy_pair()
    c1 = c1.drop(rolls)
    out = roll_adjusted_returns(c1, c2, rolls, roll_z=4.0, scale_window=60)
    assert out.at[rolls[0], "roll_status"] == "excluded_no_c1"
    assert np.isnan(out.at[rolls[0], "r_c2_adj"])
